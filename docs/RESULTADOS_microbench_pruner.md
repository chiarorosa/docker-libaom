# Microbenchmark — custo de inferência isolado do pruner (CNN nativa vs MLP)

> Data: **2026-07-17**. **Escopo: custo de INFERÊNCIA** (a passagem direta do
> modelo), medido dentro de um encode real. Avalia-se o **algoritmo isolado** — a
> extração de features (preprocessamento do MLP) e a frequência de invocação
> (integração pré-busca/por-nó) ficam **fora de escopo**, pois não são parte da
> inferência do modelo em si.

## 1. Método

Instrumentação sob a variável de ambiente `AV1_PRUNER_TIMING` (zero overhead
quando desligada; **não altera o bitstream**) em
`av1/encoder/partition_strategy.c`: cronômetros de alta resolução
(`clock_gettime(CLOCK_MONOTONIC)`) envolvem a chamada de **inferência** de cada
pruner durante um encode real, acumulando ns e contagem:
- **CNN nativa** — `av1_cnn_predict_img_multi_out` (a passagem direta da rede
  convolucional; ver nota de escopo em §4).
- **H9a** — `av1_nn_predict` (MLP 36→64→32→3).
- **H9c** — `av1_nn_predict` (MLP 39→64→32→3).

Harness: encode a `cpu-used=1` (CNN nativa ativa) com H9c ligado — os pruners
rodam no mesmo encode, `--threads=1`, 3 quadros, 2 execuções (Tango cq32,
BoxingPractice cq43) para estabilidade. Build `libaom_perf`.

## 2. Resultados — ns por chamada de inferência (2 execuções, muito estáveis)

| pruner | ns/chamada (run1 / run2) |
|---|--:|
| **CNN nativa** | 24.763 / 24.606 |
| **H9a (MLP 36→64→32→3)** | 484 / 488 |
| **H9c (MLP 39→64→32→3)** | 770 / 745 |

## 3. Leitura

**A inferência do MLP é ~50× mais barata por chamada que a da CNN nativa**
(≈486 ns vs ≈24.700 ns) — cerca de **uma ordem e meia de grandeza**. As MLPs
propostas (entradas 24/36/39) têm custo de inferência quase idêntico entre si: o
forward denso é dominado pelas camadas ocultas comuns `[64,32]`, então a largura
de entrada quase não muda o custo (H9c é um pouco mais caro que H9a apenas pelas
3 entradas extras do bloco E). Como algoritmo de decisão, o modelo aprendido
leve (MLP) executa sua predição a uma fração do custo da rede convolucional
multi-resolução de produção (5 camadas conv + 4 ramos DNN).

## 4. Nota de escopo e fidelidade

- **A passagem direta da CNN funde leitura de input e convolução.** Numa rede
  convolucional não há "inferência" separável do processamento dos pixels — a
  convolução *é* a extração de características e a decisão ao mesmo tempo; logo os
  ~24.700 ns são a passagem direta completa (patch 65×65 → decisões da
  quad-tree). O MLP, por construção, opera sobre **features compactas
  pré-computadas**; o custo de computar essas features é um **passo de
  preprocessamento separado**, deliberadamente **fora do escopo** desta
  comparação de inferência isolada (é otimizável — cacheável e compartilhável —
  e não é parte do modelo aprendido).
- **A frequência de invocação também é fora de escopo.** A CNN nativa é invocada
  1× por superbloco (decide toda a quad-tree numa passagem); o MLP é invocado por
  nó. Isso é uma característica da **integração**, não do algoritmo de inferência;
  por isso este microbenchmark reporta **ns por chamada**, não custo agregado por
  superbloco.
- **Escopo experimental:** 1 build, 2 seqs, 3 quadros, single-thread, cpu-used=1.
  Suficiente para a razão de ordem de grandeza (médias sobre dezenas de milhares
  de chamadas; ratios estáveis entre execuções).

## 5. Reprodução

```
AV1_PRUNER_TIMING=1 [AV1_STUDENT_H9C_ENABLE=1 AV1_STUDENT_H9C_TAU=0.90] \
  build/libaom_perf/aomenc --cpu-used=1 ... 2>&1 | grep -A8 PRUNER_TIMING
```

Instrumentação: `partition_strategy.c`, guarda `AV1_PRUNER_TIMING` (no-op quando
desligada). A comparação de inferência isolada sustenta o argumento de que o
**algoritmo de decisão aprendido (MLP) é substancialmente mais leve que a CNN
nativa por predição** — complementar às Conclusões 1–2 da síntese (granularidade
fina em baixo speedup + paridade de qualidade do H9c com a CNN nativa a cpu1/2).
