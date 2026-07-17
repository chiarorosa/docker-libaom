# Microbenchmark — custo computacional isolado do pruner (CNN nativa vs MLP)

> Data: **2026-07-17**. Mede o custo de inferência isolado dos pruners de
> particionamento intra, **dentro de um encode real** (frequência e preparação de
> input reais), para verificar — ou refutar — o argumento de "custo computacional"
> das soluções ML frente à CNN nativa. **Resultado: REFUTA a alegação de "ordens
> de grandeza mais barato".**

## 1. Método

Instrumentação sob a variável de ambiente `AV1_PRUNER_TIMING` (zero overhead
quando desligada) em `av1/encoder/partition_strategy.c`: cronômetros de alta
resolução (`clock_gettime(CLOCK_MONOTONIC)`) envolvem as chamadas **reais**
durante um encode, acumulando ns e contagem, impressos ao final:
- **CNN nativa** — `av1_intra_mode_cnn_partition` (a chamada de
  `av1_cnn_predict_img_multi_out`, all-in: leitura do input 65×65 + convolução).
- **H9a inferência** — `av1_nn_predict` (MLP 36→64→32→3).
- **H9a features** — `student_node_features` (cálculo dos 36 atributos), medido
  **separado** da inferência.
- **H9c inferência** — `av1_nn_predict` (MLP 39→64→32→3).

Harness: um encode a `cpu-used=1` (onde a CNN nativa está ativa) com H9c ligado —
os três pruners rodam no mesmo encode, contagens e ns diretamente comparáveis.
Build `libaom_perf`; `--threads=1`; 3 quadros; 2 execuções (Tango cq32,
BoxingPractice cq43) para estabilidade.

## 2. Resultados (2 execuções, muito estáveis)

| pruner | ns/chamada (run1 / run2) | chamadas/encode |
|---|--:|--:|
| CNN nativa (all-in) | 24.763 / 24.606 | 5.940 / 5.937 (**1× por SB**) |
| H9a — inferência MLP | 484 / 488 | 58.535 / 52.410 (**~10× por SB**) |
| H9a — cálculo de features | 3.618 / 3.898 | idem |
| H9c — inferência MLP | 770 / 745 | 51.751 / 47.024 |

**Custo por superbloco (métrica implantada honesta):**

| pruner | por-chamada (all-in) | chamadas/SB | **por-SB** |
|---|--:|--:|--:|
| CNN nativa | ~24.700 ns | 1 | **~24,7 μs** |
| H9a (features + inferência) | ~4.250 ns | ~10 | **~40 μs** |

## 3. Leitura — as duas óticas discordam

1. **Por inferência isolada, o MLP é ~50× mais barato** que a CNN (≈486 ns vs
   ≈24.700 ns/chamada). Este é o único sentido em que o MLP é "muito mais barato".
2. **O custo implantado é dominado pela extração de features** — `student_node_features`
   custa ~3.760 ns/chamada, **~8× a inferência**: lê os pixels do bloco, variâncias
   por quadrante, gradientes, perfis linha/coluna, contexto do bloco pai (cópia da
   região 2n×2n) e contraste com irmãos. A CNN embute a leitura de input na sua
   própria chamada (65×65), já contabilizada nos 24,7 μs.
3. **O MLP roda por nó (~10×/SB), a CNN 1×/SB** — a CNN decide toda a quad-tree do
   superbloco numa passagem convolucional.
4. **Resultado líquido: por superbloco, o pruner MLP (~40 μs) é ~1,6× MAIS CARO
   que a CNN nativa (~24,7 μs).**

> **RETRATAÇÃO.** A afirmação anterior de que as soluções MLP têm custo
> computacional "ordens de grandeza menor" que a CNN nativa **está refutada**. A
> vantagem de custo do modelo aprendido está **confinada à inferência** (~50×/
> chamada) e **não sobrevive ao pipeline completo**: com extração de features por
> nó e invocação por nó, o custo total por superbloco é **comparável a — de fato,
> ~1,6× maior que — o da CNN nativa**.

## 4. Ressalvas (contextualizam, não desfazem)

- **Features não-otimizadas:** `student_node_features` recomputa o contexto do
  pai a cada nó; e quando H9a (pré-busca) e H9c (pós-NONE) rodam juntos, os
  atributos A+B+C são computados **duas vezes** por nó. Uma implantação de
  produção com cache/simplificação reduziria o custo de features
  substancialmente — o **custo inerente** (inferência) favorece o MLP, o **custo
  como-implementado** não.
- **FLOPs de inferência corroboram a inferência barata mas enganam no total:** em
  MACs de inferência, o MLP (~4,4 mil/nó × ~10 nós ≈ 44 mil/SB) é ~6× menor que a
  CNN (~270 mil MACs conv/SB); porém os FLOPs **ignoram a extração de features**,
  que domina o tempo de parede. Wall-clock all-in é a métrica honesta e favorece
  a CNN.
- **Escopo:** 1 build, 2 seqs, 3 quadros, single-thread, cpu-used=1. Suficiente
  para a conclusão de ordem de grandeza (médias sobre 50k+ chamadas, ratios
  estáveis entre execuções); não é um estudo de perf exaustivo.

## 5. Consequência para a tese

O argumento de valor das soluções ML **não é** custo computacional — é (i)
**granularidade fina em baixo speedup** que a escada discreta dos presets nativos
não cobre, e (ii) **paridade de qualidade** (H9c ≈ CNN nativa a cpu1/2 na
fronteira BD×tempo). A alegação de custo deve ser reformulada: *a inferência do
modelo é trivial; o gargalo é a engenharia de atributos, otimizável mas hoje
comparável/superior ao SOTA embarcado.* Ver correções em
`docs/SINTESE_resultados_metodologia.md` §4/§6 e `docs/ANDAMENTO_tese.md` §5.

Instrumentação: `partition_strategy.c` (guarda `AV1_PRUNER_TIMING`, no-op quando
desligada — não altera bitstream). Reprodução:
`AV1_PRUNER_TIMING=1 [AV1_STUDENT_H9C_ENABLE=1] build/libaom_perf/aomenc --cpu-used=1 ... 2>&1 | grep -A8 PRUNER_TIMING`.
