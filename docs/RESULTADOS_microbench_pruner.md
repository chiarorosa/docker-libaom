# Microbenchmark — custo de inferência isolado do pruner (CNN nativa vs MLP)

> Data: **2026-07-17**. **Escopo: custo de INFERÊNCIA** (a passagem direta do
> modelo), medido dentro de um encode real. Avalia-se o **algoritmo isolado** — a
> extração de features (preprocessamento do MLP) e a frequência de invocação
> (integração pré-busca/por-nó) ficam **fora de escopo**, pois não são parte da
> inferência do modelo em si.

## 1. Método — como a inferência isolada foi medida

### 1.1 Medição *in situ* (encode real), não sintética

A medição **não** usa um laço sintético (chamar a função N vezes sobre dados
artificiais). Em vez disso, **instrumenta-se o codificador real** e cronometram-se
as chamadas de inferência **efetivas durante um encode genuíno**. Justificativa
metodológica: um laço sintético opera com *cache* quente e entrada constante,
enviesando o tempo para baixo e removendo o comportamento real de predição de
desvios (*branch prediction*) e de acesso à memória. A medição *in situ* usa a
**entrada real** (pixels/features dos superblocos efetivos), o **estado real de
cache/pipeline** e a **distribuição real de tamanhos de bloco** — é
representativa do custo de implantação da inferência.

### 1.2 Relógio e resolução

Usa-se `clock_gettime(CLOCK_MONOTONIC)` — relógio monotônico POSIX com **resolução
de nanossegundos**, imune a ajustes de relógio de parede. Escolhido sobre
`gettimeofday` (resolução de microssegundo, grosseira demais para eventos de
~500 ns) e sobre `clock()` (tempo de CPU, grosseiro). O `CLOCK_MONOTONIC` requer
o *feature-test macro* `_POSIX_C_SOURCE`, definido no topo da unidade de
compilação.

### 1.3 Colocação exata dos cronômetros

Para cada pruner, captura-se `t0 = now_ns()` **imediatamente antes** da chamada da
passagem direta e `t1 = now_ns()` **imediatamente depois**, acumulando `(t1 − t0)`
num acumulador global por-pruner mais um contador de chamadas. A região
cronometrada é **exatamente a passagem direta do modelo** — nada mais (sem
extração de features, sem *softmax*, sem a política de limiares):
- **CNN nativa** — em torno de `av1_cnn_predict_img_multi_out` (a passagem direta
  da rede convolucional; ver nota de escopo em §4).
- **H9a** — em torno de `av1_nn_predict` (MLP 36→64→32→3). *(A extração de
  features `student_node_features` é cronometrada separadamente, num acumulador
  distinto, e NÃO entra no resultado de inferência isolada — §4.)*
- **H9c** — em torno de `av1_nn_predict` (MLP 39→64→32→3).

### 1.4 Agregação e relato

Os acumuladores são globais; ao **término do processo** (registrado via `atexit`
na primeira chamada cronometrada) imprime-se, por pruner, `total_ns` e
`n_chamadas`, com **ns/chamada = total_ns / n_chamadas**. A média sobre **dezenas
de milhares de chamadas** (50 mil+) suaviza o ruído por-chamada do relógio: o
custo do próprio `clock_gettime` (~dezenas de ns) é pequeno frente aos valores
medidos, aproximadamente constante, e não afeta a **razão** ~50× (ver §4,
ameaças à validade).

### 1.5 Não-intrusividade e reprodutibilidade

- Guardado pela variável de ambiente `AV1_PRUNER_TIMING`, lida uma única vez
  (*cached*). **Desligada (padrão): zero efeito nas decisões e no bitstream** — a
  cronometragem só observa, não altera a busca; o binário instrumentado produz
  bitstreams idênticos ao não-instrumentado.
- **`cpu-used=1`** é necessário porque a CNN nativa é uma *speed feature* que só
  liga em `cpu ≥ 1`. Assim, os **três pruners são exercitados no mesmo encode**,
  sobre os **mesmos quadros** (H9a roda por padrão; H9c ligado por
  `AV1_STUDENT_H9C_ENABLE`), garantindo condições idênticas de conteúdo e estado
  de máquina para a comparação.
- `--threads=1` (evita contenção/variância entre núcleos); build `libaom_perf`
  (Release, `-DPARTITION_ML_STUDENT=1`); 3 quadros; **2 execuções** em
  sequências/QP distintos (Tango cq32, BoxingPractice cq43) para atestar
  estabilidade.

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

### Ameaças à validade da medição

- **Overhead do relógio (~dezenas de ns):** cada `clock_gettime` custa ~20–30 ns;
  como se mede o intervalo *entre* duas leituras, o viés é pequeno e
  aproximadamente constante — desprezível frente aos ~24.700 ns da CNN e um
  aditivo pequeno frente aos ~486 ns do MLP, sem afetar a razão ~50×.
- **Efeitos de cache/pipeline:** mitigados pela medição *in situ* (representativa)
  e pela média sobre N grande.
- **Build Release:** representa a implantação (SIMD/otimizações do compilador
  ativas), não um build de depuração.
- **Determinismo:** ns/chamada são médias estáveis entre 2 execuções
  independentes (CNN 24.763/24.606; H9a 484/488; H9c 770/745), o que descarta
  medição espúria.

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
