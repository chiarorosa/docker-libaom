# Microbenchmark — custo de inferência isolado do pruner (CNN nativa vs MLP)

> Data: **2026-07-17**. **Escopo: custo de INFERÊNCIA** (a passagem direta do
> modelo), medido dentro de um encode real. Avalia-se o **algoritmo isolado** — a
> extração de features (preprocessamento do MLP) e a frequência de invocação
> (integração pré-busca/por-nó) ficam **fora de escopo**, pois não são parte da
> inferência do modelo em si.

> ### ⚠ Correção de leitura (2026-07-19) — ler a §6 antes de citar a razão "~50×"
>
> A medição das §§1–5 está correta **para o que mede**: a passagem direta isolada.
> Ela foi reproduzida em 2026-07-19 (49,4–49,9×). O que **não** se sustenta é a
> extrapolação dessa razão para o custo do *pruner implantado*.
>
> Incluindo a extração de atributos e a frequência real de invocação — ambas
> medidas, não estimadas — o caminho do MLP custa **1,6× mais** que o da CNN
> nativa, não 50× menos. A razão agregada é **0,61–0,63×**.
>
> Nenhuma conclusão de implantação muda com isso, porque os dois caminhos são
> ≤0,32% do tempo de encode (§6.3). O que muda é a **narrativa**: "inferência mais
> barata" deixa de ser vantagem alegável da proposta.

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

---

## 6. Adendo (2026-07-19) — custo agregado do pruner implantado

### 6.1 Por que a §2 não responde à pergunta de implantação

As §§1–5 medem a passagem direta isolada e **declaram** (§4) que excluem a
extração de atributos e a frequência de invocação. A dificuldade é que a razão
resultante passou a ser citada como propriedade do *pruner*, e não do *modelo*.
São coisas diferentes: o que o codificador paga por superbloco é

    custo_pruner = (chamadas por SB) × (extração + inferência)

e nenhum dos dois fatores excluídos é pequeno.

A medição abaixo não extrapola: a instrumentação em C já acumulava
`total_ns` e contagem de chamadas por pruner, **incluindo um acumulador separado
para a extração do H9a** (`g_pt_h9a_feat`, `partition_strategy.c:156`,
alimentado em `:2128-2134`) que nunca havia sido lido. Totais sobre o encode
inteiro dispensam qualquer hipótese sobre nós por superbloco.

### 6.2 Resultado (3 quadros, cpu-used=1, `--threads=1`, build `libaom_perf`)

| | Tango cq32 | BoxingPractice cq43 |
|---|--:|--:|
| CNN nativa — chamadas | 5.940 | 5.937 |
| CNN nativa — ns/chamada | 24.800 | 24.494 |
| CNN nativa — total | 147,3 ms | 145,4 ms |
| H9a inferência — chamadas | 58.535 | 52.410 |
| H9a inferência — ns/chamada | 497 | 496 |
| H9a inferência — total | 29,1 ms | 26,0 ms |
| **H9a extração — ns/chamada** | **3.596** | **3.883** |
| **H9a extração — total** | **210,5 ms** | **203,5 ms** |
| H9c inferência — ns/chamada | 756 | 720 |

Derivados:

| Métrica | Tango | Boxing |
|---|--:|--:|
| Chamadas de MLP por chamada de CNN (nós/SB) | 9,85 | 8,83 |
| Razão **só inferência**, por chamada | 49,9× | 49,4× |
| Razão **com extração**, por chamada | 6,1× | 5,6× |
| **Razão agregada no encode** | **0,61×** | **0,63×** |
| Extração / inferência | 7,2× | 7,8× |

**Três correções factuais.**

1. **A razão "~50×" reproduz-se** (49,4–49,9×) — a medição original é sólida no
   seu escopo.
2. **A extração custa 7,2–7,8× a inferência.** É o termo dominante do MLP, e era
   exatamente o que estava fora de escopo. Com ela, a vantagem por chamada cai de
   ~50× para ~6×.
3. **A frequência inverte o sinal.** Com 8,83–9,85 nós modelados por superbloco
   contra 1 chamada de CNN, o caminho do MLP custa **1,6× mais** que o da CNN no
   encode inteiro. (Nota: 9,85 nós/SB, não os 21 do pior caso `1+4+16` — o portão
   exige unidade 64×64 inteira no quadro e `bsize ∈ {16,32,64}`, e a própria poda
   remove filhos da busca.)

### 6.3 Calibração de magnitude — a ressalva que impede o exagero

Uma razão dramática entre dois termos que são ambos desprezíveis não muda
conclusão nenhuma. Medindo contra o tempo de parede do mesmo encode:

| | Tango cq32 | BoxingPractice cq43 |
|---|--:|--:|
| Tempo de encode | 92,3 s | 70,8 s |
| Caminho da CNN | **0,16%** | **0,21%** |
| Caminho do H9a (extração+inferência) | **0,26%** | **0,32%** |

A diferença absoluta entre os dois é de ~0,1 ponto percentual do encode. Portanto:

- **Nenhum resultado de BD×tempo deste projeto precisa ser revisto.** Os speedups
  medidos vêm das **decisões de poda**, não do custo de inferência — o custo do
  pruner é ruído frente ao que ele economiza (ou desperdiça) em busca RD.
- **Mas "inferência mais barata" deixa de ser vantagem alegável.** O custo próprio
  do pruner é um pequeno **lastro líquido**, não uma fonte de ganho. Qualquer
  frase da tese que atribua parte do speedup à leveza do MLP está errada.

### 6.4 Consequências acionáveis

- **A extração é o alvo de otimização óbvio**, não a inferência: 210,5 ms contra
  29,1 ms. A região-pai 64×64 é recopiada para cada nó do superbloco
  (`student_node_features`, `partition_strategy.c:2059-2090`); um cache por
  superbloco — espelhando `part_info->cnn_output_valid` (`:207`/`:279`) —
  eliminaria a maior parte disso. Poderia levar a razão agregada de 0,61× para
  perto de 4×, mas com efeito de ~0,2 pp no tempo de encode: vale como argumento
  de engenharia, não como ganho de desempenho.
- **O custo do H9c está subestimado nos números publicados.** `student_h9c_decide`
  chama `student_node_features` em `partition_strategy.c:2163` **fora de qualquer
  cronômetro**; só `av1_nn_predict` é medido (`:2171-2173`). Os 720–756 ns
  omitem a extração inteira — que, pelo H9a, é ~3.600–3.900 ns. O custo real do
  H9c por chamada é da ordem de 4.400–4.700 ns.

### 6.5 Ressalvas

- Ambos os pruners rodam **no mesmo encode** (cpu-used=1, H9a por padrão, H9c por
  `AV1_STUDENT_H9C_ENABLE`). É o que permite a comparação casada em conteúdo e
  estado de máquina, mas significa que os totais vêm de uma execução em que os
  três coexistem — não da configuração de implantação de cada um isoladamente.
- 2 sequências, 3 quadros, 1 build, single-thread. Suficiente para razões de
  ordem de grandeza (dezenas de milhares de chamadas por acumulador, razões
  estáveis entre as duas execuções), não para uma estimativa de precisão fina.
- O tempo de parede inclui a sobrecarga da própria cronometragem
  (`clock_gettime` ~20–30 ns × ~220 mil leituras ≈ 5–7 ms, <0,01% do encode).

### 6.6 Reprodução

```bash
venv-ml/bin/python src/scripts/benchmark/microbench_pruner.py \
  --out results/benchmark/microbench/pruner_cost.csv
```
