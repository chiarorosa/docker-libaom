# Rastreabilidade — Pipeline de poda de particionamento AV1 guiada por ML

**Documento-mestre.** Amarra, num só lugar, **scripts ↔ dataset ↔ modelos ↔
resultados ↔ commits**, para que qualquer artefato seja auditável e reproduzível.
Atualizado em 2026-07-10 (branch `ml-partition-dev`).

Convenções: caminhos relativos à raiz do repositório `C:\dev\av1-docker`.
Execução sempre no container Docker `av1_bench` (mount `/workspace`), venv
`/workspace/build/venv-ml`. Documentos irmãos: `PLANO_hipoteses_experimentos.md`
(H1–H10), `PREH7_analise_alavancas.md`, `METODOLOGIA_pipeline_ML.md`,
`PLANO_H9_contribuicao_tese.md`, `PROTOCOLO_avaliacao.md` (congelado).

---

## 1. Grafo de proveniência (o que produz o quê)

```
 sequências-fonte (src/samples/*.yuv, 16 × UVG 4K)
        │
        │  libaom_logpart (LOG_PARTITION_DATA=1)  ── instrumentação em
        │  av1/encoder/partition_search.c (PartitionSample, 4144 B)
        ▼
 build_dataset.py ── convert_partition_data.py ──►  results/dataset_h9/
   (orquestra 16×4 QP,                                *.pkl (o *.bin bruto
    amostragem temporal, manifest)                     é descartado)
                                                      + manifest.csv   [DATASET]
                                                      + label_histogram.csv
        │
        │  data.py (regroup em superblocos, _denorm_uint8, contexto RD por nó)
        │  features.py (node_features_h9 = 41 atributos)
        ▼
 train.py ──►  results/models/surrogate_real/  (ConvNeXt professor)  [MODELO]
        │              │
        │              │  distill.py (destilação de conhecimento)
        │              ▼
        │      export_weights.py ──►  results/models/student_real/    [MODELO]
        │                             + av1/encoder/partition_student_weights.h
        │
        ├─ simulate_pruning.py / gate2_signal.py ──►  gate2_*.csv     [RESULTADO]
        │   (simulação oráculo, gate de sinal)
        │
        └─ surrogate_replay.py ──► h8_probs/*.bin (teto H8 via replay)
                 │
                 ▼
 h7h8_bench.py / ablation_attrib.py ──►  results/benchmark/*         [RESULTADO]
   (encoder real: libaom_perf vs libaom_perf_anchor;
    BD-rate + speedup; ablação de atribuição)
```

---

## 2. Scripts (inventário)

### 2.1 Modelagem — `src/scripts/partition_model/`
| Script | Papel | Entra | Sai |
|---|---|---|---|
| `partition_defs.py` | Fonte única: enum PARTITION_TYPE, níveis (64/32/16/8), legalidade, geometria do superbloco, matriz de custo geométrico | — | (definições) |
| `data.py` | Montagem do dataset: descoberta de pkls, split por sequência, reagrupamento em superblocos, `_denorm_uint8` (correção luma-em-branco), contexto RD por nó, guarda `assert_real_luma` | pkls | superblocos + ctx |
| `features.py` | Atributos manuais: `block_features` (18), `node_features` (24, +contexto hierárquico), `node_features_h9` (**41**: +B vizinhança +C quant/pos +D SATD +E none_rdcost), `H9_SUBSETS` | luma+ctx | vetor de atributos |
| `model.py` | Substituto ConvNeXt multinível (entrada luma+qindex 64×64) | — | (arquitetura) |
| `train.py` | Treino do substituto (F1/SPLIT-recall como gate) | dataset | `surrogate_best.pt` |
| `student.py` | MLP estudante (por tamanho), `collapse_label`, `export_nn_config` (layout `av1_nn_predict`) | — | (arquitetura) |
| `distill.py` | Destilação substituto→estudante (α·CE + KD) | surrogate+dataset | `students.pt` |
| `export_weights.py` | Exporta pesos do estudante para header C | `students.pt` | `partition_student_weights.h` |
| `simulate_pruning.py` | Simulação oráculo: métrica de **custo** (candidatos×n²), τ por nível, ação rect-off | modelo+dataset | `oracle_sim*.csv` |
| `surrogate_replay.py` | Pré-computa probs do substituto para o replay H8 | surrogate+YUV | `h8_probs/*.bin` |
| `check_feature_parity.py` | Harness de paridade C↔Python (atributos **e** probabilidades) | build+YUV | (verde/vermelho) |
| `gate2_signal.py` | **Gate 2** offline: MLP por subconjunto (variância/pixels/H9a/b/c), custo em risco casado | dataset | `gate2_*.csv` |

### 2.2 Dataset — `src/scripts/partition_dataset/`
| Script | Papel |
|---|---|
| `build_dataset.py` | Orquestrador N seqs × N QP; amostragem temporal (`--skip`); **retomável**; escreve `manifest.csv`. Lançar com venv python. |
| `convert_partition_data.py` | `.bin` → `.pkl`; parseia `PartitionSample` (4144 B); expõe luma + rótulo + **contexto RD** (above/left_bsize, neigh_avail, dc_q, none_rate/dist/rdcost) |
| `validate_partition_data.py` | Integridade + acurácia de pixels vs YUV-fonte + exportação PNG |
| `pkl_to_npz.py` | Converte `.pkl` (float32) → `.npz` uint8 para arquivamento/DOI (Zenodo); ver `ZENODO_datasheet.md`. Não executado por padrão. |
| `rebuild_manifest_stats.py` | Reconstrói as colunas estatísticas do `manifest.csv` a partir dos `.pkl` (os `.bin` são apagados na geração) e emite o histograma conjunto. Usado no reparo 4116→4144 de 2026-07-19; ver §3. | 
| `analyze_label_histogram.py` | Relatório de distribuição de rótulos a partir de `label_histogram.csv`: balanço de 3 classes por nível, dependência de CQ, orientação dentro de REST, dispersão por sequência. |

### 2.3 Avaliação — `src/scripts/benchmark/`
| Script | Papel |
|---|---|
| `bd_rate.py` | Taxa BD de Bjøntegaard |
| `run_benchmark.py` | Helpers encode/decode/PSNR-Y externo; BD-rate + speedup |
| `h7h8_bench.py` | Driver combinado H7+H8 (um âncora, vários pontos operacionais; presets safe/aggressive) |
| `ablation_attrib.py` | Ablação de atribuição: mesma política, fonte do escore = ml / variance / random |
| `analyze_ablation.py` | Comparação em **speedup casado** (interpolação); veredito de Pareto |
| `microbench_pruner.py` | Custo do pruner sob `AV1_PRUNER_TIMING`: ns/chamada **e** agregado por encode (extração + frequência de invocação), com o peso relativo ao tempo de parede. Ver `RESULTADOS_microbench_pruner.md` §6 |

### 2.3b CTC / Fase 6 — `src/scripts/fase6/`
| Script | Papel |
|---|---|
| `encode_ctc.py`, `encode_swap.py`, `encode_swap_h9c.py`, `encode_swap_combo.py`, `encode_h9adef.py`, `encode_h9c_cq20.py`, `encode_h9c_iso.py` | Drivers de encode das campanhas CTC; escrevem `raw_results.csv` por lote |
| `report_ctc.py` | Agregado da Fase 6 contra a âncora cpu0: `bdrate_per_seq.csv`, `bdrate_average.csv`, `tables.tex`. **`CONFIG_ORDER:42` é lista fixa** — configurações fora dela são ignoradas |
| `report_swap.py` | Agregado dos lotes de troca por `cpu-used` casado; descobre as configurações no CSV (antes ignorava H9c). `swap_per_seq.csv`, `swap_average.csv`, `swap_tables.tex` |
| `analyze_frontier.py` | Análise cruzada dos três lotes: decomposição por regime de CQ, TS por CQ individual, testes t pareados por sequência, fronteira de Pareto em 8 seqs, comparação das duas definições de TS. Ver `RESULTADOS_fase6_swap_h9c.md` |

### 2.4 Instrumentação C — `src/aom/av1/encoder/`
| Arquivo | Guarda | Papel |
|---|---|---|
| `partition_search.c` | `LOG_PARTITION_DATA` (padrão 0) | Logger `PartitionSample` (4144 B): luma + rótulo RD + **contexto B/C/E**; `static_assert` de layout |
| `partition_strategy.c` | `PARTITION_ML_STUDENT` (padrão 0) | Inferência do estudante (`av1_nn_predict`), política de 3 ações, baselines (variance/random), replay H8 |
| `partition_student_weights.h` | — | **Gerado** por `export_weights.py`; pesos do estudante implantado |

Compilação padrão (flags 0) verificada **byte-a-byte idêntica** ao
`src/aom_baseline` (controle cego, intocado).

---

## 3. Dataset — `results/dataset_h9/` (canônico)

- **Formato:** 64 `.pkl` (16 sequências × 4 QP), 31 GB. Os `.bin` intermediários
  foram apagados na geração (`--no-keep-bin`); o `.pkl` é o payload autoritativo.
  Luma `float32` [0,1] (= uint8/255, sem perdas; de-normalizada no carregador).
  Contexto RD por amostra (blocos B/C/E). `manifest.csv` registra a proveniência
  (caminhos, quadros, cpu-used) e as estatísticas de rótulo.
- **Geração:** `cpu-used=0` (ground truth por busca RD completa), 5 quadros com
  **amostragem temporal** por sequência, QPs cq 20/32/43/55 (base_qindex=4·cq).
- **Partição congelada** (`PROTOCOLO_avaliacao.md`), sem vazamento:

| Conjunto | Sequências | Amostras (Σ 4 QP) | das quais nós de decisão |
|---|---|---|---|
| **Teste** | Jockey (1,49 M), RaceNight (1,83 M), RiverBank (2,01 M) | 5,32 M | 1,91 M |
| **Validação** | HoneyBee (1,72 M), FlowerPan (2,32 M), Lips (1,85 M) | 5,89 M | 2,07 M |
| **Treino** | Beauty, Bosphorus, CityAlley, FlowerFocus, FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight, YachtRide | 15,76 M | 6,09 M |

- **Total:** **26,98 M** amostras (nós da árvore de particionamento), das quais
  **10,07 M são nós de decisão** (`block_dim ∈ {16,32,64}`). Os 16,91 M restantes
  (62,7%) são blocos 8×8, folhas terminais de rótulo constante NONE — medido:
  zero não-NONE em 16,91 M — excluídos do modelo (`partition_defs.py:41-54`),
  mas mantidos na árvore porque podar um 16×16 economiza exatamente seus quatro
  filhos.

> **Nota (2026-07-19).** Os totais acima eram ≈27,5 M até esta data. As colunas
> estatísticas do `manifest.csv` (`num_samples`, `base_qindex`, `dim*`, `part_*`)
> haviam sido geradas com o layout de registro antigo (4116 B) sobre registros de
> 4144 B, o que inflava `num_samples` em exatamente 4144/4116 = +0,68% e
> invalidava os histogramas. Reparado a partir dos `.pkl` por
> `src/scripts/partition_dataset/rebuild_manifest_stats.py`; original preservado em
> `manifest.csv.bak-4116`. Nenhum modelo, BD-rate ou resultado de Gate dependia
> dessas colunas — o carregador de treino lê apenas `pkl_path`, `sequence` e
> `cq_level` (`data.py:49-62`). Distribuição completa de rótulos em
> `results/dataset_h9/label_distribution.md`.
- **Reprodução:** determinística; ver §6. **Não versionado no git** (tamanho);
  vive no disco do host + volume do container.

> Datasets anteriores (`dataset/`, `dataset_new/`, `dataset_reduced_cq32/`,
> `dataset_smoke/`) eram da **era luma-em-branco** (bug corrigido em `cb9d407`) e
> foram **removidos** (§8). Não são comparáveis nem reutilizáveis.

---

## 4. Modelos treinados — `results/models/`

| Modelo | Arquivo | Origem | O que entregou | Estado |
|---|---|---|---|---|
| **surrogate_real** | `surrogate_best.pt` (107 MB) | `train.py` sobre `dataset_h9` (luma real) | Professor da destilação; teto H8 (−0,11 % BD via replay); **macro-F1 0,203**, SPLIT-recall 0,67/0,62/0,53 (64/32/16) | **canônico** |
| **student_real** | `students.pt` (51 KB) + header C | `distill.py` do surrogate_real | Estudante implantado (`av1_nn_predict`); curva de speedup 7–29 % @ 0,4–1,6 % BD | **canônico + versionado** |

`surrogate_real/metrics.csv` (última época) e `*/oracle_sim_v2.csv` guardam as
métricas de treino/simulação. O **student_real está versionado** no git
(`results/models/student_real/`); o surrogate_real (>100 MB) fica fora do git,
reproduzível por `train.py`.

> Modelos anteriores (`surrogate_fs/pre/v2/v5/…`, `student_ctx/v2/v3/v4/fs/final`,
> `smoke_*`, `cache/`) eram da era luma-em-branco e foram **removidos** (§8).

---

## 5. Resultados — `results/`

### 5.1 H7 + H8 (encoder real, Jockey held-out) — `benchmark/h7h8_real_summary.csv`
| ponto | política | taxa BD % | speedup |
|---|---|---:|---:|
| P0 | só NONE-commit | 0,25 | 1,03× |
| P_rect | + poda rect | 0,49 | 1,05× |
| P_ref | refinado/nível | 0,42 | 1,07× |
| H8 | substituto (teto) | −0,11 | 1,02× |

Sweep agressivo (`benchmark/h7h8_aggr/`): até **1,29× @ 1,62 % BD**.

### 5.2 Ablação de atribuição (NONE-commit) — `benchmark/ablation_matched.csv`
Em speedup casado, **variância domina o ML** em todos os níveis (ex.: 1,3×: ML
1,39 % vs variância 0,76 %). Resultado negativo que motivou o H9.

### 5.3 Gate 2 (H9, definitivo) — `models/gate2_final.csv` + `_sweep.csv`
cost_red% em risco casado (SPLIT-lost cap; rect_off_wrong≤5 %):
| subset | SL0,5 | SL1,0 | SL2,0 | SL3,0 |
|---|---:|---:|---:|---:|
| variância | 0,00 | 0,00 | 0,00 | 0,00 |
| pixels24 | 44,3 | 45,3 | 47,9 | 50,5 |
| **H9a** | 47,0 | 49,7 | 52,6 | 57,3 |
| H9b | 46,9 | 50,5 | 53,4 | 57,8 |
| H9c (teto) | 57,4 | 57,4 | 63,4 | 63,4 |

No lever NONE-commit isolado (relevante para tempo): pixels24 10–19 %, **H9a
16–25 %** — contexto RD grátis supera pixels ~50 % relativo. Veredito: **Gate 2
PASSOU**, cenário (a).

### 5.4 CTC Fase 6 e trocas — `benchmark/fase6*/`
| Diretório | Conteúdo | Sumário versionado |
|---|---|---|
| `fase6/` | âncora cpu0 + ml_balanced/ml_aggr + native_cpu1/2/3 (8 seqs) e pilotos H9c em cpu0 (3–6 seqs) | `bdrate_average.csv`, `bdrate_per_seq.csv`, `tables.tex` |
| `fase6_swap/` | troca H9a em cpu1/2/3, 8 seqs × 4 CQ (192 encodes) | `swap_average.csv`, `swap_per_seq.csv` |
| `fase6_swap_h9c/` | troca H9c τ=0,90/0,95 em cpu1/2/3, 8 seqs × 4 CQ (192 encodes) | `swap_average.csv`, `swap_per_seq.csv`, `swap_tables.tex` — **agregados só em 2026-07-19** |
| `fase6_swap_combo/` | piloto H9a+H9c empilhados, 1 seq | — (pilotagem) |
| `fase6_analysis/` | análise cruzada dos três lotes | `cq_decomposition.csv`, `ts_per_cq.csv`, `paired_tests.csv`, `pareto_frontier.csv`, `ts_definitions.csv` |

Resultado principal: **`RESULTADOS_fase6_swap_h9c.md`** — o H9c é substituto drop-in da CNN nativa
intra, com vantagem de BD-rate significativa em CQ 20+32 (cpu1 p=0,043, 6/8 seqs; cpu2 p=0,015,
7/8 seqs), empate na grade CTC completa e desvantagem em cpu-used=3.

### 5.5 Outros diretórios em `results/benchmark/`
`ablation_attrib/`, `ablation_fill_{ml,var,rnd}/` (curvas da ablação de
atribuição), `h7h8/`, `h7h8_aggr/`, `h7h8_real/` (runs por ponto operacional),
`h8_probs/` (arquivos de replay do substituto). CSVs `ablation_matched.csv` e
`h7h8_real_summary.csv` são os sumários versionados.

---

## 6. Pipeline de reprodução (comando a comando, no container)

```bash
# 0. Dataset (≈8 h, cpu-used=0, retomável) — venv python obrigatório
venv-ml/bin/python src/scripts/partition_dataset/build_dataset.py \
  --out-dir results/dataset_h9 --qps 20 32 43 55 --frames 5 --cpu-used 0 \
  --aomenc build/libaom_logpart/aomenc
# (libaom_logpart = src/aom configurado com -DLOG_PARTITION_DATA=1)

# 0b. Estatísticas do manifesto + histograma de rótulos (~7 min, só leitura)
venv-ml/bin/python src/scripts/partition_dataset/rebuild_manifest_stats.py \
  --dataset-dir results/dataset_h9
venv-ml/bin/python src/scripts/partition_dataset/analyze_label_histogram.py \
  --hist results/dataset_h9/label_histogram.csv \
  --out results/dataset_h9/label_distribution.md
# Verificação: em cada linha do manifesto, Σdim* = Σpart_* = num_samples.

# 1. Substituto (professor)
venv-ml/bin/python src/scripts/partition_model/train.py \
  --dataset-dir results/dataset_h9 --out-dir results/models/surrogate_real

# 2. Estudante (destilação) + export dos pesos para C
venv-ml/bin/python src/scripts/partition_model/distill.py \
  --surrogate results/models/surrogate_real/surrogate_best.pt \
  --out-dir results/models/student_real --no-class-weight --temp 1.0
venv-ml/bin/python src/scripts/partition_model/export_weights.py \
  --students results/models/student_real/students.pt

# 3b. Agregados CTC (só análise, ~1 min; nenhum encode)
venv-ml/bin/python src/scripts/fase6/report_swap.py \
  --out-dir results/benchmark/fase6_swap_h9c
venv-ml/bin/python src/scripts/fase6/analyze_frontier.py \
  --out-dir results/benchmark/fase6_analysis

# 3. Gates offline
venv-ml/bin/python src/scripts/partition_model/gate2_signal.py \
  --dataset-dir results/dataset_h9 --per-pkl 1500 \
  --train-seqs Beauty Bosphorus CityAlley FlowerFocus FlowerKids ReadySetGo \
               ShakeNDry SunBath Twilight YachtRide \
  --val-seqs HoneyBee FlowerPan Lips

# 4. Build flag-ON + paridade + no-op (ver METODOLOGIA §6)
#    libaom_perf (-DPARTITION_ML_STUDENT=1), libaom_ml_check (generic)
venv-ml/bin/python src/scripts/partition_model/check_feature_parity.py \
  --aomenc build/libaom_ml_check/aomenc \
  --yuv src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv --cq 20 \
  --students results/models/student_real/students.pt

# 5. Benchmark de tese (teste held-out) + ablação
venv-ml/bin/python src/scripts/benchmark/h7h8_bench.py \
  --seq src/samples/Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv --cpu-used 0
venv-ml/bin/python src/scripts/benchmark/ablation_attrib.py --seq <...>
venv-ml/bin/python src/scripts/benchmark/analyze_ablation.py --dirs <...>
```

---

## 7. Trilha de commits (log de auditoria, branch `ml-partition-dev`)

| Commit | Marco |
|---|---|
| `5d446e8` | Limpeza: remove manifesto do dataset antigo |
| `98cc15f` | **Preserva estado-GOAL**: scripts + student_real + descarte do defasado |
| `8866757` | Gate 2: ensemble de sementes + risco BD-relevante + lever NONE-commit |
| `1f7535d` | Gate 2 offline (`gate2_signal.py`) |
| `a6748c5` | Extração de features H9 (blocos B/C/D/E) |
| `a17c525` | Instrumentação RD-context em C (blocos B/C/E) |
| `aabbbee` | **Protocolo de avaliação congelado** (Fase 0) |
| `30393cc` | Plano de contribuição H9 |
| `2380e91` | Ablação de atribuição: variância bate o ML (resultado negativo) |
| `4c5fb9f` | Ablação de atribuição + doc de metodologia |
| `172e669` | Curva operacional H7 (7–29 % speedup) |
| `9b2f1d9` | Resultados reais H7/H8; marca tabelas blank como superadas |
| `63f299c` | Retreino sobre luma real; guarda anti-blank; pesos reais |
| `cb9d407` | **Correção do bug luma-em-branco** (de-normalização uint8) |
| `834639e` | Alavanca de poda rect + atributos hierárquicos (Pré-H7) |
| `71097b1` | Plano de hipóteses H1–H10 |
| `30ad9c8`…`e8b6cf9` | Infra inicial (surrogate, destilação, instrumentação, dataset cq32) |

O log completo (`git log`) é o registro de auditoria canônico.

---

## 8. Artefatos removidos (registro histórico)

Removidos do **tracking git** (`98cc15f`, `5d446e8`) e do **disco local** (era
luma-em-branco, cientificamente inválida — bug `cb9d407`). Reconstruíveis pelos
scripts se necessário; **não** são comparáveis ao `dataset_h9`:

- Datasets: `results/dataset/`, `results/dataset_new/`,
  `results/dataset_reduced_cq32/`, `results/dataset_smoke/`, PNGs de debug.
- Modelos: `surrogate_{fs,pre,v2,v5,'',smoke}/`,
  `student_{ctx,v2,v3,v4,fs,final,smoke}/`, `models/cache/` (.npz regenerável).
- Saídas antigas: `results/libaom_perf_baseline/`, `results/mlcheck/`.

**Preservados** (canônicos, intocados): `dataset_h9/`, `surrogate_real/`,
`student_real/`, `benchmark/`.
