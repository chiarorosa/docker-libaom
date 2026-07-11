# Spec — Fase 3: estudante tabular sobre H9a

**Data:** 2026-07-11. Branch `ml-partition-dev`. Contexto: `docs/ANDAMENTO_tese.md`
(§4), `docs/PLANO_H9_contribuicao_tese.md` (Fase 3), `docs/PROTOCOLO_avaliacao.md`
(split congelado), `docs/RASTREABILIDADE.md` (inventário de scripts).

---

## 1. Objetivo

Produzir o **estudante implantável** — MLP por tamanho de bloco, executada por
`av1_nn_predict` — treinado **diretamente** (CE de rótulo duro, sem professor)
sobre o vetor de features **H9a (36 entradas)**, a partir de `results/dataset_h9`,
respeitando o **split congelado 10/3/3**, e validá-lo no gate de simulação oráculo
contra o baseline de variância.

Isto reproduz, no artefato implantável, a configuração que o **Gate 2** já provou
vencer (MLP por tamanho sobre H9a supera a variância ~50% relativo). Ver
`ANDAMENTO_tese.md` §1.2 (o estudante é o que roda; o ConvNeXt nunca foi embarcado)
e §3 (veredito do Gate 2).

## 2. Fora de escopo

- Sincronização das features B/C em C (`student_node_features`) — **Fase 4**.
- Benchmark held-out no encoder real + ablação de atribuição — **Fase 5**.
- Retreino do ConvNeXt / ramo lateral RD — **descartado por decisão** (o professor
  pixel-only não vê B/C; o Gate 2 dispensa a destilação para o H9a).
- Destilação por professor **tabular** — refinamento opcional e posterior, só se o
  Gate 3 ficar aquém do teto H9c por margem que valha o esforço.

## 3. Split congelado (inegociável)

Conforme `PROTOCOLO_avaliacao.md`:

| Conjunto | Sequências |
|---|---|
| Treino (10) | Beauty, Bosphorus, CityAlley, FlowerFocus, FlowerKids, ReadySetGo, ShakeNDry, SunBath, Twilight, YachtRide |
| Validação (3) | HoneyBee, FlowerPan, Lips |
| Teste (3) | Jockey, RaceNight, RiverBank — **nunca tocados nesta fase** |

Os defaults atuais de `distill.py`/`simulate_pruning.py` usam `--val-seqs Jockey`,
que **agora é teste**. Todos os comandos desta fase passam os splits explicitamente.

## 4. Conjunto de features

`H9a = features.node_features_h9(...)[:36]`:
- **A (0..23):** pixels (`node_features`; idx 0 = `log1p(var)` = o baseline de
  variância). Já com paridade C↔Python verificada.
- **B (24..31):** contexto de vizinhança de particionamento (has_above/left,
  tamanhos log2 dos vizinhos, `neigh_finer`, `neigh_aniso`).
- **C (32..35):** `log_dc_q2`, `pos_row`, `pos_col`, `depth_log2`.

D (SATD) e E (`none_rdcost`) **não entram** no modelo implantado.

**Nota técnica:** dentro de um modelo por tamanho de bloco, `depth_log2` (idx 35) é
**constante** → desvio-padrão ≈ 0 → clampeado ao piso 0.1 na padronização → vira uma
entrada inócua (contribuição ≈ 0). Comportamento idêntico ao de `q_norm` dentro de
um único QP no estudante de pixels; **não requer tratamento especial**.

## 5. Componentes e mudanças

Alterações pequenas e bem delimitadas. Princípio: **tornar os scripts a jusante
cientes da contagem de features via o bundle**, em vez de assumir 24 fixo.

### 5.1 `features.py`
- Adicionar constante `NUM_FEATURES_H9A = 36`. `node_features_h9` e `H9_SUBSETS`
  já existem — sem mais mudanças.

### 5.2 `student.py`
- Sem mudança. `make_student(in_features, hidden)` já parametriza a entrada.

### 5.3 `distill.py` (mudança mínima, para reúso)
- Parametrizar `train_student(..., in_features)` em vez do hardcode
  `featmod.NUM_FEATURES` (linha 111). O `main()` de destilação continua passando
  `NUM_FEATURES`. Assim a lógica de treino — incluindo o **fold-in da padronização
  na 1ª camada** (para o C receber features cruas) — fica reutilizável pela Fase 3.

### 5.4 `train_student_h9.py` (novo)
Treinador direto do estudante implantável. Fluxo:
1. `discover_pkls` + `split_entries(val_seqs, train_seqs)` + `assert_real_luma`
   (treino) — reusa `data.py`.
2. Por `dim` em `MODEL_LEVELS`: coletar `(feat = node_features_h9(...)[:36]` com o
   `ctx` por nó vindo de `iter_superblock_members`, `label = collapse_label(truth))`.
3. Treinar **uma** rede por tamanho via `distill.train_student(in_features=36,
   alpha=1.0, use_class_weight=False)` — `alpha=1.0` zera o termo KD (rótulo duro
   puro); `use_class_weight=False` reproduz a receita do Gate 2 / H6 (CE não
   ponderado dá P(NONE) mais nítido para o pruner por limiar); `teacher`
   preenchido com distribuição uniforme (placeholder inócuo, multiplicado por 0).
   Épocas/lr alinhados ao Gate 2 (epochs≈30, lr=1e-3).
4. Salvar bundle em `results/models/student_h9a/students.pt`:
   `{"hidden", "students":{dim:state_dict}, "norm":{dim}, "num_features":36,
   "feature_set":"h9a"}`.
5. Self-check por tamanho: contagem N/S/R + acurácia vs. rótulo.

Topologia: `hidden=[64,32]` (igual ao Gate 2 e ao estudante atual).

### 5.5 `export_weights.py` (ciente da contagem de features)
- Ler `bundle.get("num_features", featmod.NUM_FEATURES)` e usar em
  `make_student(...)`, no comentário do header e em
  `AV1_PARTITION_STUDENT_NUM_FEATURES`. Contrato explícito para a Fase 4: o C
  deverá computar **36** features.

### 5.6 `simulate_pruning.py` (modo H9a + baseline de variância)
- `collect_superblocks`: carregar `sb["ctx"]` e computar o vetor conforme
  `feature_set` do bundle — `node_features` (24) ou `node_features_h9[:36]`.
- `score_with_student`: usar `bundle["num_features"]` em `make_student`.
- Adicionar baseline de variância (regra `P(NONE)=exp(-var/v0)` sobre a feature 0,
  como em `gate2_signal.variance_threshold_probs`) para a comparação em risco
  casado do Gate 3.

## 6. Gates (não avança sem passar)

- **Gate 3a (sanidade):** por tamanho, macro-F1 / SPLIT-recall do estudante H9a
  ≥ referência do estudante de pixels (`student_real`).
- **Gate 3b (decisivo):** na **validação** (HoneyBee/FlowerPan/Lips), a redução de
  custo da simulação oráculo do estudante H9a ≥ variância em **risco casado**
  (split-lost caps {0,5 / 1 / 2}%), consistente com a margem do Gate 2.
- Só após 3a+3b: **exportar pesos** (`export_weights.py`) → header C.
  **Sem rebuild C nesta fase** (isso é Fase 4).

## 7. Reprodução (container `av1_bench`, venv `venv-ml`)

```bash
# 1. Treinar o estudante H9a (split congelado explícito)
venv-ml/bin/python src/scripts/partition_model/train_student_h9.py \
  --dataset-dir results/dataset_h9 \
  --train-seqs Beauty Bosphorus CityAlley FlowerFocus FlowerKids ReadySetGo \
               ShakeNDry SunBath Twilight YachtRide \
  --val-seqs HoneyBee FlowerPan Lips \
  --out-dir results/models/student_h9a

# 2. Gate 3b: simulação oráculo na validação (estudante vs variância)
venv-ml/bin/python src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir results/dataset_h9 --val-seqs HoneyBee FlowerPan Lips \
  --students results/models/student_h9a/students.pt --tau-rest 0.1 0.2 0.3 \
  --out-csv results/models/student_h9a/oracle_sim.csv

# 3. Exportar pesos (só se os gates passarem; sem rebuild C)
venv-ml/bin/python src/scripts/partition_model/export_weights.py \
  --students results/models/student_h9a/students.pt \
  --out src/aom/av1/encoder/partition_student_weights.h
```

## 8. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Estudante único (deploy) não reproduz a margem do Gate 2 (que usou ensemble 3-seed) | O ensemble era só de-noise de **avaliação**; o operating point ainda deve superar a variância. Se necessário, ajustar épocas/seed e escolher a semente por validação. |
| `depth_log2` constante por tamanho vira entrada morta | Inócuo (folded ≈ 0); documentado (§4). |
| Sobrescrever o header do `student_real` implantado | Exportar para `student_h9a/` e só sincronizar o header quando a Fase 4 iniciar; `student_real` permanece o implantado até lá. |
| Dataset não versionado / pesado | Rodar no container; `dataset_h9` vive no disco do host + volume. |

## 9. Entregáveis desta fase

1. `results/models/student_h9a/students.pt` (bundle com `num_features=36`).
2. `results/models/student_h9a/oracle_sim.csv` (Gate 3b, estudante vs variância).
3. Header C exportado (contrato de 36 features) — **não** compilado ainda.
4. Scripts: `train_student_h9.py` (novo); `distill.py`, `export_weights.py`,
   `simulate_pruning.py`, `features.py` (edições delimitadas acima).
