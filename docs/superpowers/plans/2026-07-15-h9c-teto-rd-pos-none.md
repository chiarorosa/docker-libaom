# H9c — poda pós-NONE aprendida (teto de contexto RD) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar, destilar e integrar em C um estudante H9c (features A+B+C+E = 39, incluindo o rdcost real do `PARTITION_NONE`) que decide, **depois** que o encoder já avaliou `PARTITION_NONE`, se vale a pena continuar buscando SPLIT/retangular/AB/4-way — competindo contra as heurísticas pós-NONE nativas (`early_term_after_none_split`, `skip_non_sq_part_based_on_none`), que são regras artesanais, não CNN. Validar com um piloto de tempo real barato antes de gastar o benchmark CTC.

**Architecture:** Mesmo padrão H9a (MLP por tamanho de bloco via `av1_nn_predict`, sem professor), mas o ponto de decisão muda: em vez de `av1_prune_partitions_before_search` (pré-busca), o hook fica dentro de `av1_rd_pick_partition` (`partition_search.c`), logo após `none_partition_search()` ter preenchido `part_search_state.this_rdc`/`none_rd`, e antes de `split_partition_search()`. A ação é binária (terminar a busca aqui, comitando NONE) via `av1_disable_all_splits()` — os mesmos helpers que H9a já usa, só que chamados mais tarde com informação real, não estimada.

**Tech Stack:** Python (PyTorch, treino/destilação/simulação oráculo), C (libaom v3.10.0). Build/test **exclusivamente no container** `av1_bench`; edição no Windows.

## Global Constraints

- **Editar no Windows, compilar/testar no Docker** `av1_bench` (`docker exec av1_bench ...`). Python: `/workspace/build/venv-ml/bin/python` (ou `build/venv-ml/bin/python` já dentro do container, conforme os planos de Fase 3/4).
- **`src/aom_baseline` é INTOCÁVEL** (controle cego; nunca ler/editar).
- **Todo código C novo sob `#if PARTITION_ML_STUDENT`** (região atual `partition_strategy.c:1548–2216`). Flag off ⇒ bitstream byte-idêntico ao `aom_baseline`. O call site novo em `partition_search.c` fica **fora** do `#if` (é uma função sempre presente cujo corpo é no-op com a flag off — mesmo padrão de `av1_prune_partitions_before_search`).
- **`features.node_features_h9c` é a fonte única de verdade** do layout de 39 features (A 0–23 + B 24–31 + C 32–35 + E 36–38). O C espelha ESSA função. **Não confundir com `H9_SUBSETS["H9c"]`** (41 features, A+B+C+D+E, usado só no Gate 2 offline/ablação — nunca implantado; D já foi descartado por decisão de Fase 3).
- **Dataset:** `results/dataset_h9/*.pkl` já tem os campos `none_rate/none_dist/none_rdcost` extraídos (Fase 1); **não re-extrair**.
- **Estilo:** clang-format Google-based no C; casar o entorno. **Commits:** sem menção a IA/Claude/Co-Authored-By. Branch `ml-partition-dev` (commit aqui; não tocar main).
- **Sem suíte pytest formal:** verificação = builds + smokes no container + harness de paridade + md5 do no-op — mesmo padrão das Fases 3/4.
- **Gate de parada:** se o Gate 3 (§Task 3) não superar claramente o H9a (~55–58% cost_red @ SL≤1%) por margem, ou se o piloto de tempo real (Task 8) não mostrar ganho sobre H9a/variância, **parar antes da Task 9** (CTC) e reportar o resultado como está — mesma disciplina de "fail-fast" das fases anteriores.

---

### Task 1: Python — `node_features_h9c` (39 features, fonte única de verdade)

**Files:**
- Modify: `src/scripts/partition_model/features.py`

**Interfaces:**
- Consumes: `node_features_h9` (já existe, 41-dim, `features.py:263`), `NUM_FEATURES_H9A` (36, já existe).
- Produces: `NUM_FEATURES_H9C = 39`; `node_features_h9c(sb_luma, dim, r, c, qindex, ctx) -> np.ndarray[39]`. Consumido pelas Tasks 2, 3 e pelo harness de paridade C↔Python (Task 6).

- [ ] **Step 1: Adicionar a constante e a função, logo após `node_features_h9a`**

Em `features.py`, após a função `node_features_h9a` (hoje terminando por volta da linha 322 com `return node_features_h9(...)[:NUM_FEATURES_H9A]`), inserir:

```python
NUM_FEATURES_H9C = 39  # A(0..23) + B(24..31) + C(32..35) + E(36..38); D dropped
                       # (Fase 3 decision: SATD does not add signal over H9a).


def node_features_h9c(sb_luma, dim, r, c, qindex, ctx):
    """H9c deploy vector (A+B+C+E = 39 features): the H9a vector plus the real
    PARTITION_NONE rate/dist/rdcost, only available AFTER the encoder has
    evaluated PARTITION_NONE for this node (post-NONE decision point, not
    pre-search). Distinct from H9_SUBSETS['H9c'] (41-dim, includes D, offline
    Gate-2 ablation only -- never deployed). Single source of truth; the C side
    (student_h9c_prune_partition in partition_strategy.c) mirrors THIS."""
    c2 = dict(ctx)
    c2["bsize_enum"] = -1
    full = node_features_h9(sb_luma, dim, r, c, qindex, c2)  # 41-dim
    return np.concatenate([full[:36], full[38:41]]).astype(np.float32)
```

- [ ] **Step 2: Smoke — rodar o `__main__` do módulo e checar a forma**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python -c "
import sys; sys.path.insert(0, \"src/scripts/partition_model\")
import numpy as np, features as f
rng = np.random.default_rng(0)
sb = rng.integers(0, 256, size=(64,64), dtype=np.uint8)
ctx = {\"neigh_avail\":3, \"above_bsize\":9, \"left_bsize\":9, \"dc_q\":32,
       \"mi_row\":8, \"mi_col\":8, \"frame_w\":256, \"frame_h\":256,
       \"none_rate\":1200, \"none_dist\":800, \"none_rdcost\":50000}
v = f.node_features_h9c(sb, 32, 0, 0, 100, ctx)
print(\"shape\", v.shape, \"dtype\", v.dtype, \"finite\", np.all(np.isfinite(v)))
assert v.shape == (39,)
print(\"OK\")
"'
```
Expected: `shape (39,) dtype float32 finite True` seguido de `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/scripts/partition_model/features.py
git commit -F- <<'EOF'
H9c: add node_features_h9c (39-feat deploy vector, A+B+C+E)

Single source of truth for the post-NONE deploy feature layout: the H9a
36-feat vector plus the real PARTITION_NONE rate/dist/rdcost (block E),
skipping D per the Fase 3 decision (SATD proxy did not add signal). Distinct
from the offline H9_SUBSETS['H9c'] (41-dim, includes D, Gate-2 ablation only).
EOF
```

---

### Task 2: Python — `train_student_h9c.py` (treino do estudante implantável)

**Files:**
- Create: `src/scripts/partition_model/train_student_h9c.py` (fork de `train_student_h9.py`)

**Interfaces:**
- Consumes: `features.node_features_h9c`, `features.NUM_FEATURES_H9C`; `results/dataset_h9/*.pkl` (campo `has_rd` + `none_rate/none_dist/none_rdcost` já extraídos); `distill.train_student`, `student.make_student`.
- Produces: `results/models/student_h9c/students.pt` (bundle com `num_features=39`, `feature_set="h9c"`), consumido pelas Tasks 3 e 4.

- [ ] **Step 1: Copiar `train_student_h9.py` para `train_student_h9c.py` e trocar as 4 diferenças**

Run:
```bash
cp src/scripts/partition_model/train_student_h9.py src/scripts/partition_model/train_student_h9c.py
```

Editar `train_student_h9c.py`:

1. Docstring (topo do arquivo) — trocar para:
```python
"""H9c -- train the deployable per-block-size student on the post-NONE
feature vector (A pixels + B neighbor context + C quant/position + E real
PARTITION_NONE rate/dist/rdcost = 39 features), hard-label CE, no teacher.
Unlike H9a, this student is only valid AFTER PARTITION_NONE has been
evaluated in the live encoder (see partition_search.c:av1_rd_pick_partition,
right after none_partition_search()) -- it decides whether to keep searching
SPLIT/rect/AB/4-way, not whether to search NONE at all.
"""
```

2. Na função `collect_by_dim`, trocar a chamada de features (era `featmod.node_features_h9a(...)`):
```python
                f = featmod.node_features_h9c(sb["luma"], dim, r, c,
                                              sb["qindex"], sb["ctx"][k])
```

3. No `main`, trocar o default de `--out-dir`:
```python
    p.add_argument("--out-dir", default="/workspace/results/models/student_h9c")
```

4. Mais abaixo no `main`, trocar `nfa = featmod.NUM_FEATURES_H9A` por:
```python
    nfa = featmod.NUM_FEATURES_H9C
```
E, no dict `bundle` (mesma linha onde `"hidden"`/`"students"`/`"norm"` são setados), confirmar que `bundle["num_features"] = nfa` e `bundle["feature_set"] = "h9c"` são gravados (no `train_student_h9.py` original essas duas chaves já são setadas a partir de `nfa`/uma string fixa "h9a" — trocar essa string fixa para `"h9c"`).

- [ ] **Step 2: Treinar (10 seqs de treino, split congelado)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/train_student_h9c.py \
  --dataset-dir results/dataset_h9 --out-dir results/models/student_h9c \
  --epochs 30 2>&1 | tail -30'
```
Expected: termina sem exceção; imprime `train pkls: 40 (train seqs [...])` (10 seqs × 4 QP) e `collected N superblocos`; arquivo `results/models/student_h9c/students.pt` criado.

- [ ] **Step 3: Commit**

```bash
git add src/scripts/partition_model/train_student_h9c.py
git commit -F- <<'EOF'
H9c: train the deployable post-NONE student (39 features, hard-label CE)

Forks train_student_h9.py, swapping the feature function to
node_features_h9c (adds block E: real PARTITION_NONE rate/dist/rdcost) and
the output dir/feature_set tag. Trained bundle -> results/models/student_h9c/
(git-ignored, like all results/models/ except student_real).
EOF
```

---

### Task 3: Python — Gate 3 (validação oráculo em HoneyBee/FlowerPan/Lips)

**Files:**
- Modify: `src/scripts/partition_model/simulate_pruning.py`

**Interfaces:**
- Consumes: `results/models/student_h9c/students.pt` (Task 2); `features.node_features_h9c`.
- Produces: `cost_red%` em risco casado (SPLIT-lost) para H9c na validação — o número que decide se a Task 5+ (integração C) vale a pena.

- [ ] **Step 1: Adicionar o branch `"h9c"` em `collect_superblocks`**

Em `simulate_pruning.py`, a função `collect_superblocks` (linha ~66) só entende `feature_set in {"pixels24", "h9a"}` (linha ~73–79). Adicionar, logo após o bloco `if feature_set == "h9a":`:

```python
            elif feature_set == "h9c":
                if not sb.get("has_rd"):
                    raise SystemExit(
                        "{}: no RD context; re-extract with the H9 "
                        "instrumentation.".format(entry["path"]))
                feat = featmod.node_features_h9c(sb["luma"], dim, r, c,
                                                 sb["qindex"], sb["ctx"][k])
```
(mesma indentação/estrutura do branch `"h9a"` já existente logo acima; ajustar o nome da variável de loop para casar com o que já está em uso no arquivo — conferir o nome exato antes de colar, ex. `entry`/`e`.)

- [ ] **Step 2: Rodar a simulação oráculo na validação (HoneyBee/FlowerPan/Lips)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir results/dataset_h9 \
  --students results/models/student_h9c/students.pt \
  --val-seqs HoneyBee FlowerPan Lips \
  --tau-none 0.80 0.90 0.95 0.97 0.99 \
  --max-split-lost 1.0 \
  --out-csv results/models/student_h9c/gate3_h9c.csv 2>&1 | tail -40'
```
Expected: imprime a tabela `STUDENT ... (feature_set=h9c)` com `cost_red%` por ponto de `tau_none`, risco (SPLIT-lost) casado a ≤1%.

- [ ] **Step 3: Comparar contra o H9a já registrado (Gate 3 anterior)**

Sem script novo — comparar manualmente o `cost_red%` @ SL≤1% do CSV gerado contra os números já registrados em `docs/ANDAMENTO_tese.md` §4 (H9a: ~55–58% @ SL≤1%) e o teto do Gate 2 offline (H9c: 33–63%, `gate2_final.csv`). **Critério de decisão:**
- Se H9c (Gate 3, validação) **superar claramente** H9a (~55–58%) → prosseguir para a Task 4 (integração C).
- Se H9c **empatar ou ficar abaixo** → parar aqui, registrar em `ANDAMENTO_tese.md` que o teto offline não se sustenta na validação de tamanho real (fine, ainda é um resultado honesto: "o ganho teórico do rdcost real do NONE não sobrevive ao gate de validação"), e não integrar em C.

- [ ] **Step 4: Commit**

```bash
git add src/scripts/partition_model/simulate_pruning.py results/models/student_h9c/gate3_h9c.csv
git commit -F- <<'EOF'
H9c: Gate 3 validation (HoneyBee/FlowerPan/Lips oracle sim)

Adds the h9c feature_set branch to collect_superblocks (node_features_h9c,
requires has_rd). Gate 3 result recorded in gate3_h9c.csv for comparison
against the H9a Gate 3 baseline (~55-58% cost_red @ SL<=1%).
EOF
```

---

### Task 4: Python — header C do H9c (coexistindo com o header H9a implantado)

**Files:**
- Modify: `src/scripts/partition_model/export_weights.py`
- Create: `src/aom/av1/encoder/partition_student_h9c_weights.h` (gerado, não editar à mão)

**Interfaces:**
- Consumes: `results/models/student_h9c/students.pt` (Task 2/3, só prossegue se o Gate 3 passou).
- Produces: header C com `AV1_PARTITION_STUDENT_H9C_NUM_FEATURES 39`, `av1_partition_student_h9c_nnconfig(BLOCK_SIZE)`. Consumido pela Task 5. **Não sobrescreve** `partition_student_weights.h` (o H9a implantado continua intocado).

O script atual (`export_weights.py`) tem os nomes de símbolo (`av1_partition_student_{dim}_nnconfig`, guard, macro `AV1_PARTITION_STUDENT_NUM_FEATURES`) **hardcoded**. Como os dois headers (H9a de 36 e H9c de 39) precisam coexistir no mesmo binário (`libaom_perf`), é preciso parametrizar o prefixo.

- [ ] **Step 1: Adicionar `--prefix` e `--macro-suffix` ao script**

Em `export_weights.py`:

1. No `main`, adicionar o argumento (perto de `--students`/`--out`):
```python
    p.add_argument("--prefix", default="av1_partition_student",
                   help="C symbol prefix (must be unique per header if two "
                        "student headers are included in the same TU)")
```

2. Trocar `emit_config` para receber e usar o prefixo:
```python
def emit_config(dim, cfg, prefix):
    """Return (arrays_text, config_text) for one block size."""
    p = "{}_{}".format(prefix, dim)
    arrays = ""
    w_names, b_names = [], []
    n_layers = cfg["num_hidden_layers"] + 1
    for k in range(n_layers):
        wn = "{}_weight_{}".format(p, k)
        bn = "{}_bias_{}".format(p, k)
        arrays += fmt_array(wn, cfg["weights"][k])
        arrays += fmt_array(bn, cfg["bias"][k])
        w_names.append(wn)
        b_names.append(bn)
    hidden = ", ".join(str(h) for h in cfg["num_hidden_nodes"]) or "0"
    conf = (
        "static const NN_CONFIG {p}_nnconfig = {{\n"
        "  {ni},  // num_inputs\n"
        "  {no},  // num_outputs\n"
        "  {nl},  // num_hidden_layers\n"
        "  {{ {hid} }},  // num_hidden_nodes\n"
        "  {{ {ws} }},\n"
        "  {{ {bs} }},\n"
        "}};\n"
    ).format(p=p, ni=cfg["num_inputs"], no=cfg["num_outputs"],
             nl=cfg["num_hidden_layers"], hid=hidden,
             ws=", ".join(w_names), bs=", ".join(b_names))
    return arrays, conf
```

3. No `main`, trocar a chamada `a, c = emit_config(dim, cfg)` por `a, c = emit_config(dim, cfg, args.prefix)`, e a montagem do `header` para usar `args.prefix` no guard, no `#define ..._NUM_FEATURES`, no `switch` e no nome da função accessor:
```python
    guard = "AOM_AV1_ENCODER_{}_WEIGHTS_H_".format(args.prefix.upper())
    switch = "".join(
        "    case {}: return &{}_{}_nnconfig;\n".format(
            {8: "BLOCK_8X8", 16: "BLOCK_16X16", 32: "BLOCK_32X32",
             64: "BLOCK_64X64"}[dim], args.prefix, dim) for dim in dims)

    header = (
        LICENSE
        + "\n// Generated by src/scripts/partition_model/export_weights.py -- "
          "do not edit.\n"
          "// Distilled AV1 partition student ({}): per-size MLP, features "
          "from\n"
          "// src/scripts/partition_model/features.py (NUM_FEATURES={}).\n"
          "// Output softmax = [P(NONE), P(SPLIT), P(REST)].\n".format(
              args.prefix, nfeat)
        + "#ifndef {g}\n#define {g}\n\n".format(g=guard)
        + '#include "av1/encoder/ml.h"\n#include "av1/common/enums.h"\n\n'
        + "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
        + "#define {}_NUM_FEATURES {}\n\n".format(args.prefix.upper(), nfeat)
        + arrays_all
        + configs_all
        + "static inline const NN_CONFIG *{}_nnconfig(\n"
          "    BLOCK_SIZE bsize) {{\n"
          "  switch (bsize) {{\n".format(args.prefix)
        + switch
        + "    default: return NULL;\n  }\n}\n\n"
        + "#ifdef __cplusplus\n}  // extern \"C\"\n#endif\n\n"
        + "#endif  // {}\n".format(guard))
```

**Verificar (não quebrar o uso existente do H9a):** com `--prefix` omitido (default `av1_partition_student`), a saída deve ficar **idêntica** ao header atual (mesmos símbolos), então re-exportar o H9a existente é um no-op de conteúdo — confirmar no Step 2.

- [ ] **Step 2: Regressão — re-exportar o header H9a (deve ficar idêntico)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && cp src/aom/av1/encoder/partition_student_weights.h /tmp/before_h9a.h && \
  build/venv-ml/bin/python src/scripts/partition_model/export_weights.py \
  --students results/models/student_h9a/students.pt \
  --out src/aom/av1/encoder/partition_student_weights.h && \
  diff -q /tmp/before_h9a.h src/aom/av1/encoder/partition_student_weights.h && echo IDENTICAL'
```
Expected: `IDENTICAL` (a mudança do `--prefix` não altera o comportamento default).

- [ ] **Step 3: Exportar o header H9c**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/partition_model/export_weights.py \
  --students results/models/student_h9c/students.pt \
  --prefix av1_partition_student_h9c \
  --out src/aom/av1/encoder/partition_student_h9c_weights.h && \
  grep -m1 NUM_FEATURES src/aom/av1/encoder/partition_student_h9c_weights.h'
```
Expected: `#define AV1_PARTITION_STUDENT_H9C_NUM_FEATURES 39`.

- [ ] **Step 4: Commit**

```bash
git add src/scripts/partition_model/export_weights.py src/aom/av1/encoder/partition_student_weights.h src/aom/av1/encoder/partition_student_h9c_weights.h
git commit -F- <<'EOF'
H9c: parametrize export_weights.py symbol prefix; export H9c header

--prefix lets two student headers (H9a 36-feat, H9c 39-feat) coexist in the
same translation unit without symbol collisions. Default prefix reproduces
the existing H9a header byte-for-byte (verified). New
partition_student_h9c_weights.h carries the post-NONE student, not yet
included anywhere (Task 5 wires it in).
EOF
```

---

### Task 5: C — hook pós-NONE (`av1_prune_after_none`)

**Files:**
- Modify: `src/aom/av1/encoder/partition_strategy.c` (nova função, perto de `student_prune_partition`)
- Modify: `src/aom/av1/encoder/partition_strategy.h` (novo protótipo)
- Modify: `src/aom/av1/encoder/partition_search.c:5811-5833` (`av1_rd_pick_partition`, um novo call site)

**Interfaces:**
- Consumes: `student_node_features` (já existe, preenche `feats[0..35]`); `partition_student_h9c_weights.h` (Task 4, `av1_partition_student_h9c_nnconfig`); `part_search_state.this_rdc`/`.none_rd` (já preenchidos por `none_partition_search`, `encodeframe_utils.h:184,196`).
- Produces: `void av1_prune_after_none(const AV1_COMMON *cm, MACROBLOCK *x, PartitionSearchState *part_state);` — símbolo sempre presente (chamado sem `#if` em `partition_search.c`); corpo é no-op com a flag `PARTITION_ML_STUDENT` off ou com `AV1_STUDENT_H9C_ENABLE` não setado.

**Ação do modelo:** ao contrário do H9a (3 ações: NONE/SPLIT/REST, pré-busca), o H9c só tem uma ação sensata pós-NONE — **terminar a busca aqui** (committing a NONE), via `av1_disable_all_splits()` (já existe, `encodeframe_utils.h:261`; não mexe em `partition_none_allowed`, que já foi consumido). O modelo continua sendo o MLP de 3 classes (reuso de `student.py`/`av1_nn_predict` sem mudança de arquitetura); só `probs[0]` (P(NONE)) é usado como o sinal de "pare aqui".

- [ ] **Step 1: Incluir o header H9c e declarar as taus H9c em `partition_strategy.c`**

Logo após o `#include` do header H9a (linha ~1556, dentro do bloco `#if PARTITION_ML_STUDENT`), adicionar:
```c
#include "av1/encoder/partition_student_h9c_weights.h"
```

Logo após `student_get_taus` (que termina por volta da linha 1640), adicionar a variante H9c (um único tau — a ação é binária, "terminar" ou não):
```c
// H9c (post-NONE): single tau per level; AV1_STUDENT_H9C_ENABLE gates the
// whole feature (default off -> av1_prune_after_none is a no-op).
static int student_h9c_enabled(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *e = getenv("AV1_STUDENT_H9C_ENABLE");
    cached = (e && e[0] && e[0] != '0') ? 1 : 0;
  }
  return cached;
}

static float student_h9c_get_tau(int n) {
  static int inited = 0;
  static float tau[3];  // 0 -> 16px, 1 -> 32px, 2 -> 64px
  if (!inited) {
    static const char *suffix[3] = { "_16", "_32", "_64" };
    const float g = student_env_tau("AV1_STUDENT_H9C_TAU", 0.9f);
    for (int i = 0; i < 3; ++i) {
      char name[40];
      snprintf(name, sizeof(name), "AV1_STUDENT_H9C_TAU%s", suffix[i]);
      tau[i] = student_env_tau(name, g);
    }
    inited = 1;
  }
  return tau[n == 16 ? 0 : (n == 32 ? 1 : 2)];
}
```

- [ ] **Step 2: Implementar `av1_prune_after_none` (chamada sempre presente, corpo condicional)**

No final do bloco `#if PARTITION_ML_STUDENT ... #endif` (logo antes do `#endif  // PARTITION_ML_STUDENT` na linha ~2055), adicionar a função estática que faz o trabalho, e depois do `#endif`, a função exportada sempre presente que a chama:

```c
#if PARTITION_ML_STUDENT
// H9c: post-NONE decision. Called right after none_partition_search() has
// filled part_state->this_rdc (real PARTITION_NONE rate/dist/rdcost) and
// before split/rectangular/AB/4-way search runs. Mirrors the gate used by the
// pre-search student (intra-only, sb_size>=64, bsize in {16,32,64}, whole
// 64x64 unit in frame -- same parent-context requirement as block A).
static void student_h9c_decide(const AV1_COMMON *cm, MACROBLOCK *x,
                               PartitionSearchState *part_state) {
  const PartitionBlkParams *blk = &part_state->part_blk_params;
  if (part_state->none_rd <= 0 || part_state->this_rdc.rate == INT_MAX) return;
  const NN_CONFIG *nnconfig = av1_partition_student_h9c_nnconfig(blk->bsize);
  if (!nnconfig) return;
  float feats[AV1_PARTITION_STUDENT_H9C_NUM_FEATURES];
  student_node_features(cm, x, blk, feats);  // fills [0..35] (A+B+C)
  const RD_STATS *none_rdc = &part_state->this_rdc;
  feats[36] = (float)log1p((double)AOMMAX(none_rdc->rate, 0));
  feats[37] = (float)log1p((double)AOMMAX(none_rdc->dist, 0));
  feats[38] = (float)log1p((double)AOMMAX(none_rdc->rdcost, 0));
  float logits[3], probs[3];
  av1_nn_predict(feats, nnconfig, 1, logits);
  av1_nn_softmax(logits, probs, 3);
  const float tau = student_h9c_get_tau(block_size_wide[blk->bsize]);
  if (probs[0] > tau) av1_disable_all_splits(part_state);
}
#endif  // PARTITION_ML_STUDENT

void av1_prune_after_none(const AV1_COMMON *cm, MACROBLOCK *x,
                          PartitionSearchState *part_state) {
#if PARTITION_ML_STUDENT
  if (!student_h9c_enabled()) return;
  const PartitionBlkParams *blk = &part_state->part_blk_params;
  const CommonModeInfoParams *const mi_params = &cm->mi_params;
  const int try_h9c =
      frame_is_intra_only(cm) && cm->seq_params->sb_size >= BLOCK_64X64 &&
      blk->bsize <= BLOCK_64X64 && blk->bsize > BLOCK_8X8 &&
      (blk->mi_row & ~15) + 16 <= mi_params->mi_rows &&
      (blk->mi_col & ~15) + 16 <= mi_params->mi_cols;
  if (try_h9c) student_h9c_decide(cm, x, part_state);
#else
  (void)cm;
  (void)x;
  (void)part_state;
#endif  // PARTITION_ML_STUDENT
}
```

- [ ] **Step 3: Declarar o protótipo em `partition_strategy.h`**

Logo após a declaração de `av1_prune_partitions_before_search` (`partition_strategy.h:106`), adicionar:
```c
// H9c: post-NONE pruning decision (see partition_strategy.c). Always
// declared/linked; a no-op unless PARTITION_ML_STUDENT and
// AV1_STUDENT_H9C_ENABLE are both set, mirroring
// av1_prune_partitions_before_search's pattern.
void av1_prune_after_none(const struct AV1Common *cm, struct macroblock *x,
                          PartitionSearchState *part_state);
```
(conferir os nomes de tipo exatos usados nas declarações vizinhas no mesmo header — `partition_strategy.h` já inclui os forward-declares necessários para `av1_prune_partitions_before_search`; replicar o mesmo idioma de tipos ali.)

- [ ] **Step 4: Único novo call site em `partition_search.c`**

Em `av1_rd_pick_partition` (`partition_search.c`), logo após o bloco `#if LOG_PARTITION_DATA ... #endif` que captura `log_sample.none_rate/dist/rdcost` (termina por volta da linha 5825) e **antes** do comentário `// PARTITION_SPLIT search stage.` (linha ~5833), adicionar uma única linha:
```c
  av1_prune_after_none(cm, x, &part_search_state);
```

- [ ] **Step 5: Build (flag on) — compila limpo**

Run:
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_ml_check \
  --target aomenc -j"$(nproc)" 2>&1 | tail -15'
```
Expected: build conclui sem `error:`. Se `av1_prune_after_none` não linkar, revisar o Step 3 (tipos do protótipo devem casar exatamente com a definição).

- [ ] **Step 6: Smoke — o toggle desligado é inerte, ligado muda o bitstream**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
test -f /tmp/f4.yuv || dd if=/dev/zero of=/tmp/f4.yuv bs=1 count=$((448*256*3/2)) 2>/dev/null
build/libaom_ml_check/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 \
  --end-usage=q --cq-level=32 -w 448 -h 256 --limit=1 -o /tmp/h9c_off.ivf /tmp/f4.yuv 2>/dev/null
AV1_STUDENT_H9C_ENABLE=1 AV1_STUDENT_H9C_TAU=0.5 \
  build/libaom_ml_check/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 \
  --end-usage=q --cq-level=32 -w 448 -h 256 --limit=1 -o /tmp/h9c_on.ivf /tmp/f4.yuv 2>/dev/null
md5sum /tmp/h9c_off.ivf /tmp/h9c_on.ivf'
```
Expected: md5s **diferentes** (tau agressivo de 0.5 deve prunar algo nesse frame sintético; se forem iguais, o hook não está disparando — revisar o gate `try_h9c`/Step 4).

- [ ] **Step 7: Commit**

```bash
git add src/aom/av1/encoder/partition_strategy.c src/aom/av1/encoder/partition_strategy.h src/aom/av1/encoder/partition_search.c
git commit -F- <<'EOF'
H9c: post-NONE pruning hook in av1_rd_pick_partition

New av1_prune_after_none() runs right after none_partition_search() has
filled part_state->this_rdc/none_rd, before split/rectangular/AB/4-way
search -- the real PARTITION_NONE rate/dist/rdcost feed a 39-feature MLP
(A+B+C+E) that decides whether to terminate the search here via the existing
av1_disable_all_splits(). Gated by AV1_STUDENT_H9C_ENABLE (default off,
no-op); the single call site in partition_search.c is unconditional (the
function itself carries the #if PARTITION_ML_STUDENT no-op body), matching
the existing av1_prune_partitions_before_search pattern.
EOF
```

---

### Task 6: C↔Python — paridade do vetor de 39 features

**Files:**
- Modify: `src/aom/av1/encoder/partition_strategy.c` (dump específico do H9c, reaproveitando `student_dump_features` como referência)
- Modify: `src/scripts/partition_model/check_feature_parity.py` (ou script irmão dedicado, se ficar mais claro isolado)

**Interfaces:**
- Consumes: `student_h9c_decide` (Task 5); `features.node_features_h9c` (Task 1).
- Produces: `PARITY OK` nas 39 features + probs, contra `libaom_ml_check` com `AV1_STUDENT_H9C_ENABLE=1`.

- [ ] **Step 1: Dump dedicado dentro de `student_h9c_decide`**

Reaproveitando o padrão de `student_dump_features` (record `head[10]` já usado no H9a, mais os 3 campos de `none_rdc` que já estão disponíveis aqui), adicionar ao final de `student_h9c_decide` (Task 5, Step 2), antes do `if (probs[0] > tau) ...`:
```c
  {
    static FILE *fp = NULL;
    static int checked = 0;
    if (!checked) {
      const char *path = getenv("AV1_STUDENT_H9C_DUMP");
      if (path) fp = fopen(path, "ab");
      checked = 1;
    }
    if (fp) {
      const MACROBLOCKD *const xd = &x->e_mbd;
      const int has_above = !!xd->above_mbmi;
      const int has_left = !!xd->left_mbmi;
      const int neigh_avail = has_above | (has_left << 1);
      const int above_bsize = has_above ? xd->above_mbmi->bsize : blk->bsize;
      const int left_bsize = has_left ? xd->left_mbmi->bsize : blk->bsize;
      const int dc_q = av1_dc_quant_QTX(x->qindex, 0, xd->bd) >> (xd->bd - 8);
      const int32_t head[10] = { block_size_wide[blk->bsize], blk->mi_row,
                                 blk->mi_col, x->qindex, neigh_avail,
                                 above_bsize, left_bsize, dc_q, cm->width,
                                 cm->height };
      fwrite(head, sizeof(head), 1, fp);
      fwrite(feats, sizeof(float), AV1_PARTITION_STUDENT_H9C_NUM_FEATURES, fp);
      fwrite(probs, sizeof(float), 3, fp);
    }
  }
```
Record: `head[10]` (40B) + `feats[39]` (156B) + `probs[3]` (12B) = **208 bytes**.

- [ ] **Step 2: Harness Python — copiar `check_feature_parity.py` para uma variante H9c**

Run:
```bash
cp src/scripts/partition_model/check_feature_parity.py src/scripts/partition_model/check_feature_parity_h9c.py
```
Nessa cópia, seguindo exatamente o mesmo padrão da Task 2 do plano `2026-07-11-fase4-integracao-c-h9a.md` (struct `REC`, `--students` default, reconstrução do `ctx`, chamada de `node_features_h9c` em vez de `node_features_h9a`), trocar:
- `REC = struct.Struct("<10i{}f".format(39 + 3))`
- `--students` default: `results/models/student_h9c/students.pt`
- `ctx` ganha `"none_rate"`, `"none_dist"`, `"none_rdcost"` — **não vêm do dump** (o dump de 39 features já inclui esses valores dentro do próprio vetor `feats[36..38]`, não como campos separados do `head`). Portanto a comparação aqui é mais simples que a do H9a: **não precisa reconstruir** as 3 últimas features via `ctx` — basta comparar `cf[36:39]` (do dump C) contra o que o Python computaria SE tivesse o mesmo `none_rate/dist/rdcost`. Como esses valores não estão no `head`, a paridade das features 36-38 não pode ser verificada por reconstrução independente neste harness — **ela é verificada por construção** (o C aplica `log1p` aos mesmos três `int64` que já vieram do `RD_STATS` real; não há como o Python recalcular esse valor sem lê-lo do C). Reportar apenas a paridade de **0..35** (idêntica ao H9a) como `PARITY OK (0..35)`, e documentar essa limitação explicitamente no cabeçalho do script.
- Usar `AV1_STUDENT_H9C_DUMP` como env var de entrada, `AV1_STUDENT_H9C_ENABLE=1` no encode de smoke.

- [ ] **Step 3: Rodar o gate de paridade (0..35 green)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
AV1_STUDENT_H9C_ENABLE=1 AV1_STUDENT_H9C_DUMP=/tmp/h9c_parity.bin \
  build/libaom_ml_check/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 \
  --end-usage=q --cq-level=32 -w 448 -h 256 --limit=1 -o /tmp/h9c_parity.ivf /tmp/f4.yuv 2>/dev/null
build/venv-ml/bin/python src/scripts/partition_model/check_feature_parity_h9c.py \
  --dump /tmp/h9c_parity.bin --students results/models/student_h9c/students.pt 2>&1 | tail -45'
```
Expected: features 0–35 sem `MISMATCH` (mesma tolerância do H9a, ~1e-6); `PARITY OK (0..35)`.

- [ ] **Step 4: Commit**

```bash
git add src/aom/av1/encoder/partition_strategy.c src/scripts/partition_model/check_feature_parity_h9c.py
git commit -F- <<'EOF'
H9c: parity dump (39-feat record, 208B) + dedicated parity harness

Verifies features 0-35 bit-for-bit against node_features_h9c (same coverage
as the H9a harness); features 36-38 (block E) are verified by construction
(the C side applies log1p to the same RD_STATS the live encoder just
computed, with no independent Python recomputation path -- documented as a
known harness limitation, not a gap in correctness).
EOF
```

---

### Task 7: C — no-op byte-idêntico + testes (Gate de segurança)

**Files:** nenhum (verificação). Reaproveita `libaom_noop` (já existe da Fase 4).

**Interfaces:** Consumes Tasks 5-6. Produces evidência do Gate.

- [ ] **Step 1: No-op byte-idêntico vs `aom_baseline` (`AV1_STUDENT_H9C_ENABLE` unset)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace &&
for b in libaom_noop libaom_perf_anchor; do
  build/$b/aomenc --usage=2 --passes=1 --threads=1 --cpu-used=0 --end-usage=q \
    --cq-level=32 -w 448 -h 256 --limit=1 -o /tmp/h9c_noop_$b.ivf /tmp/f4.yuv 2>/dev/null
done
md5sum /tmp/h9c_noop_libaom_noop.ivf /tmp/h9c_noop_libaom_perf_anchor.ivf'
```
Expected: md5s **idênticos** (o `av1_prune_after_none` novo em `partition_search.c` é uma chamada sempre presente mas cujo corpo é no-op sem a flag — `libaom_noop` é compilado de `src/aom` com `PARTITION_ML_STUDENT` OFF, então o corpo cai no ramo `#else` que só faz `(void)` nos parâmetros).

**Nota:** este teste cobre a flag `PARTITION_ML_STUDENT` off. Também confirmar que com a flag ON mas `AV1_STUDENT_H9C_ENABLE` **unset** (o estado do binário `libaom_perf` usado hoje pelo H9a em produção/Fase 6), o bitstream continua **idêntico ao H9a sem a mudança** — reaproveitar o par de comandos do Step 6 da Task 5 (`h9c_off.ivf` já gerado) comparando seu md5 contra um encode feito com o `libaom_perf` **anterior** a esta feature (se disponível) ou, na ausência dele, contra `libaom_perf_anchor` com os taus do H9a apenas (confirma que a ausência do H9c não interfere no fluxo do H9a).

- [ ] **Step 2: Sanidade de testes**

Run:
```bash
docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_dev_generic \
  --target test_libaom -j"$(nproc)" >/dev/null 2>&1 &&
/workspace/build/libaom_dev_generic/test_libaom \
  --gtest_filter="EncodeAPI.AllIntra*:KeyValAPI.*partition*:KeyValAPI.*intra*" \
  2>&1 | tail -6'
```
Expected: `[  PASSED  ] N tests.`, zero falhas.

- [ ] **Step 3: Registrar evidência e commit**

```bash
docker exec av1_bench bash -lc 'cd /workspace && {
  echo "H9c Gate evidence ($(date -u +%FT%TZ)):"
  echo "no-op md5:"; md5sum /tmp/h9c_noop_*.ivf
} > results/models/student_h9c/gate_c_evidence.txt && cat results/models/student_h9c/gate_c_evidence.txt'
git add -f results/models/student_h9c/gate_c_evidence.txt
git commit -F- <<'EOF'
H9c: Gate evidence (no-op byte-identical, tests pass)

Flag-off is byte-identical to aom_baseline; H9c-disabled (flag on, env
unset) leaves the H9a-only path untouched. Scoped unit tests pass. H9c is
now wired but dormant by default -- only active with
AV1_STUDENT_H9C_ENABLE=1.
EOF
```

---

### Task 8: Piloto de tempo real (decisão antes da CTC)

**Files:**
- Modify: `src/scripts/benchmark/h7h8_bench.py` (novo preset `h9c`)

**Interfaces:**
- Consumes: build `libaom_perf` (rebuilt com Tasks 5-7); binário `libaom_perf_anchor` (âncora, já existe).
- Produces: `results/benchmark/h9c_pilot/{Jockey,RaceNight,RiverBank ou só 1 seq}/summary.csv` — BD-rate/TS%/speedup do H9c sozinho, comparável linha a linha com o H9a já medido (`results/benchmark/h9_test/*/curve_safe/summary.csv`).

- [ ] **Step 1: Adicionar o preset `h9c` aos `points` do driver**

Em `h7h8_bench.py`, junto aos blocos existentes de `points = [...]` (linhas ~90 e ~107, selecionados por `--preset`), adicionar um terceiro branch:
```python
    elif args.preset == "h9c":
        points = [
            ("anchor", args.anchor_enc, {}, None),
            ("H9c_tau90", args.test_enc,
             {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.90"},
             None),
            ("H9c_tau95", args.test_enc,
             {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.95"},
             None),
        ]
```
E adicionar `"h9c"` às opções válidas de `--preset` (`choices=["safe", "aggressive", "h9c"]`).

- [ ] **Step 2: Rodar o piloto em 1 sequência held-out (Jockey), poucos quadros**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/benchmark/h7h8_bench.py \
  --test-enc build/libaom_perf/aomenc --anchor-enc build/libaom_perf_anchor/aomenc \
  --decoder build/libaom_ml_check/aomdec \
  --seq src/samples/Jockey_3840x2160_120fps_...yuv \
  --preset h9c --frames 2 --cqs 20 32 43 55 \
  --out-dir results/benchmark/h9c_pilot 2>&1 | tail -40'
```
(ajustar o caminho exato do `.yuv` da Jockey conforme já usado nas execuções de Fase 5/6 — conferir em `results/benchmark/h9_test/Jockey/curve_safe/run.log` ou equivalente para o path correto.)
Expected: tabela final com BD-rate/TS%/speedup para `H9c_tau90` e `H9c_tau95` vs `anchor`.

- [ ] **Step 3: Decisão — comparar contra o H9a já medido no mesmo ponto**

Sem script novo — olhar `results/benchmark/h9_test/Jockey/curve_safe/summary.csv` (H9a, ponto "safe") e comparar BD-rate/TS% do H9c no mesmo regime de TS aproximado. **Critério:**
- Se H9c entregar **BD-rate menor a TS comparável** (ou TS maior a BD-rate comparável) que o H9a → sinal real, prosseguir para a Task 9 (CTC).
- Se H9c **não bater o H9a** em tempo real (mesmo tendo vencido no Gate 3 oráculo) → **parar aqui**. Registrar em `docs/ANDAMENTO_tese.md`: "o teto de contexto RD (H9c) não sobrevive à medição de tempo real, apesar de superar H9a no oráculo — mesma superestimativa histórica do oráculo (~5x)". Não gastar a CTC.

- [ ] **Step 4: Commit**

```bash
git add src/scripts/benchmark/h7h8_bench.py results/benchmark/h9c_pilot
git commit -F- <<'EOF'
H9c: pilot wall-clock benchmark (Jockey, 2 frames x 4 cq)

New h7h8_bench.py preset "h9c" (anchor + 2 tau points). Cheap sanity check
before spending the CTC benchmark: does the post-NONE learned pruner beat
H9a in real wall-clock terms, not just in the Gate 3 oracle simulation?
EOF
```

---

### Task 9 (condicional — só se a Task 8 passou): Benchmark pragmático CTC

**Files:**
- Create: `src/scripts/fase6/encode_h9c.py` (fork de `encode_swap.py`)
- Modify: `docs/RESULTADOS_fase6.md` (nova seção §5)

**Interfaces:**
- Consumes: âncora/H9a/nativo cpu1-3 **já medidos** (`results/benchmark/fase6/raw_results.csv`, `results/benchmark/fase6_swap/raw_results.csv` — reaproveitados, **não re-executados**, mesmo padrão da Task de swap).
- Produces: `results/benchmark/fase6_h9c/raw_results.csv` — só os encodes novos do H9c (8 seqs × 4 cq × 1 τ escolhido = 32 encodes, não 192).

- [ ] **Step 1: Fork do driver de encode, trocando só a config**

Run:
```bash
cp src/scripts/fase6/encode_swap.py src/scripts/fase6/encode_h9c.py
```
Editar `encode_h9c.py`: remover as configs `h9a_bal_cpuN`/`h9a_aggr_cpuN` × `cpu∈{1,2,3}`, deixar só:
```python
CONFIGS = [
    ("H9c", "build/libaom_perf/aomenc", 0,
     {"AV1_STUDENT_H9C_ENABLE": "1",
      "AV1_STUDENT_H9C_TAU": str(CHOSEN_TAU)}),  # do Step 3 da Task 8
]
```
(`CHOSEN_TAU` = o valor de `AV1_STUDENT_H9C_TAU` que venceu o piloto na Task 8 Step 3; `cpu-used=0`, mesmo regime do H9a na Fase 6 original — a comparação de categoria aqui é âncora/H9a/H9c todos em cpu0, mais os presets nativos cpu1-3 como referência de mercado, igual à tabela §3 original.)

- [ ] **Step 2: Rodar (32 encodes, 8 seqs A1 × 4 cq)**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace && build/venv-ml/bin/python \
  src/scripts/fase6/encode_h9c.py \
  --out-dir results/benchmark/fase6_h9c 2>&1 | tail -60'
```
Expected: 32 encodes completam; `results/benchmark/fase6_h9c/raw_results.csv` criado.

- [ ] **Step 3: Montar a tabela final (ancora / H9a / H9c / nativo cpu1-3)**

Sem script novo — juntar manualmente (ou um `report_h9c.py` curto, fork de `report_ctc.py`, se o merge dos 3 CSVs for tedioso) os `bd_rate`/`ts_pct`/`speedup` médios de:
`results/benchmark/fase6/bdrate_average.csv` (âncora, H9a balanced/aggressive, nativo cpu1/2/3) + `results/benchmark/fase6_h9c/raw_results.csv` (H9c) numa única tabela, no mesmo formato da tabela em `RESULTADOS_fase6.md` §3.

- [ ] **Step 4: Escrever `RESULTADOS_fase6.md` §5 e commitar**

Adicionar ao final do documento uma seção `## 5. H9c — teto de contexto RD pós-NONE na CTC` com a tabela do Step 3 e uma leitura honesta (mesmo padrão de §3/§4.3): se H9c superar H9a e/ou os presets nativos, é o resultado headline da tese; se não, documenta que nem o teto de contexto RD sobrevive à comparação prática, fechando a investigação com uma caracterização completa (pixels saturam → contexto RD pré-busca não bate SOTA → nem o teto pós-NONE bate).

```bash
git add src/scripts/fase6/encode_h9c.py docs/RESULTADOS_fase6.md results/benchmark/fase6_h9c
git commit -F- <<'EOF'
H9c: CTC pragmatic benchmark (ancora vs H9a vs H9c vs nativo cpu1-3)

Reuses the anchor/H9a/native CTC data already measured in Fase 6 (fase6,
fase6_swap); only the 32 new H9c encodes (8 A1 seqs x 4 cq, single tau
chosen from the Task 8 pilot) were run. Table + honest reading recorded in
RESULTADOS_fase6.md S5.
EOF
```

---

## Notas de execução

- **A Task 3 (Gate 3) e a Task 8 (piloto) são os dois portões de parada.** Não seguir para C (Task 5+) sem o Gate 3 passar; não seguir para a CTC (Task 9) sem o piloto mostrar ganho real sobre H9a. Isso é o que torna o plano "inatacável" (mesma disciplina de H9a/Fase 2).
- **`av1_disable_all_splits` pós-NONE é seguro:** não mexe em `partition_none_allowed` (NONE já foi avaliado e seu RD já está em `part_search_state.this_rdc`); só impede SPLIT/retangular/AB/4-way de rodar depois. Confirmado lendo `encodeframe_utils.h:246-268`.
- **Linha exatas em `partition_search.c`/`partition_strategy.c` podem ter deslocado** desde a exploração que embasou este plano (ex.: se algum commit recente tocou essas áreas) — usar os comentários/nomes de função como âncora de busca (`none_partition_search`, `LOG_PARTITION_DATA`, `student_get_taus`) em vez de confiar cegamente no número da linha; os builds e gates pegam qualquer desalinhamento.
- **Diferença de escopo vs. H9a:** o H9a decide **antes** de qualquer busca (pode economizar o custo do NONE também); o H9c só evita SPLIT/retangular/AB/4-way **depois** de já ter pago o NONE — o teto de ganho é estruturalmente menor. Isso é esperado e já está registrado como a limitação honesta do H9c desde o plano original (`PLANO_H9_contribuicao_tese.md`).
