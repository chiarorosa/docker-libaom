# Solução 4 — NN regressora de *regret* RD para poda intra — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar uma NN que *regride* o custo RD de podar cada nó (*regret*) e usá-la como *pruner* pré-busca do particionamento intra do AV1, decidindo por custo predito (`predicted_regret < τ_regret`) em vez de confiança de um classificador.

**Architecture:** Reusa toda a infra da Solução 2 (H9a): as 36 *features* A+B+C (`features.node_features_h9a`, fonte única, já com paridade C↔Python) e o MLP por tamanho de bloco. Troca-se a cabeça *softmax* 3-vias por uma **saída de regressão única**. O alvo `regret` é reconstruído **offline** da árvore comprometida de cada superbloco (`data.iter_superblock_members` → `none_rdcost` por nó), sem re-extração. A cadeia de gates da tese (0/2/3/4/5/6) é mantida.

**Tech Stack:** Python 3 (numpy, torch/GPU), pytest; libaom v3.10.0 (C, build em Docker); scripts em `src/scripts/partition_model/` e `src/scripts/fase6/`.

## Global Constraints

- **Execução sempre em Docker**, no container detached `av1_bench` (`docker exec av1_bench …`). Windows é só edição. (memória `execution-always-in-docker`)
- **Interpretador Python** = o mesmo que já roda o pipeline (numpy+torch+GPU, o que gerou `results/models/student_h9a/`). Referido abaixo como `PY` (definido na Task 0).
- **Sem re-extração:** usar exclusivamente `results/dataset_h9/*.pkl` (already on disk).
- **Partição de sequências CONGELADA** (`docs/PROTOCOLO_avaliacao.md`): treino = `["Beauty","Bosphorus","CityAlley","FlowerFocus","FlowerKids","ReadySetGo","ShakeNDry","SunBath","Twilight","YachtRide"]`; validação = `["HoneyBee","FlowerPan","Lips"]`; **teste reservado = `["Jockey","RaceNight","RiverBank"]`** (nunca em treino/val). Copiar de `train_student_h9.TRAIN_SEQS`/`VAL_SEQS`.
- **Fonte única das *features*** = `features.node_features_h9a` (36). Nunca reimplementar.
- **Índices de partição** (`partition_defs.PARTITION_NAMES`): `NONE=0`, `SPLIT=3`; retangulares = {1,2,4,5,6,7,8,9}. Níveis de decisão = `MODEL_LEVELS = [(64,1),(32,2),(16,4)]` (8px é sempre NONE, fora do modelo).
- **Todo código C novo** sob `#if PARTITION_ML_STUDENT`; **no-op byte-idêntico** à âncora quando as envs não estão setadas (md5 verificado). Reusa `av1_nn_predict` (nenhuma inferência nova em C).
- **Commits:** sem qualquer menção a Claude/AI/Co-Authored-By (memória `no-ai-attribution-in-commits`). **Ao fim de cada tarefa: `git commit` + `git push`** na branch `ml-partition-dev` (memória `commit-push-on-task-completion`).
- **Idioma:** docs/comentários em PT-BR; jargão técnico em inglês quando consagrado.

---

## File Structure

**Criar:**
- `src/scripts/partition_model/regret.py` — núcleo puro: reconstrução da árvore comprometida + cálculo de `regret_rel` por nó (sem I/O, testável isolado).
- `src/scripts/partition_model/build_regret_targets.py` — varre os pkls, emite arrays por tamanho `{feat(36), regret, exact}` e cacheia `.npz`.
- `src/scripts/partition_model/gate0_regret.py` — Gate 0 (viabilidade do sinal sobre os pkls de treino).
- `src/scripts/partition_model/train_regret.py` — treino da NN regressora (cabeça única) → bundle.
- `src/scripts/partition_model/tests/test_regret.py` — testes do núcleo (`regret.py`) e do builder.
- `docs/RESULTADOS_solucao4.md` — resultados (preenchido nos gates 5/6).

**Modificar:**
- `src/scripts/partition_model/simulate_pruning.py` — modo *regret-score* (Gate 2/3).
- `src/scripts/partition_model/export_weights.py` — exportar cabeça de regressão → header C.
- `src/scripts/partition_model/ablation_attrib.py` (em `src/scripts/fase6/` se for lá que vive; senão o do partition_model) — fonte de escore `regret` (Gate 5).
- `src/aom/av1/encoder/partition_strategy.c` — `student_regret_decide` + envs `AV1_STUDENT_REGRET_ENABLE`/`AV1_STUDENT_TAU_REGRET`.
- `docs/SINTESE_resultados_metodologia.md`, `docs/ANDAMENTO_tese.md` — nova seção "Solução 4".

---

## Task 0: Setup do ambiente e do diretório de testes

**Files:**
- Create: `src/scripts/partition_model/tests/__init__.py` (vazio)
- Create: `src/scripts/partition_model/tests/conftest.py`

**Interfaces:**
- Produces: variável `PY` (comando do interpretador no container) usada por todas as tarefas; `pytest` disponível nesse interpretador.

- [ ] **Step 1: Confirmar o interpretador do pipeline (numpy+torch)**

Run:
```bash
docker exec av1_bench bash -lc 'python3 -c "import numpy,torch;print(numpy.__version__,torch.__version__,torch.cuda.is_available())"'
```
Expected: imprime versões de numpy/torch e `True`/`False` de CUDA sem erro. **Se falhar**, localizar o interpretador usado em `results/models/student_h9a_train.log` e usá-lo como `PY` no lugar de `python3`. Definir `PY="docker exec av1_bench python3"` para o restante do plano.

- [ ] **Step 2: Instalar pytest nesse interpretador**

Run:
```bash
docker exec av1_bench bash -lc 'python3 -m pip install --quiet pytest || python3 -m pip install --quiet --break-system-packages pytest'
docker exec av1_bench bash -lc 'python3 -m pytest --version'
```
Expected: `pytest 7.x`/`8.x`.

- [ ] **Step 3: Criar o pacote de testes**

`src/scripts/partition_model/tests/__init__.py`: arquivo vazio.

`src/scripts/partition_model/tests/conftest.py`:
```python
import os
import sys

# Torna os módulos do pipeline (regret, data, features, partition_defs)
# importáveis sem instalar o pacote.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
```

- [ ] **Step 4: Commit + push**

```bash
git add src/scripts/partition_model/tests/__init__.py src/scripts/partition_model/tests/conftest.py
git commit -m "solucao4: scaffolding de testes (pytest) do pipeline de partition_model"
git push
```

---

## Task 1: Núcleo `regret.py` — reconstrução da árvore + `regret_rel` por nó

**Files:**
- Create: `src/scripts/partition_model/regret.py`
- Test: `src/scripts/partition_model/tests/test_regret.py`

**Interfaces:**
- Consumes: `members` e `ctx` no formato de `data.iter_superblock_members` — `members[k] = (dim, r, c, block_luma, label)`, `ctx[k] = {"none_rdcost": int, ...}`.
- Produces:
  - `node_regrets(members, ctx) -> list[dict]`, cada dict = `{"dim":int, "r":int, "c":int, "k":int, "regret_rel":float, "exact":bool}`. `k` é o índice em `members` (para casar a feature). Só níveis `64/32/16`; nós sem `none_rdcost` válido (≤0) são omitidos.

- [ ] **Step 1: Escrever o teste que falha**

`src/scripts/partition_model/tests/test_regret.py`:
```python
import numpy as np
import regret


def _mk(dim, r, c, label, rd):
    """member tuple + ctx dict com o none_rdcost dado."""
    luma = np.zeros((dim, dim), dtype=np.uint8)
    return (dim, r, c, luma, label), {"none_rdcost": rd}


def _split_sb(root_rd, child_rds):
    """SB 64 que faz SPLIT na raiz; 4 filhos 32 todos NONE com os rd dados."""
    members, ctx = [], []
    m, x = _mk(64, 0, 0, 3, root_rd)   # 3 == SPLIT
    members.append(m); ctx.append(x)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (r, c), rd in zip(cells, child_rds):
        m, x = _mk(32, r, c, 0, rd)     # 0 == NONE
        members.append(m); ctx.append(x)
    return members, ctx


def test_root_none_has_zero_regret():
    # Raiz NONE: podar não custa nada.
    members, ctx = [ _mk(64, 0, 0, 0, 1000)[0] ], [ _mk(64, 0, 0, 0, 1000)[1] ]
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert out[(64, 0, 0)]["regret_rel"] == 0.0
    assert out[(64, 0, 0)]["exact"] is True


def test_split_regret_matches_hand_calc():
    # Raiz SPLIT, custo NONE(raiz)=1000; subárvore = soma dos filhos = 4*200=800.
    # regret_rel = (1000-800)/800 = 0.25 ; exato (nenhuma folha retangular).
    members, ctx = _split_sb(1000, [200, 200, 200, 200])
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert abs(out[(64, 0, 0)]["regret_rel"] - 0.25) < 1e-9
    assert out[(64, 0, 0)]["exact"] is True
    # Cada filho NONE tem regret 0.
    assert out[(32, 0, 0)]["regret_rel"] == 0.0


def test_rectangular_leaf_is_censored():
    # Raiz SPLIT; um filho é retangular (label HORZ=1) -> subárvore censurada.
    members, ctx = _split_sb(1000, [200, 200, 200, 200])
    members[1] = (32, 0, 0, np.zeros((32, 32), np.uint8), 1)  # filho vira HORZ
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert out[(64, 0, 0)]["exact"] is False   # herdou censura do filho retangular


def test_missing_none_rdcost_node_is_dropped():
    members, ctx = _split_sb(1000, [200, 200, 200, 200])
    ctx[0] = {"none_rdcost": 0}   # raiz sem NONE avaliado (sentinela)
    keys = {(n["dim"], n["r"], n["c"]) for n in regret.node_regrets(members, ctx)}
    assert (64, 0, 0) not in keys
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py -q'`
Expected: FAIL (`ModuleNotFoundError: No module named 'regret'`).

- [ ] **Step 3: Implementar `regret.py`**

`src/scripts/partition_model/regret.py`:
```python
#!/usr/bin/env python3
"""Núcleo puro da Solução 4: reconstrói a árvore comprometida de um superbloco
e calcula, por nó de decisão (64/32/16), o *regret* relativo de podar (comprometer
PARTITION_NONE) em vez de seguir a decisão RD-ótima.

regret_rel(n) = ( none_rdcost(n) - RD_subtree(n) ) / RD_subtree(n)

RD_subtree segue os rótulos: NONE -> folha (rd = none_rdcost, EXATO);
SPLIT -> soma dos 4 filhos; folha retangular -> rd = none_rdcost (LIMITE SUPERIOR,
marca a subárvore como censurada -> exact=False). Ver
docs/superpowers/specs/2026-07-17-solucao4-regret-regression-design.md §3.
"""

NONE = 0
SPLIT = 3
DECISION_DIMS = (64, 32, 16)


def _children(dim, r, c):
    cd = dim // 2
    return [(cd, 2 * r, 2 * c), (cd, 2 * r, 2 * c + 1),
            (cd, 2 * r + 1, 2 * c), (cd, 2 * r + 1, 2 * c + 1)]


def node_regrets(members, ctx):
    """Ver docstring do módulo. Retorna lista de dicts (um por nó de decisão com
    none_rdcost válido)."""
    node = {}
    for k, (dim, r, c, _luma, label) in enumerate(members):
        node[(dim, r, c)] = {"label": int(label),
                             "rd": int(ctx[k].get("none_rdcost", 0)),
                             "k": k, "dim": int(dim), "r": int(r), "c": int(c)}

    def subtree_rd(key):
        """(rd_da_subárvore_comprometida, censurada?) ou (None, True) se ausente."""
        nd = node.get(key)
        if nd is None or nd["rd"] <= 0:
            return None, True
        if nd["label"] == SPLIT:
            total, cens = 0, False
            for ch in _children(*key):
                crd, ccen = subtree_rd(ch)
                if crd is None:            # filho ausente -> cai no NONE do pai (upper bound)
                    return nd["rd"], True
                total += crd
                cens = cens or ccen
            return total, cens
        # NONE ou retangular: folha; rd = none_rdcost (exato só se NONE).
        return nd["rd"], (nd["label"] != NONE)

    out = []
    for key, nd in node.items():
        if nd["dim"] not in DECISION_DIMS or nd["rd"] <= 0:
            continue
        srd, cens = subtree_rd(key)
        if srd is None or srd <= 0:
            continue
        regret_rel = (nd["rd"] - srd) / srd
        out.append({"dim": nd["dim"], "r": nd["r"], "c": nd["c"], "k": nd["k"],
                    "regret_rel": max(float(regret_rel), 0.0),
                    "exact": (not cens)})
    return out
```

- [ ] **Step 4: Rodar e ver passar**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py -q'`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/partition_model/regret.py src/scripts/partition_model/tests/test_regret.py
git commit -m "solucao4: núcleo regret.py (árvore comprometida + regret_rel por nó) + testes"
git push
```

---

## Task 2: `build_regret_targets.py` — dataset de treino (feat36 + regret)

**Files:**
- Create: `src/scripts/partition_model/build_regret_targets.py`
- Test: adicionar `test_build_regret_targets_smoke` em `tests/test_regret.py`

**Interfaces:**
- Consumes: `data.iter_superblock_members`, `features.node_features_h9a`, `regret.node_regrets`.
- Produces: `collect_regret_by_dim(entries, per_pkl=None, limit=None, exact_only=True) -> dict{dim: {"feat":(N,36)f32, "regret":(N,)f32, "exact":(N,)bool}}`. Cabeçalho CLI grava `.npz` por split.

- [ ] **Step 1: Escrever o teste (usa 1 pkl real, cap pequeno)**

Adicionar em `tests/test_regret.py`:
```python
def test_build_regret_targets_smoke():
    import os
    import build_regret_targets as brt
    import data as datamod
    ds = "/workspace/results/dataset_h9"
    entries = datamod.discover_pkls(ds)
    assert entries, "dataset_h9 não encontrado"
    one = [entries[0]]
    out = brt.collect_regret_by_dim(one, per_pkl=20, exact_only=True)
    # Há nós de decisão nos três tamanhos, features com largura 36, regret >= 0.
    for dim in (64, 32, 16):
        assert dim in out
    feats = out[32]["feat"]
    assert feats.shape[1] == 36
    assert (out[32]["regret"] >= 0).all()
    assert out[32]["exact"].all()   # exact_only=True filtra censurados
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py::test_build_regret_targets_smoke -q'`
Expected: FAIL (`No module named 'build_regret_targets'`).

- [ ] **Step 3: Implementar `build_regret_targets.py`**

`src/scripts/partition_model/build_regret_targets.py`:
```python
#!/usr/bin/env python3
"""Monta o dataset de treino da Solução 4: para cada nó de decisão dos superblocos
do dataset_h9, empilha a feature H9a (36) e o alvo regret_rel (regret.node_regrets).
Espelha train_student_h9.collect_by_dim, trocando o rótulo pelo regret contínuo."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import regret as regretmod  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def collect_regret_by_dim(entries, per_pkl=None, limit=None, exact_only=True):
    acc = {dim: {"feat": [], "regret": [], "exact": []} for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD (dataset_h9 requerido)."
                                 .format(e["path"]))
            for nr in regretmod.node_regrets(sb["members"], sb["ctx"]):
                dim = nr["dim"]
                if dim not in acc:
                    continue
                if exact_only and not nr["exact"]:
                    continue
                k = nr["k"]
                _dim, r, c, _luma, _label = sb["members"][k]
                f = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                              sb["qindex"], sb["ctx"][k])
                acc[dim]["feat"].append(f)
                acc[dim]["regret"].append(nr["regret_rel"])
                acc[dim]["exact"].append(nr["exact"])
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and n_sb >= limit:
                return _finalize(acc)
    return _finalize(acc)


def _finalize(acc):
    return {dim: {"feat": np.asarray(v["feat"], dtype=np.float32),
                  "regret": np.asarray(v["regret"], dtype=np.float32),
                  "exact": np.asarray(v["exact"], dtype=bool)}
            for dim, v in acc.items()}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--per-pkl", type=int, default=3000)
    p.add_argument("--exact-only", action="store_true", default=True)
    p.add_argument("--include-censored", dest="exact_only", action="store_false")
    p.add_argument("--out", default="/workspace/results/models/regret/targets_train.npz")
    args = p.parse_args(argv)

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs, args.train_seqs)
    ents = train_e if args.split == "train" else val_e
    datamod.assert_real_luma(ents)
    per_pkl = args.per_pkl or None
    out = collect_regret_by_dim(ents, per_pkl=per_pkl, exact_only=args.exact_only)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    flat = {}
    for dim, v in out.items():
        flat["feat_{}".format(dim)] = v["feat"]
        flat["regret_{}".format(dim)] = v["regret"]
        print("[dim {:>2}] n={} regret[min/med/max]={:.4f}/{:.4f}/{:.4f}".format(
            dim, len(v["regret"]),
            float(v["regret"].min()) if len(v["regret"]) else 0.0,
            float(np.median(v["regret"])) if len(v["regret"]) else 0.0,
            float(v["regret"].max()) if len(v["regret"]) else 0.0), flush=True)
    np.savez_compressed(args.out, **flat)
    print("Saved ->", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Rodar e ver passar**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py::test_build_regret_targets_smoke -q'`
Expected: PASS.

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/partition_model/build_regret_targets.py src/scripts/partition_model/tests/test_regret.py
git commit -m "solucao4: build_regret_targets.py (feat36 + regret por tamanho) + smoke"
git push
```

---

## Task 3: Gate 0 — viabilidade do sinal (milestone)

**Files:**
- Create: `src/scripts/partition_model/gate0_regret.py`

**Interfaces:**
- Consumes: `build_regret_targets.collect_regret_by_dim` com `exact_only=False` (para medir a censura).
- Produces: relatório impresso + `results/models/regret/gate0.csv`; código de saída 0 se PASSA.

- [ ] **Step 1: Implementar `gate0_regret.py`**

`src/scripts/partition_model/gate0_regret.py`:
```python
#!/usr/bin/env python3
"""Gate 0 (Solução 4): antes de treinar, valida que o sinal de regret é
computável a partir do dataset_h9 — cobertura de none_rdcost, variância do
regret, ordenação esperada e fração censurada (folhas retangulares). Ver spec §4."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as datamod  # noqa: E402
import build_regret_targets as brt  # noqa: E402
from partition_defs import MODEL_LEVELS  # noqa: E402

# Critérios de PASSAGEM (propostas iniciais; calibráveis).
MIN_EXACT_FRACTION = 0.40   # >=40% dos nós de decisão são exatos (sem censura)
MIN_REGRET_STD = 1e-3       # regret tem variância real (não é degenerado)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--per-pkl", type=int, default=500)
    p.add_argument("--out", default="/workspace/results/models/regret/gate0.csv")
    args = p.parse_args(argv)

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, brt.VAL_SEQS, brt.TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    alln = brt.collect_regret_by_dim(train_e, per_pkl=args.per_pkl,
                                     exact_only=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ok = True
    rows = ["dim,n,exact_frac,regret_std,regret_med,frac_zero"]
    for dim, _ in MODEL_LEVELS:
        v = alln[dim]
        n = len(v["regret"])
        if n == 0:
            print("[dim {}] SEM NÓS -> FALHA".format(dim))
            ok = False
            continue
        exact_frac = float(v["exact"].mean())
        std = float(v["regret"].std())
        med = float(np.median(v["regret"]))
        frac_zero = float((v["regret"] == 0).mean())
        rows.append("{},{},{:.4f},{:.6f},{:.6f},{:.4f}".format(
            dim, n, exact_frac, std, med, frac_zero))
        dim_ok = (exact_frac >= MIN_EXACT_FRACTION) and (std >= MIN_REGRET_STD)
        ok = ok and dim_ok
        print("[dim {:>2}] n={} exact={:.2%} std={:.4f} med={:.4f} "
              "zero={:.2%} -> {}".format(dim, n, exact_frac, std, med,
                                         frac_zero, "OK" if dim_ok else "FALHA"),
              flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Gate 0:", "PASSOU" if ok else "FALHOU", "->", args.out)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Rodar o Gate 0 sobre os pkls de treino**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 gate0_regret.py --per-pkl 500 2>&1 | tee /workspace/logs/gate0_regret.log'`
Expected: imprime por tamanho e termina com `Gate 0: PASSOU`. **Se FALHAR** (censura dominante ou regret degenerado), PARAR o plano e reportar — a hipótese A não é sustentada pelos dados (registrar em `docs/ANDAMENTO_tese.md`).

- [ ] **Step 3: Commit + push**

```bash
git add src/scripts/partition_model/gate0_regret.py results/models/regret/gate0.csv
git commit -m "solucao4: Gate 0 (viabilidade do sinal de regret) + resultado"
git push
```

---

## Task 4: `train_regret.py` — NN regressora (cabeça única)

**Files:**
- Create: `src/scripts/partition_model/train_regret.py`
- Test: `test_regret_head_shapes` em `tests/test_regret.py`

**Interfaces:**
- Consumes: `build_regret_targets.collect_regret_by_dim`; `student`/`model` (MLP existente) e `torch`.
- Produces: bundle salvo em `results/models/regret/students.pt` com `{"hidden":[64,32], "students":{dim:state_dict}, "norm":{dim:(mean,std)}, "num_features":36, "feature_set":"h9a", "head":"regret"}`. Função `build_regressor(in_features, hidden) -> torch.nn.Module` com **1 saída linear** (sem softmax).

- [ ] **Step 1: Escrever o teste de forma da rede**

Adicionar em `tests/test_regret.py`:
```python
def test_regret_head_shapes():
    import torch
    import train_regret as tr
    net = tr.build_regressor(in_features=36, hidden=[64, 32])
    x = torch.zeros((5, 36), dtype=torch.float32)
    y = net(x)
    assert y.shape == (5,)   # saída escalar por amostra (regressão)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py::test_regret_head_shapes -q'`
Expected: FAIL (`No module named 'train_regret'`).

- [ ] **Step 3: Implementar `train_regret.py`**

`src/scripts/partition_model/train_regret.py`:
```python
#!/usr/bin/env python3
"""Treina a NN regressora de regret da Solução 4: um MLP por tamanho de bloco
(mesma topologia [64,32] do estudante H9a), com CABEÇA DE REGRESSÃO ÚNICA. Alvo =
log1p(regret_rel). Perda Huber. Entrada = 36 features H9a (fonte única). O bundle é
compatível com export_weights/simulate (num_features/feature_set), com head='regret'."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import build_regret_targets as brt  # noqa: E402
import data as datamod  # noqa: E402


def build_regressor(in_features, hidden):
    layers, d = [], in_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers)

    class Reg(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            return self.body(x).squeeze(-1)

    return Reg(net)


def _fit_dim(feat, target, hidden, device, epochs, lr):
    mean = feat.mean(0)
    std = feat.std(0) + 1e-6
    xn = (feat - mean) / std
    x = torch.tensor(xn, dtype=torch.float32, device=device)
    y = torch.tensor(np.log1p(target), dtype=torch.float32, device=device)
    net = build_regressor(feat.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    return net, (mean.astype(np.float32), std.astype(np.float32)), float(loss.item())


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--out-dir", default="/workspace/results/models/regret")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-pkl", type=int, default=3000)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, brt.VAL_SEQS, brt.TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    data = brt.collect_regret_by_dim(train_e, per_pkl=args.per_pkl or None,
                                     exact_only=True)

    os.makedirs(args.out_dir, exist_ok=True)
    bundle = {"hidden": args.hidden, "students": {}, "norm": {},
              "num_features": 36, "feature_set": "h9a", "head": "regret"}
    for dim, _ in MODEL_LEVELS:
        rec = data[dim]
        if len(rec["regret"]) == 0:
            print("[dim {}] sem amostras".format(dim))
            continue
        net, norm, loss = _fit_dim(rec["feat"], rec["regret"], args.hidden,
                                   device, args.epochs, args.lr)
        print("[dim {:>2}] n={} huber={:.5f}".format(
            dim, len(rec["regret"]), loss), flush=True)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm
    torch.save(bundle, os.path.join(args.out_dir, "students.pt"))
    print("Saved ->", os.path.join(args.out_dir, "students.pt"))


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Rodar o teste de forma (passar)**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 -m pytest tests/test_regret.py::test_regret_head_shapes -q'`
Expected: PASS.

- [ ] **Step 5: Treinar de fato (smoke reduzido primeiro)**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 train_regret.py --per-pkl 300 --epochs 50 2>&1 | tee /workspace/logs/train_regret_smoke.log'`
Expected: imprime `huber=` finito por tamanho e salva `students.pt`. Depois rodar o treino completo (`--per-pkl 3000 --epochs 200`) e guardar o log.

- [ ] **Step 6: Commit + push**

```bash
git add -f src/scripts/partition_model/train_regret.py src/scripts/partition_model/tests/test_regret.py results/models/regret/students.pt
git commit -m "solucao4: train_regret.py (MLP regressor de regret, cabeça única) + modelo treinado"
git push
```

---

## Task 5: Gate 2/3 — modo *regret-score* no simulador de oráculo (milestone)

**Files:**
- Modify: `src/scripts/partition_model/simulate_pruning.py`

**Interfaces:**
- Consumes: o bundle `results/models/regret/students.pt` (`head=="regret"`).
- Produces: no simulador, uma **fonte de escore** `regret` que ordena a poda por regret predito (poda quando `regret_pred < τ_regret`), comparável, no mesmo arcabouço de "custo casado por SPLIT-lost", com o classificador H9a e a variância.

- [ ] **Step 1: Ler o simulador e localizar o ponto de escore**

Run: `docker exec av1_bench bash -lc 'grep -n "def " /workspace/src/scripts/partition_model/simulate_pruning.py | head -40'`
Objetivo: identificar a função que produz, por nó, o escore que a política limiariza (o mesmo ponto onde `variance`/`pixels`/`H9a` já são plugados — ver o argumento de baseline/score existente citado em `docs/ANDAMENTO_tese.md`).

- [ ] **Step 2: Adicionar o carregamento e o escore de regret**

No `simulate_pruning.py`, no mesmo lugar onde os escores existentes (H9a-classificador, variância) são selecionados por uma flag `--score`/`--baseline`, acrescentar o ramo `regret`:
```python
# --- Solução 4: escore por regret predito ---------------------------------
# Carrega o bundle de regressão (head=='regret') e prediz log1p(regret) por nó;
# a política poda quando o regret predito fica ABAIXO de tau_regret (custo baixo).
def _load_regret_bundle(path):
    import torch
    b = torch.load(path, map_location="cpu")
    assert b.get("head") == "regret", "bundle não é de regressão de regret"
    return b

def _regret_scores(bundle, dim, feats):
    """feats: (N,36) float32 já na MESMA convenção de node_features_h9a.
    Retorna a saída CRUA da rede == log1p(regret) — a MESMA escala que o C
    compara (Task 7), para que τ_regret seja idêntico offline e deployado."""
    import torch
    import train_regret
    mean, std = bundle["norm"][dim]
    net = train_regret.build_regressor(bundle["num_features"], bundle["hidden"])
    net.load_state_dict(bundle["students"][dim])
    net.eval()
    xn = (feats - mean) / std
    with torch.no_grad():
        return net(torch.tensor(xn, dtype=torch.float32)).numpy()  # log1p-escala
```
Ligar `--score regret` para: podar o nó se `_regret_scores(...) < tau_regret` (τ na escala `log1p`, **a mesma do C**). Reusar a MESMA malha de avaliação ("custo casado por SPLIT-lost") que já compara os escores existentes, de modo que a saída seja diretamente comparável (mesma tabela do Gate 2/3).

- [ ] **Step 3: Gate 2 (treino) — rodar e comparar**

Run:
```bash
docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 simulate_pruning.py --dataset-dir /workspace/results/dataset_h9 --score regret --regret-bundle /workspace/results/models/regret/students.pt --compare h9a variance 2>&1 | tee /workspace/logs/gate2_regret.log'
```
Expected: tabela de redução de custo a SPLIT-lost casado {0.5/1/2%} com colunas `regret`, `h9a`, `variance`. **Critério de PASSAGEM:** `regret` ≥ `h9a` **e** > `variance` em risco baixo. (Ajustar nomes de flags aos existentes no simulador — Step 1.)

- [ ] **Step 4: Gate 3 (validação) — HoneyBee/FlowerPan/Lips**

Run o mesmo com `--val-seqs HoneyBee FlowerPan Lips` (ou o mecanismo de split do simulador), salvando `logs/gate3_regret.log`.
Expected: mesmo padrão na validação. **Se `regret` não superar `h9a` em nenhum regime**, registrar como resultado (o alvo contínuo não separa do classificador → confirma teto; ainda defensável), e decidir com o usuário se segue ao Gate 5.

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/partition_model/simulate_pruning.py results/models/regret/
git commit -m "solucao4: modo regret-score no simulador + Gates 2/3 (oráculo)"
git push
```

---

## Task 6: `export_weights.py` — header C da cabeça de regressão

**Files:**
- Modify: `src/scripts/partition_model/export_weights.py`

**Interfaces:**
- Consumes: bundle `students.pt` com `head=="regret"`, `num_features==36`.
- Produces: `src/aom/av1/encoder/partition_regret_weights.h` — pesos/normalização por tamanho e **uma saída** (formato consumível por `av1_nn_predict`, análogo ao header do H9a mas com `NUM_OUTPUTS=1`). Não sobrescreve `partition_student_weights.h` (H9a implantado).

- [ ] **Step 1: Ler o exporter atual**

Run: `docker exec av1_bench bash -lc 'grep -n "def \|NUM_OUTPUTS\|softmax\|header\|write" /workspace/src/scripts/partition_model/export_weights.py | head -40'`
Objetivo: reusar a rotina que serializa `nn_config` do H9a, mudando a última camada para 1 saída e emitindo um nome de símbolo/arquivo dedicado (`partition_regret_weights.h`).

- [ ] **Step 2: Adicionar o caminho `--head regret`**

Estender `export_weights.py` para, quando o bundle tem `head=="regret"`, emitir o header com `NUM_OUTPUTS=1`, símbolos `av1_partition_regret_nnconfig_{64,32,16}` e o vetor de normalização (mean/std) por tamanho. Reusar a serialização de camadas já existente (mesma estrutura `av1_nn_predict`).

- [ ] **Step 3: Gerar o header**

Run: `docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && python3 export_weights.py --bundle /workspace/results/models/regret/students.pt --head regret --out /workspace/src/aom/av1/encoder/partition_regret_weights.h'`
Expected: cria o header; `grep -c nnconfig` mostra 3 símbolos.

- [ ] **Step 4: Commit + push**

```bash
git add src/scripts/partition_model/export_weights.py src/aom/av1/encoder/partition_regret_weights.h
git commit -m "solucao4: export da cabeça de regressão para header C (partition_regret_weights.h)"
git push
```

---

## Task 7: Integração C — `student_regret_decide` (Gate 4: paridade + no-op)

**Files:**
- Modify: `src/aom/av1/encoder/partition_strategy.c`
- Modify: `src/scripts/partition_model/check_feature_parity.py` (validar o escalar de regressão)

**Interfaces:**
- Consumes: `partition_regret_weights.h`; as 36 features já preenchidas por `student_node_features` (Fase 4, reuso total).
- Produces: função de decisão pré-busca por regret, ativada por env, no-op por default.

- [ ] **Step 1: Localizar o padrão H9a/H9c a espelhar**

Run: `docker exec av1_bench bash -lc 'grep -n "student_node_features\|try_student_prune\|student_h9c_decide\|av1_nn_predict\|PARTITION_ML_STUDENT" /workspace/src/aom/av1/encoder/partition_strategy.c | head'`
Objetivo: usar `student_node_features` (features 36, já com paridade) e o padrão de `av1_nn_predict` como base do novo caminho.

- [ ] **Step 2: Adicionar `student_regret_decide` (sob `#if PARTITION_ML_STUDENT`)**

Inserir, junto aos demais helpers do student, lendo as envs uma vez (cacheadas):
```c
// Solução 4: poda pré-busca por regret predito (cabeça de regressão única).
// Ativada só quando AV1_STUDENT_REGRET_ENABLE=1; default = ausente = no-op.
// Poda (comprometa NONE) quando o regret predito < AV1_STUDENT_TAU_REGRET.
static int av1_student_regret_enabled(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *e = getenv("AV1_STUDENT_REGRET_ENABLE");
    cached = (e && e[0] == '1') ? 1 : 0;
  }
  return cached;
}

static float av1_student_regret_tau(void) {
  static float cached = -1.0f;
  if (cached < 0.0f) {
    const char *e = getenv("AV1_STUDENT_TAU_REGRET");
    cached = e ? (float)atof(e) : 0.0f;  // 0 => nunca poda (no-op mesmo se ligado)
  }
  return cached;
}

// Retorna 1 se o nó deve ser podado (comprometer NONE). `feats` são as 36
// features H9a já montadas por student_node_features; `nnconfig` é o header de
// regressão para este tamanho (av1_partition_regret_nnconfig_{64,32,16}).
static int student_regret_decide(const NN_CONFIG *nnconfig,
                                 const float *feats) {
  float out[1];
  av1_nn_predict(feats, nnconfig, 1, out);
  const float regret_pred = out[0];  // == log1p(regret) na escala de treino
  return (regret_pred < av1_student_regret_tau()) ? 1 : 0;
}
```
Ligar `student_regret_decide` no MESMO ponto pré-busca onde o H9a decide comprometer NONE, atrás de `if (av1_student_regret_enabled())`. **Nota de escala:** o header emite os pesos treinados sobre `log1p(regret_rel)`; portanto `τ_regret` também está nessa escala (documentar em `ARQUITETURA_pruner_implantado.md`).

- [ ] **Step 3: Build de paridade/validação**

Run: `docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_ml_check -j"$(nproc)" 2>&1 | tail -5'`
Expected: build limpo (o header novo compila).

- [ ] **Step 4: Gate 4a — no-op byte-idêntico (env ausente)**

Codificar 1 frame com o build do student SEM as envs e comparar md5 com a âncora:
```bash
docker exec av1_bench bash -lc '/workspace/build/libaom_noop/aomenc ...<mesmos args de 1 frame>... -o /tmp/a.obu <seq> && md5sum /tmp/a.obu'
docker exec av1_bench bash -lc 'AV1_STUDENT_TAU_REGRET= /workspace/build/libaom_perf/aomenc ...<idem>... -o /tmp/b.obu <seq> && md5sum /tmp/b.obu'
```
Expected: **md5 idêntico** (no-op). (Reusar o comando exato de 1 frame de `docs/RESULTADOS_fase6.md §2`.)

- [ ] **Step 5: Gate 4b — paridade do escalar C↔Python**

Estender `check_feature_parity.py` para, além das 36 features, comparar a saída de regressão do C (`student_regret_decide`/dump) contra `train_regret.build_regressor` em Python sobre as mesmas features. Rodar e exigir |Δ| < 1e-3 nos nós amostrados.

- [ ] **Step 6: Commit + push**

```bash
git add src/aom/av1/encoder/partition_strategy.c src/scripts/partition_model/check_feature_parity.py
git commit -m "solucao4: integração C do pruner por regret (env-gated, no-op byte-idêntico, paridade)"
git push
```

---

## Task 8: Gate 5 — benchmark real no teste reservado + *swap* de escore (milestone)

**Files:**
- Modify: `src/scripts/fase6/encode_swap.py` e `report_swap.py` (adicionar a fonte `regret`), OU criar `src/scripts/fase6/encode_regret.py` reaproveitando o arcabouço.
- Create: `docs/RESULTADOS_solucao4.md`

**Interfaces:**
- Consumes: `build/libaom_perf` (com o header de regret) e `build/libaom_perf_anchor` (âncora cpu0).
- Produces: BD-Rate/TS%/speedup no teste reservado (Jockey/RaceNight/RiverBank) e a comparação decisiva **swap de escore sob política casada** `{regret, h9a-classificador, variância, aleatório}`.

- [ ] **Step 1: Rebuild do benchmark**

Run: `docker exec av1_bench bash -lc 'cmake --build /workspace/build/libaom_perf -j"$(nproc)" 2>&1 | tail -5'`
Expected: build limpo com o caminho de regret compilado.

- [ ] **Step 2: Curva BD×TS (teste reservado, cpu0)**

Varrer `AV1_STUDENT_REGRET_ENABLE=1` + `AV1_STUDENT_TAU_REGRET ∈ {grade}` (calibrada na validação, Task 5) sobre Jockey/RaceNight/RiverBank, ≥10 frames, cq 20/32/43/55, cpu-used=0, single-thread, vs âncora `libaom_perf_anchor` cpu0. Reusar `run_full.sh`/`matched_bd.py`. Salvar em `results/benchmark/solucao4/`.

- [ ] **Step 3: Swap de escore sob política casada (o resultado defensável)**

Com a MESMA cascata NONE-commit, medir os quatro escores `{regret, h9a, variance, random}` a *speedup casado*. **Critério de vitória (bônus):** `regret` atinge BD menor ao mesmo speedup que `h9a`. **Resultado garantido:** empatar confirma o teto independente do objetivo (registrar como achado). Reusar `ablation_attrib.py`/`analyze_ablation.py` adicionando a fonte `regret`.

- [ ] **Step 4: Escrever `docs/RESULTADOS_solucao4.md`**

Documentar configuração, curva BD×TS, tabela do swap, e a leitura honesta (vitória de fronteira OU confirmação de teto), no mesmo padrão de `RESULTADOS_fase5.md`.

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/fase6/ docs/RESULTADOS_solucao4.md results/benchmark/solucao4/
git commit -m "solucao4: Gate 5 (benchmark teste reservado + swap de escore) + resultados"
git push
```

---

## Task 9 (opcional, bônus): Gate 6 — verificação de fronteira na CTC

**Files:**
- Modify: `src/scripts/fase6/run_swap.sh`/`report_swap.py` (config `regret` como *pruner* único vs CNN nativa em cpu1/2/3).
- Modify: `docs/RESULTADOS_solucao4.md`

- [ ] **Step 1: Swap ML-vs-ML na CTC (8 seqs A1)**

Com `AV1_DISABLE_NATIVE_CNN=1` + o pruner de regret como único, a cpu1/2/3, medir vs CNN nativa (reaproveitando os `native_cpuN` já no CSV da Fase 6). **Pergunta:** o pruner por regret fura a fronteira BD×TS da CNN nativa em algum ponto?

- [ ] **Step 2: Atualizar `RESULTADOS_solucao4.md` + commit + push**

```bash
git add docs/RESULTADOS_solucao4.md results/benchmark/solucao4/
git commit -m "solucao4: Gate 6 (verificação de fronteira na CTC, ML-vs-CNN nativa)"
git push
```

---

## Task 10: Documentação de fechamento + Approach B (trabalho futuro)

**Files:**
- Modify: `docs/SINTESE_resultados_metodologia.md` (nova seção "Solução 4")
- Modify: `docs/ANDAMENTO_tese.md`

- [ ] **Step 1: Nova seção "Solução 4" na síntese**

Adicionar a Solução 4 (NN regressora de regret) ao documento vivo: motivação (o alvo certo é custo, não rótulo), método (regret da árvore comprometida, sem re-extração), gates e resultado (fronteira ou teto), e como ela se encaixa nas três conclusões.

- [ ] **Step 2: Registrar a Approach B como teste final/futuro**

Em `ANDAMENTO_tese.md`, documentar a **NN estruturada em árvore** (recursiva/GNN sobre o contexto RD, decisão conjunta do quadtree, possivelmente prevendo regret de forma estruturada) como o **teste de fechamento da tese** — o teste direto de se o teto da Conclusão 3 é informacional ou artefato de nós independentes. Não implementar agora.

- [ ] **Step 3: Commit + push**

```bash
git add docs/SINTESE_resultados_metodologia.md docs/ANDAMENTO_tese.md
git commit -m "solucao4: síntese atualizada + Approach B (NN estruturada) como teste final da tese"
git push
```

---

## Notas de execução / ameaças à validade

- **Oráculo superestima o tempo (~5×):** Gates 2/3 valem pelas margens **relativas**; o árbitro é o Gate 5 (tempo de parede).
- **Censura retangular:** o treino primário usa `exact_only=True`. Isso pode enviesar (retangulares ocorrem em blocos texturizados). Um refinamento (perda unilateral incorporando nós censurados/own-rect) é a variante de controle prevista no spec §3.3 — abrir como tarefa extra se o Gate 5 empatar e se quiser explorar a margem.
- **Reconstrução depende de logging completo:** o Gate 0 (Task 3) é a salvaguarda; se a fração exata for baixa demais, a hipótese A cai ali, a custo quase zero.
- **Escala do τ_regret:** os pesos são treinados sobre `log1p(regret_rel)`; τ está nessa escala. A grade de τ é calibrada na validação (Task 5) e **congelada** antes do teste (Task 8), sem test-tuning.
```
