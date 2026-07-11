# Fase 3 — Estudante tabular H9a — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar o estudante implantável (MLP por tamanho de bloco) diretamente sobre o vetor de features H9a (36 entradas), sobre `dataset_h9`, e validá-lo na simulação oráculo contra o baseline de variância.

**Architecture:** Reúso máximo do pipeline existente. Um novo script `train_student_h9.py` coleta `(feat_H9a, rótulo)` por tamanho de bloco e treina com CE de rótulo duro (sem professor) reutilizando `distill.train_student` (com fold-in de padronização). Os scripts a jusante (`export_weights.py`, `simulate_pruning.py`) passam a ler a contagem de features do próprio bundle, em vez de assumir 24 fixo.

**Tech Stack:** Python 3, PyTorch, NumPy. Execução **exclusivamente no container Docker** `av1_bench`, com o interpretador `/workspace/build/venv-ml/bin/python` (o host Windows é só edição). Repositório montado em `/workspace`.

## Global Constraints

- **Execução sempre no container** `av1_bench`: `docker exec av1_bench /workspace/build/venv-ml/bin/python ...`. Se o container estiver parado: `docker start av1_bench`.
- **Split congelado 10/3/3 (inegociável)** — `docs/PROTOCOLO_avaliacao.md`:
  - Treino (10): `Beauty Bosphorus CityAlley FlowerFocus FlowerKids ReadySetGo ShakeNDry SunBath Twilight YachtRide`
  - Validação (3): `HoneyBee FlowerPan Lips`
  - Teste (3): `Jockey RaceNight RiverBank` — **nunca tocados nesta fase**.
- **Conjunto de features H9a** = `features.node_features_h9(...)[:36]` = A(0..23 pixels) + B(24..31 vizinhança) + C(32..35 quant/pos). D/E **fora** do modelo implantado.
- **Sem rebuild C nesta fase** (integração C = Fase 4). O header exportado vai para `results/models/student_h9a/`, **não** sobrescreve o `partition_student_weights.h` implantado (do `student_real`).
- **Sem menção a IA/Claude/Co-Authored-By** nas mensagens de commit.
- Sem suíte pytest no projeto: a verificação de cada tarefa é um **smoke run no container** com saída esperada, e o gate final é a simulação oráculo.

---

### Task 1: Parametrizar `distill.train_student` com `in_features`

Torna a função de treino (com fold-in de padronização) reutilizável para vetores de largura diferente de 24, sem quebrar a destilação existente.

**Files:**
- Modify: `src/scripts/partition_model/distill.py:98-111`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `distill.train_student(rec, hidden, device, epochs, lr, alpha, temp, wd=1e-4, batch=4096, use_class_weight=True, in_features=None)` — quando `in_features` é `None`, usa `featmod.NUM_FEATURES` (comportamento atual); caso contrário constrói a MLP com `in_features` entradas. Retorna `(net, {"mean","std"})` como hoje.

- [ ] **Step 1: Verificar que a chamada com `in_features` falha hoje (red)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python -c "
import sys; sys.path.insert(0,'/workspace/src/scripts/partition_model')
import inspect, distill
print('in_features' in inspect.signature(distill.train_student).parameters)
"
```
Expected: imprime `False` (o parâmetro ainda não existe).

- [ ] **Step 2: Editar a assinatura e a construção da rede**

Em `src/scripts/partition_model/distill.py`, trocar a assinatura (linha 98-99):
```python
def train_student(rec, hidden, device, epochs, lr, alpha, temp, wd=1e-4,
                  batch=4096, use_class_weight=True, in_features=None):
```
E trocar a linha que constrói a rede (atual linha 111):
```python
    nfeat = featmod.NUM_FEATURES if in_features is None else in_features
    net = studentmod.make_student(nfeat, hidden).to(device)
```

- [ ] **Step 3: Verificar que o parâmetro existe e a destilação ainda importa (green)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python -c "
import sys; sys.path.insert(0,'/workspace/src/scripts/partition_model')
import inspect, distill
p = inspect.signature(distill.train_student).parameters
assert 'in_features' in p and p['in_features'].default is None
print('OK in_features param present, default None')
"
```
Expected: imprime `OK in_features param present, default None` sem erro.

- [ ] **Step 4: Commit**

```bash
git add src/scripts/partition_model/distill.py
git commit -m "Parametrize train_student with in_features (H9a reuse)"
```

---

### Task 2: Novo `train_student_h9.py` + constante `NUM_FEATURES_H9A`

Treinador direto do estudante implantável sobre H9a. Reúsa `distill.train_student` (Task 1).

**Files:**
- Modify: `src/scripts/partition_model/features.py` (adicionar constante após `NUM_FEATURES_H9 = 41`, ~linha 203)
- Create: `src/scripts/partition_model/train_student_h9.py`

**Interfaces:**
- Consumes: `distill.train_student(..., in_features=36, alpha=1.0, use_class_weight=False)`; `features.node_features_h9(sb_luma, dim, r, c, qindex, ctx)`; `features.NUM_FEATURES_H9A`; `data.iter_superblock_members`, `data.split_entries`, `data.discover_pkls`, `data.assert_real_luma`; `student.collapse_label`; `partition_defs.MODEL_LEVELS`.
- Produces: bundle salvo em `<out-dir>/students.pt` = `{"hidden":[64,32], "students":{dim:state_dict}, "norm":{dim:{"mean","std"}}, "num_features":36, "feature_set":"h9a"}`. Consumido por Task 3 (export) e Task 4 (simulate).

- [ ] **Step 1: Adicionar a constante em `features.py`**

Logo após `NUM_FEATURES_H9 = 41` (~linha 203), adicionar:
```python
NUM_FEATURES_H9A = 36  # A(0..23) + B(24..31) + C(32..35); D/E excluded from deploy
```

- [ ] **Step 2: Verificar que o script ainda não existe (red)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/train_student_h9.py --help
```
Expected: FALHA com `No such file or directory` / `can't open file`.

- [ ] **Step 3: Criar `train_student_h9.py`**

Conteúdo completo:
```python
#!/usr/bin/env python3
"""Fase 3 -- train the deployable per-block-size student directly on the H9a
feature vector (A pixels + B neighbor context + C quant/position = 36 features)
with hard-label cross-entropy (no teacher). Reproduces, in the deployable
artifact, the winning Gate 2 configuration.

The distilled ConvNeXt teacher is deliberately not used: it is pixel-only and
cannot carry the B/C signal that dataset_h9 exists to capture (see
docs/ANDAMENTO_tese.md 1.2 / 3). The output bundle is a superset of the pixel
student's, adding num_features and feature_set so export_weights.py and
simulate_pruning.py score with the right input width.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from distill import train_student  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def collect_by_dim(entries, per_pkl=None, limit=None):
    """Per-block-size {'feat':(N,36), 'truth':(N,)} arrays over the
    H9-instrumented dataset. per_pkl caps superblocks taken from each pkl
    (diverse sampling across seqs/QPs); limit is a global superblock cap."""
    nfa = featmod.NUM_FEATURES_H9A
    acc = {dim: {"feat": [], "truth": []} for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: no RD context; re-extract with the H9 "
                                 "instrumentation.".format(e["path"]))
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in acc:
                    continue
                ctx = dict(sb["ctx"][k])
                ctx["bsize_enum"] = -1
                f = featmod.node_features_h9(sb["luma"], dim, r, c,
                                             sb["qindex"], ctx)[:nfa]
                acc[dim]["feat"].append(f)
                acc[dim]["truth"].append(studentmod.collapse_label(label))
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and n_sb >= limit:
                return _finalize(acc), n_sb
    return _finalize(acc), n_sb


def _finalize(acc):
    return {dim: {"feat": np.asarray(v["feat"], dtype=np.float32),
                  "truth": np.asarray(v["truth"], dtype=np.int64)}
            for dim, v in acc.items()}


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--out-dir", default="/workspace/results/models/student_h9a")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-pkl", type=int, default=3000,
                   help="cap superblocks per pkl (diverse sampling); 0 = all")
    p.add_argument("--limit", type=int, default=None,
                   help="global superblock cap (smoke)")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, args.val_seqs, args.train_seqs)
    datamod.assert_real_luma(train_e)
    print("train pkls: {} (train seqs {})".format(
        len(train_e), args.train_seqs), flush=True)
    per_pkl = args.per_pkl or None
    data, n_sb = collect_by_dim(train_e, per_pkl=per_pkl, limit=args.limit)
    print("collected {} superblocks".format(n_sb), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    nfa = featmod.NUM_FEATURES_H9A
    bundle = {"hidden": args.hidden, "students": {}, "norm": {},
              "num_features": nfa, "feature_set": "h9a"}
    for dim, _ in MODEL_LEVELS:
        rec = data[dim]
        if len(rec["truth"]) == 0:
            print("[dim {}] no samples, skipping".format(dim))
            continue
        rec = {"feat": rec["feat"], "truth": rec["truth"],
               "teacher": np.full((len(rec["truth"]), 3), 1.0 / 3.0,
                                  dtype=np.float32)}
        net, norm = train_student(rec, args.hidden, device, args.epochs,
                                  args.lr, alpha=1.0, temp=1.0,
                                  use_class_weight=False, in_features=nfa)
        with torch.no_grad():
            fb = torch.tensor(rec["feat"], dtype=torch.float32, device=device)
            pred = net(fb).argmax(-1).cpu().numpy()
        counts = np.bincount(rec["truth"], minlength=3).tolist()
        acc = float((pred == rec["truth"]).mean())
        print("[dim {:>2}] n={} counts(N/S/R)={} truth-acc={:.3f}".format(
            dim, len(pred), counts, acc), flush=True)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm

    torch.save(bundle, os.path.join(args.out_dir, "students.pt"))
    print("Saved ->", os.path.join(args.out_dir, "students.pt"))


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Smoke run (green) — treino minúsculo em 1 seq de treino**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/train_student_h9.py \
  --train-seqs Beauty --val-seqs HoneyBee \
  --out-dir /workspace/results/models/student_h9a_smoke \
  --per-pkl 0 --limit 40 --epochs 3
```
Expected: imprime `collected 40 superblocks`, três linhas `[dim 64|32|16] n=... truth-acc=...`, e `Saved -> .../student_h9a_smoke/students.pt` sem erro.

- [ ] **Step 5: Verificar o contrato do bundle (num_features=36, 1ª camada com 36 entradas)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python -c "
import torch
b = torch.load('/workspace/results/models/student_h9a_smoke/students.pt', map_location='cpu')
assert b['num_features'] == 36, b['num_features']
assert b['feature_set'] == 'h9a', b['feature_set']
for dim, sd in b['students'].items():
    w0 = sd['0.weight']  # first Linear
    assert w0.shape[1] == 36, (dim, tuple(w0.shape))
print('OK bundle contract: num_features=36, first-layer in=36, dims', sorted(b['students']))
"
```
Expected: imprime `OK bundle contract: num_features=36, first-layer in=36, dims [16, 32, 64]`.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/partition_model/features.py src/scripts/partition_model/train_student_h9.py
git commit -m "Add direct H9a student trainer (hard-label, per block size)"
```

---

### Task 3: `export_weights.py` ciente da contagem de features

O exportador deve emitir o header com o número de entradas do bundle (36 para H9a), não 24 fixo. Contrato explícito para a Fase 4 (o C computará 36 features).

**Files:**
- Modify: `src/scripts/partition_model/export_weights.py:80-110`

**Interfaces:**
- Consumes: bundle da Task 2 (`num_features`, `hidden`, `students`).
- Produces: header C com `AV1_PARTITION_STUDENT_NUM_FEATURES <nfeat>` e cada `NN_CONFIG` com `num_inputs = <nfeat>`.

- [ ] **Step 1: Verificar que hoje o header sai com 24 mesmo para bundle H9a (red)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/export_weights.py \
  --students /workspace/results/models/student_h9a_smoke/students.pt \
  --out /workspace/results/models/student_h9a_smoke/weights.h ; \
docker exec av1_bench grep -c "AV1_PARTITION_STUDENT_NUM_FEATURES 24" \
  /workspace/results/models/student_h9a_smoke/weights.h
```
Expected: o export **falha** ao carregar o state_dict (mismatch 24 vs 36 em `make_student(NUM_FEATURES,...)`), OU o grep encontra `... 24` — de todo modo, o header está errado. (Se falhar no load, esse é o red.)

- [ ] **Step 2: Ler `num_features` do bundle e usar em todo o header**

Em `src/scripts/partition_model/export_weights.py`, dentro de `main` (após `bundle = torch.load(...)`, ~linha 80), adicionar:
```python
    nfeat = bundle.get("num_features", featmod.NUM_FEATURES)
```
Trocar a construção da rede (atual linha 85):
```python
        net = studentmod.make_student(nfeat, bundle["hidden"])
```
Trocar as duas referências a `featmod.NUM_FEATURES` no bloco do header (comentário ~linha 103-105 e `#define` ~linha 109-110) por `nfeat`:
```python
        + "\n// Generated by src/scripts/partition_model/export_weights.py -- "
          "do not edit.\n"
          "// Distilled AV1 partition student: per-size MLP, features from\n"
          "// src/scripts/partition_model/features.py (NUM_FEATURES={}).\n".format(
              nfeat)
        + "#ifndef {g}\n#define {g}\n\n".format(g=guard)
        + '#include "av1/encoder/ml.h"\n#include "av1/common/enums.h"\n\n'
        + "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
        + "#define AV1_PARTITION_STUDENT_NUM_FEATURES {}\n\n".format(nfeat)
```

- [ ] **Step 3: Verificar que o header H9a sai com 36 (green)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/export_weights.py \
  --students /workspace/results/models/student_h9a_smoke/students.pt \
  --out /workspace/results/models/student_h9a_smoke/weights.h && \
docker exec av1_bench grep -m1 "AV1_PARTITION_STUDENT_NUM_FEATURES" \
  /workspace/results/models/student_h9a_smoke/weights.h && \
docker exec av1_bench grep -m1 "num_inputs" \
  /workspace/results/models/student_h9a_smoke/weights.h
```
Expected: `#define AV1_PARTITION_STUDENT_NUM_FEATURES 36` e uma linha `  36,  // num_inputs`.

- [ ] **Step 4: Regressão — bundle de 24 (pixels) ainda sai com 24**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/export_weights.py \
  --students /workspace/results/models/student_real/students.pt \
  --out /tmp/weights_pixels.h && \
docker exec av1_bench grep -m1 "AV1_PARTITION_STUDENT_NUM_FEATURES" /tmp/weights_pixels.h
```
Expected: `#define AV1_PARTITION_STUDENT_NUM_FEATURES 24` (fallback preservado para bundles sem `num_features`).

- [ ] **Step 5: Commit**

```bash
git add src/scripts/partition_model/export_weights.py
git commit -m "Make export_weights emit the bundle's feature count (24 or 36)"
```

---

### Task 4: `simulate_pruning.py` — modo H9a + baseline de variância

Habilita o Gate 3b: pontuar o estudante H9a (36 features) e comparar com o baseline de variância na validação, com resumo de custo em risco casado.

**Files:**
- Modify: `src/scripts/partition_model/simulate_pruning.py:66-106` (collect + scoring) e `:254-304` (main)

**Interfaces:**
- Consumes: bundle da Task 2 (`num_features`, `feature_set`); `features.node_features_h9`; `partition_defs.MODEL_LEVELS`.
- Produces: CLI com `--baseline variance` e resumo "cost_red% em split-lost<=cap" impresso ao final; scoring do estudante usa a largura correta de features.

- [ ] **Step 1: Verificar que hoje simulate não carrega o estudante H9a (red)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir /workspace/results/dataset_h9 --val-seqs HoneyBee \
  --students /workspace/results/models/student_h9a_smoke/students.pt \
  --limit 30 --out-csv /tmp/sim_smoke.csv
```
Expected: FALHA — `make_student(NUM_FEATURES=24)` não carrega o state_dict de 36 (size mismatch) e/ou `node_features` de 24 é passado à rede de 36.

- [ ] **Step 2: Tornar `collect_superblocks` ciente do feature_set**

Trocar `collect_superblocks` (linhas 66-82) por:
```python
def collect_superblocks(entries, feature_set="pixels24", limit=None):
    """List of superblocks: {nodes:{(dim,r,c):{truth,feat}}, luma, qindex}.
    feature_set: 'pixels24' -> node_features (24); 'h9a' -> node_features_h9[:36]
    with the per-node RD context (blocks B/C)."""
    nfa = featmod.NUM_FEATURES_H9A
    sbs = []
    for e in entries:
        for sb in datamod.iter_superblock_members(e["path"]):
            nodes = {}
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if feature_set == "h9a":
                    ctx = dict(sb["ctx"][k])
                    ctx["bsize_enum"] = -1
                    feat = featmod.node_features_h9(sb["luma"], dim, r, c,
                                                    sb["qindex"], ctx)[:nfa]
                else:
                    feat = featmod.node_features(sb["luma"], dim, r, c,
                                                 sb["qindex"])
                nodes[(dim, r, c)] = {"truth": label, "feat": feat}
            sbs.append({"nodes": nodes, "luma": sb["luma"],
                        "qindex": sb["qindex"]})
            if limit and len(sbs) >= limit:
                return sbs
    return sbs
```

- [ ] **Step 3: `score_with_student` usa a largura do bundle + baseline de variância**

Trocar `make_student(featmod.NUM_FEATURES, ...)` em `score_with_student` (linha 90) por:
```python
            nfeat = bundle.get("num_features", featmod.NUM_FEATURES)
            net = studentmod.make_student(nfeat, bundle["hidden"])
```
Adicionar, logo após a função `score_with_student` (após linha 106), a função de baseline e o relatório de classificação (Gate 3a):
```python
def score_with_variance(sbs, v0=1000.0):
    """Non-learned baseline: P(NONE)=exp(-var/v0) from feature 0 (log_var).
    Only modeled levels are scored; 8x8 leaves stay terminal, as the models do."""
    modeled = {d for d, _ in MODEL_LEVELS}
    for sb in sbs:
        for key, nd in sb["nodes"].items():
            if key[0] not in modeled:
                continue
            var = float(np.expm1(nd["feat"][0]))     # feature 0 is log1p(var)
            flat = float(np.exp(-var / v0))
            nd["prob"] = np.array([flat, 1.0 - flat, 0.0])
    return sbs


def classification_report(sbs):
    """Per-size macro-F1 and SPLIT-recall from the scored probs (argmax) vs the
    3-class collapsed truth (Gate 3a). truth is the raw 10-class PARTITION_TYPE;
    collapse_label maps it to [NONE, SPLIT, REST]."""
    per = {d: {"pred": [], "true": []} for d, _ in MODEL_LEVELS}
    for sb in sbs:
        for (dim, r, c), nd in sb["nodes"].items():
            if dim not in per or "prob" not in nd:
                continue
            per[dim]["pred"].append(int(np.argmax(nd["prob"])))
            per[dim]["true"].append(studentmod.collapse_label(nd["truth"]))
    print("\nper-size classification (Gate 3a): macro-F1 | SPLIT-recall")
    for dim, _ in MODEL_LEVELS:
        pred = np.array(per[dim]["pred"])
        true = np.array(per[dim]["true"])
        if len(true) == 0:
            continue
        f1s = []
        for cls in range(3):
            tp = int(((pred == cls) & (true == cls)).sum())
            fp = int(((pred == cls) & (true != cls)).sum())
            fn = int(((pred != cls) & (true == cls)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        split_tp = int(((pred == 1) & (true == 1)).sum())
        split_fn = int(((pred != 1) & (true == 1)).sum())
        srec = split_tp / (split_tp + split_fn) if split_tp + split_fn else 0.0
        print("  {:>2}px  macroF1={:.3f}  SPLITrecall={:.3f}".format(
            dim, float(np.mean(f1s)), srec))
```

- [ ] **Step 4: Selecionar modo + feature_set em `main`, e resumo de risco casado**

Em `main`, trocar o bloco de scoring (linhas 284-304) por:
```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, val_e = datamod.split_entries(entries, args.val_seqs, None)
    datamod.assert_real_luma(val_e)

    if args.baseline == "variance":
        feature_set, mode = "pixels24", "variance"
    elif args.surrogate:
        feature_set, mode = "pixels24", "surrogate"
    else:
        bundle = torch.load(args.students, map_location=device)
        feature_set, mode = bundle.get("feature_set", "pixels24"), "student"

    sbs = collect_superblocks(val_e, feature_set=feature_set, limit=args.limit)
    total_nodes = sum(len(s["nodes"]) for s in sbs)

    if mode == "surrogate":
        from model import PartitionSurrogate
        ckpt = torch.load(args.surrogate, map_location=device)
        sa = ckpt.get("args", {})
        net = PartitionSurrogate(sa.get("variant", "tiny"),
                                 sa.get("fusion_dim", 128)).to(device).eval()
        net.load_state_dict(ckpt["model"])
        sbs = score_with_surrogate(sbs, net, device)
        model_tag = "SURROGATE " + args.surrogate
    elif mode == "variance":
        sbs = score_with_variance(sbs, args.v0)
        model_tag = "VARIANCE-BASELINE v0={}".format(args.v0)
    else:
        sbs = score_with_student(sbs, bundle, device)
        model_tag = "STUDENT {} (feature_set={})".format(args.students,
                                                          feature_set)
    print("scored with:", model_tag)
    print("superblocks: {}, nodes: {} (val seqs {})".format(
        len(sbs), total_nodes, args.val_seqs))
    classification_report(sbs)
```
Adicionar os dois argumentos ao parser (junto aos demais `add_argument`, ~linha 263):
```python
    p.add_argument("--baseline", choices=["variance"], default=None,
                   help="score with the non-learned variance threshold instead "
                        "of a model (Gate 3 comparison)")
    p.add_argument("--v0", type=float, default=1000.0,
                   help="variance-baseline scale: P(NONE)=exp(-var/v0)")
```
Ao final de `main`, antes de `print("wrote", args.out_csv)` (após escrever o CSV, ~linha 377), adicionar o resumo de risco casado:
```python
    caps = [0.5, 1.0, 2.0]
    print("\ncost_red% at matched risk (split-lost cap):")
    print("  cap%   cost_red%")
    for cap in caps:
        feas = [r[4] for r in rows if r[6] <= cap]  # r[4]=cost_red, r[6]=split_lost
        print("  {:<5} {:8.2f}".format(cap, max(feas) if feas else 0.0))
```

- [ ] **Step 5: Smoke — estudante H9a e variância na mesma validação minúscula (green)**

Run:
```bash
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir /workspace/results/dataset_h9 --val-seqs HoneyBee \
  --students /workspace/results/models/student_h9a_smoke/students.pt \
  --tau-rest -1.0 0.2 --limit 30 --out-csv /tmp/sim_student.csv && \
docker exec av1_bench /workspace/build/venv-ml/bin/python \
  /workspace/src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir /workspace/results/dataset_h9 --val-seqs HoneyBee \
  --baseline variance --tau-rest -1.0 0.2 --limit 30 --out-csv /tmp/sim_var.csv
```
Expected: ambos rodam sem erro; o 1º imprime `scored with: STUDENT ... (feature_set=h9a)`, o bloco `per-size classification (Gate 3a)` (macroF1/SPLITrecall por tamanho) e um bloco `cost_red% at matched risk`; o 2º imprime `scored with: VARIANCE-BASELINE ...`.

- [ ] **Step 6: Commit**

```bash
git add src/scripts/partition_model/simulate_pruning.py
git commit -m "Add H9a feature mode + variance baseline to the oracle sim"
```

---

### Task 5: Execução da Fase 3 — treino, Gate 3, export (compute pesado)

Produz o artefato real da fase. **Não é código** — é a execução no container, com os gates. Rodar em background (treino sobre ~milhões de nós). Só exporta o header se os gates passarem.

**Files:**
- Produces: `results/models/student_h9a/students.pt`, `results/models/student_h9a/oracle_sim_student.csv`, `results/models/student_h9a/oracle_sim_var.csv`, `results/models/student_h9a/partition_student_weights.h` (contrato; **não** compilado).

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: bundle e CSVs de gate versionados; header em `student_h9a/` (não sobrescreve o implantado).

- [ ] **Step 1: Garantir o container ativo**

Run: `docker start av1_bench` (idempotente).
Expected: imprime `av1_bench`.

- [ ] **Step 2: Treinar o estudante H9a (split congelado, background)**

Run:
```bash
docker exec av1_bench bash -lc '
cd /workspace &&
nohup build/venv-ml/bin/python src/scripts/partition_model/train_student_h9.py \
  --dataset-dir results/dataset_h9 \
  --train-seqs Beauty Bosphorus CityAlley FlowerFocus FlowerKids ReadySetGo \
               ShakeNDry SunBath Twilight YachtRide \
  --val-seqs HoneyBee FlowerPan Lips \
  --out-dir results/models/student_h9a --per-pkl 3000 --epochs 30 \
  > results/models/student_h9a_train.log 2>&1 &
echo started'
```
Expected: `started`. Acompanhar com `docker exec av1_bench tail -f /workspace/results/models/student_h9a_train.log` até `Saved -> .../student_h9a/students.pt` e as três linhas `[dim ..] truth-acc=...`.

- [ ] **Step 3: Gate 3 — simulação oráculo na validação (estudante H9a vs variância vs pixels)**

Run (3 execuções na mesma validação: estudante H9a, baseline de variância, e o estudante de pixels `student_real` como referência do Gate 3a):
```bash
docker exec av1_bench bash -lc '
cd /workspace &&
build/venv-ml/bin/python src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir results/dataset_h9 --val-seqs HoneyBee FlowerPan Lips \
  --students results/models/student_h9a/students.pt \
  --tau-none 0.6 0.7 0.8 0.85 0.9 0.95 --tau-rest -1.0 0.1 0.2 0.3 \
  --out-csv results/models/student_h9a/oracle_sim_student.csv &&
build/venv-ml/bin/python src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir results/dataset_h9 --val-seqs HoneyBee FlowerPan Lips \
  --baseline variance \
  --tau-none 0.6 0.7 0.8 0.85 0.9 0.95 --tau-rest -1.0 0.1 0.2 0.3 \
  --out-csv results/models/student_h9a/oracle_sim_var.csv &&
build/venv-ml/bin/python src/scripts/partition_model/simulate_pruning.py \
  --dataset-dir results/dataset_h9 --val-seqs HoneyBee FlowerPan Lips \
  --students results/models/student_real/students.pt \
  --tau-none 0.6 0.7 0.8 0.85 0.9 0.95 --tau-rest -1.0 0.1 0.2 0.3 \
  --out-csv results/models/student_h9a/oracle_sim_pixels.csv'
```
Expected: cada execução imprime `per-size classification (Gate 3a)` e `cost_red% at matched risk (split-lost cap)`.

**Critérios dos gates:**
- **Gate 3a (sanidade):** o macro-F1 e o SPLIT-recall por tamanho do estudante H9a ≥ os do estudante de pixels (`student_real`). Esperado passar (o Gate 2 já mostrou H9a > pixels).
- **Gate 3b (decisivo):** para o estudante H9a, `cost_red%` em split-lost≤1% é **materialmente maior** que o da variância (baseline ≈ 0), consistente com o Gate 2 (H9a ~16-25% no lever NONE-commit vs variância 0).

Se o Gate 3b **não** passar, **parar** e reportar (não exportar, não seguir para a Fase 4) — recuo conforme `PLANO_H9` §7.

- [ ] **Step 4: Exportar o header (contrato Fase 4; sem sobrescrever o implantado)**

Só se o Step 3 passar. Run:
```bash
docker exec av1_bench bash -lc '
cd /workspace &&
build/venv-ml/bin/python src/scripts/partition_model/export_weights.py \
  --students results/models/student_h9a/students.pt \
  --out results/models/student_h9a/partition_student_weights.h &&
grep -m1 AV1_PARTITION_STUDENT_NUM_FEATURES \
  results/models/student_h9a/partition_student_weights.h'
```
Expected: `#define AV1_PARTITION_STUDENT_NUM_FEATURES 36`. O `src/aom/.../partition_student_weights.h` implantado (student_real) permanece intocado.

- [ ] **Step 5: Commit dos artefatos de gate (pequenos)**

```bash
git add results/models/student_h9a/students.pt \
        results/models/student_h9a/oracle_sim_student.csv \
        results/models/student_h9a/oracle_sim_var.csv \
        results/models/student_h9a/partition_student_weights.h
git commit -m "Fase 3: H9a student + Gate 3 oracle sim (student vs variance)"
```

- [ ] **Step 6: Limpeza dos artefatos de smoke**

```bash
docker exec av1_bench rm -rf /workspace/results/models/student_h9a_smoke
```
Expected: sem saída (sucesso).

---

## Notas de execução

- **`depth_log2` (idx 35) é constante dentro de um modelo por tamanho** → std≈0 → clampeado a 0.1 no fold-in → entrada inócua (≈0). Esperado; não é bug.
- **`per_pkl 3000`** mantém o treino tratável (~sub-amostragem diversa, como o Gate 2). Para treinar em todos os nós, usar `--per-pkl 0` (mais lento; a margem relativa não deve mudar).
- Se o Gate 3b (Step 3, Task 5) reprovar, a decisão de tese recua conforme `PLANO_H9` §7 (diagnóstico + teto H9c); **não** prosseguir para a Fase 4.
