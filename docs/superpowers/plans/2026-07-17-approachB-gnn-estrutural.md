# Approach B (Stage 1) — GNN estrutural diagnóstico do teto — Plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir, offline e barato, se decidir o quadtree do superbloco **conjuntamente** (GNN com *message-passing*) extrai sinal de poda **além** dos nós independentes (H9a) — via a ablação controlada **`L=0` (MLP baseline) vs `L≥1` (GNN)**, tudo o mais idêntico.

**Architecture:** Reconstrói o grafo do quadtree de cada superbloco (nós 64/32/16, arestas pai↔filho↔irmão) a partir dos pkls `dataset_h9`; nós carregam as **36 features H9a idênticas**. Um GNN expressivo (PyTorch Geometric: SAGE/GAT/GIN) classifica cada nó em 3 classes (N/S/R). O **mesmo modelo com `n_layers=0`** é, por construção, um MLP independente por nó — o baseline. Gate 1-B compara `L=0` vs `L≥1` (SPLIT-recall + redução de custo no oráculo a SPLIT-lost casado). Stage 2 (implantável causal) é condicional e fica fora deste plano.

**Tech Stack:** Python 3 (numpy, torch/GPU), pytest; scripts em `src/scripts/partition_model/`. Reusa `data`, `features`, `regret`, `student`, `partition_defs`, `simulate_pruning`.

## Global Constraints

- **Execução sempre em Docker**, container detached `av1_bench` (`docker exec av1_bench …`). Windows só edição.
- **Interpretador Python** = `/workspace/build/venv-ml/bin/python3` (numpy+torch+CUDA; bare `python3` NÃO tem numpy). Referido como `PY`.
- **Rodar pytest:** `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest <args>'` (prefixe `true; ` se aparecer erro `C:/Program...: No such file`).
- **Sem re-extração:** só `results/dataset_h9/*.pkl`.
- **Dependência `torch_geometric`** — instalada no `venv-ml` e **verificada compatível** (torch 2.5.1+cu121, CUDA intacto; SAGEConv/GATConv OK). Dá à estrutura o melhor tiro (camadas expressivas testadas). Fallback: message-passing à mão se algum executor achar o ambiente quebrado. A escolha PyG-vs-manual NÃO é metodológica — a ablação controlada (`L=0` vs `L≥1`) é.
- **Partição de sequências CONGELADA:** treino = `["Beauty","Bosphorus","CityAlley","FlowerFocus","FlowerKids","ReadySetGo","ShakeNDry","SunBath","Twilight","YachtRide"]`; validação = `["HoneyBee","FlowerPan","Lips"]`; teste reservado = `["Jockey","RaceNight","RiverBank"]` (não usado no Stage 1).
- **Fonte única das features** = `features.node_features_h9a` (36). **Rótulo 3-classes** = `student.collapse_label` (NONE=0, SPLIT=1, REST=2). Nunca reimplementar.
- **Máscara de legalidade:** nos níveis de decisão 64/32/16 as 3 classes N/S/R são todas legais → máscara trivial (omitida). 8px é excluído (não é nó de decisão).
- **Comparação controlada:** `L=0` (MLP baseline) e `L≥1` (GNN) partilham features, dados, split, hiperparâmetros e treino — só o número de camadas de grafo (`--n-layers`) muda.
- **Todo código offline** — Stage 1 NÃO toca em C, não exporta header, não roda benchmark real.
- **Commits:** sem menção a Claude/AI/Co-Authored-By. **Ao fim de cada tarefa: `git commit` + `git push`** (branch `ml-partition-dev`).
- Docs/comentários em PT-BR; jargão técnico em inglês quando consagrado.

---

## File Structure

**Criar** (em `src/scripts/partition_model/`):
- `graph_data.py` — arestas do quadtree a partir das chaves de nó; monta o dataset de grafos (features 36 + rótulo colapsado + nível + arestas).
- `gnn_model.py` — encoder + camadas PyG (SAGE/GAT/GIN, `L` camadas + `--layer`) + head 3-classes; `L=0` (`n_layers=0`) = MLP independente.
- `train_gnn.py` — treina `L=0` e `L≥1` com hiperparâmetros idênticos; salva bundles.
- `gate1_gnn.py` — ablação `L=0` vs `L≥1` (macro-F1, SPLIT-recall por tamanho) na validação.
- `tests/test_graph.py` — testes de `graph_data` e `gnn_model`.

**Modificar:**
- `simulate_pruning.py` — modo `--gnn-bundle` (`score_with_gnn`): monta o grafo de `sbs["nodes"]`, roda o GNN, anexa `prob=softmax(logits)` por nó; reusa `simulate`/`metrics`/`report`.

**Reuso (não reimplementar):** `data.iter_superblock_members`, `features.node_features_h9a`, `student.collapse_label`, `regret._children`, `partition_defs.MODEL_LEVELS`, harness do `simulate_pruning`.

**Nota:** o diretório `tests/` e `tests/conftest.py` já existem (Solução 4) — reusar.

---

## Task 0: Verificar `torch_geometric` no `venv-ml`

**Files:** nenhum (verificação de ambiente).

**Interfaces:**
- Produces: garantia de que PyG está instalado e compatível; caso contrário, sinal para o fallback manual.

- [ ] **Step 1: Verificar (idempotente — já instalado pelo controller)**

Run:
```bash
MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'PY=/workspace/build/venv-ml/bin/python3; $PY -m pip install --quiet torch_geometric; $PY -c "import torch;from torch_geometric.nn import SAGEConv,GATConv,GINConv;import torch as t;x=t.randn(5,16);ei=t.tensor([[0,1],[1,0]]);print(\"torch\",torch.__version__,torch.cuda.is_available());print(\"sage\",tuple(SAGEConv(16,16)(x,ei).shape));print(\"PyG OK\")"'
```
Expected: `torch 2.5.1+cu121 True`, `sage (5, 16)`, `PyG OK`. **Se falhar** (ambiente quebrado): PARAR e escalar ao controller — usar o fallback de message-passing à mão (spec §4-3) em `gnn_model.py`, mantendo a MESMA ablação `L=0` vs `L≥1`.

- [ ] **Step 2: Sem commit** (nada versionado nesta tarefa; é gate de ambiente).

---

## Task 1: `graph_data.py` — arestas do quadtree + dataset de grafos

**Files:**
- Create: `src/scripts/partition_model/graph_data.py`
- Test: `src/scripts/partition_model/tests/test_graph.py`

**Interfaces:**
- Consumes: `regret._children`, `data.iter_superblock_members`, `features.node_features_h9a`, `student.collapse_label`, `partition_defs`.
- Produces:
  - `sb_edges(node_keys) -> list[tuple[int,int]]` — arestas **não-direcionadas** (ambos os sentidos) entre ÍNDICES na lista `node_keys` (cada key = `(dim,r,c)`): pai↔filho (`pai(dim,r,c)=(2*dim, r//2, c//2)`) e irmão↔irmão (mesmo pai). Só entre keys presentes. Pura, testável.
  - `build_graph_dataset(entries, per_pkl=None, limit=None) -> list[dict]` — um dict por superbloco: `{"x":(n,36)f32, "y":(n,)i64 (colapsado), "level":(n,)i64 (dim), "edge_index":(2,E)i64}`. Só nós de decisão (dim∈{64,32,16}).
  - `collate(graphs) -> dict` — concatena uma lista de grafos em um batch bloco-diagonal: `{"x","y","level","edge_index"}` com índices de aresta deslocados por offset acumulado de nós.

- [ ] **Step 1: Escrever os testes que falham**

`src/scripts/partition_model/tests/test_graph.py`:
```python
import numpy as np
import graph_data


def test_sb_edges_parent_child_and_sibling():
    # Raiz 64 + seus quatro filhos 32.
    keys = [(64, 0, 0), (32, 0, 0), (32, 0, 1), (32, 1, 0), (32, 1, 1)]
    edges = set(graph_data.sb_edges(keys))
    # pai<->filho: índice 0 (raiz) liga a 1,2,3,4 (ambos sentidos).
    for j in (1, 2, 3, 4):
        assert (0, j) in edges and (j, 0) in edges
    # irmão<->irmão entre os quatro 32 (ex.: 1<->2).
    assert (1, 2) in edges and (2, 1) in edges
    # sem auto-aresta.
    assert not any(a == b for a, b in edges)


def test_sb_edges_tolerates_missing_child():
    keys = [(64, 0, 0), (32, 0, 0)]   # só um filho presente
    edges = set(graph_data.sb_edges(keys))
    assert (0, 1) in edges and (1, 0) in edges
    # sem arestas para filhos ausentes (nenhum outro índice existe).
    assert max(max(a, b) for a, b in edges) == 1


def test_build_graph_dataset_smoke():
    import data as datamod
    entries = datamod.discover_pkls("/workspace/results/dataset_h9")
    assert entries
    graphs = graph_data.build_graph_dataset([entries[0]], per_pkl=10)
    assert graphs
    g = graphs[0]
    assert g["x"].shape[1] == 36
    assert g["y"].shape[0] == g["x"].shape[0] == g["level"].shape[0]
    assert set(np.unique(g["y"])).issubset({0, 1, 2})   # N/S/R
    assert g["edge_index"].shape[0] == 2


def test_collate_offsets_edges():
    g1 = {"x": np.zeros((2, 36), np.float32), "y": np.zeros(2, np.int64),
          "level": np.array([64, 32], np.int64),
          "edge_index": np.array([[0, 1], [1, 0]], np.int64)}
    g2 = {"x": np.zeros((3, 36), np.float32), "y": np.zeros(3, np.int64),
          "level": np.array([64, 32, 32], np.int64),
          "edge_index": np.array([[0, 1], [1, 0]], np.int64)}
    b = graph_data.collate([g1, g2])
    assert b["x"].shape[0] == 5
    # As arestas do 2º grafo devem ser deslocadas por +2.
    assert [2, 3] in b["edge_index"].T.tolist()
    assert [3, 2] in b["edge_index"].T.tolist()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py -q'`
Expected: FAIL (`No module named 'graph_data'`).

- [ ] **Step 3: Implementar `graph_data.py`**

`src/scripts/partition_model/graph_data.py`:
```python
#!/usr/bin/env python3
"""Approach B (Stage 1): monta o grafo do quadtree de cada superbloco a partir dos
pkls dataset_h9. Nós = nós de decisão (64/32/16) com as 36 features H9a; arestas =
pai<->filho e irmão<->irmão (não-causal, para o diagnóstico). Rótulo 3-classes
(NONE/SPLIT/REST). Ver docs/superpowers/specs/2026-07-17-approachB-gnn-estrutural-design.md."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from regret import _children  # noqa: E402

DECISION_DIMS = (64, 32, 16)


def sb_edges(node_keys):
    """Arestas não-direcionadas (ambos os sentidos) entre índices de node_keys:
    pai<->filho e irmão<->irmão. Só entre keys presentes."""
    idx = {k: i for i, k in enumerate(node_keys)}
    edges = set()
    for k in node_keys:
        dim, r, c = k
        for ch in _children(dim, r, c):
            if ch in idx:
                edges.add((idx[k], idx[ch]))
                edges.add((idx[ch], idx[k]))
        # irmãos: filhos do mesmo pai.
        if dim < 64:
            parent = (2 * dim, r // 2, c // 2)
            for sib in _children(*parent):
                if sib != k and sib in idx:
                    edges.add((idx[k], idx[sib]))
    return sorted(edges)


def build_graph_dataset(entries, per_pkl=None, limit=None):
    """Lista de grafos por superbloco: {x:(n,36), y:(n,), level:(n,), edge_index:(2,E)}."""
    graphs = []
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD (dataset_h9).".format(e["path"]))
            keys, feats, ys, levels = [], [], [], []
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in DECISION_DIMS:
                    continue
                keys.append((dim, r, c))
                feats.append(featmod.node_features_h9a(
                    sb["luma"], dim, r, c, sb["qindex"], sb["ctx"][k]))
                ys.append(studentmod.collapse_label(int(label)))
                levels.append(dim)
            if not keys:
                continue
            e_list = sb_edges(keys)
            ei = (np.asarray(e_list, dtype=np.int64).T
                  if e_list else np.zeros((2, 0), np.int64))
            graphs.append({"x": np.asarray(feats, np.float32),
                           "y": np.asarray(ys, np.int64),
                           "level": np.asarray(levels, np.int64),
                           "edge_index": ei})
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and len(graphs) >= limit:
                return graphs
    return graphs


def collate(graphs):
    """Concatena grafos em um batch bloco-diagonal (MP não cruza superblocos)."""
    xs, ys, lv, eis, off = [], [], [], [], 0
    for g in graphs:
        xs.append(g["x"]); ys.append(g["y"]); lv.append(g["level"])
        if g["edge_index"].shape[1] > 0:
            eis.append(g["edge_index"] + off)
        off += g["x"].shape[0]
    return {"x": np.concatenate(xs, 0),
            "y": np.concatenate(ys, 0),
            "level": np.concatenate(lv, 0),
            "edge_index": (np.concatenate(eis, 1) if eis
                           else np.zeros((2, 0), np.int64))}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py -q'`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/partition_model/graph_data.py src/scripts/partition_model/tests/test_graph.py
git commit -m "approachB: graph_data.py (arestas do quadtree + dataset de grafos) + testes"
git push
```

---

## Task 2: `gnn_model.py` — TreeGNN com PyG (`n_layers=0` == MLP independente)

**Files:**
- Create: `src/scripts/partition_model/gnn_model.py`
- Test: adicionar testes em `tests/test_graph.py`

**Interfaces:**
- Consumes: `torch`, `torch_geometric.nn` (SAGEConv/GATConv/GINConv).
- Produces:
  - `class TreeGNN(nn.Module)` com `__init__(in_features=36, hidden=64, layer="sage", n_layers=2, num_classes=3)` e `forward(x, edge_index) -> logits (N,3)`. Com `n_layers=0`, NÃO há camada de grafo → logits dependem só de cada nó → **MLP independente** (não consome arestas).
  - `build_gnn(in_features=36, hidden=64, layer="sage", n_layers=2) -> TreeGNN`.

- [ ] **Step 1: Escrever os testes que falham**

Adicionar em `tests/test_graph.py`:
```python
def test_gnn_l0_is_independent_of_edges():
    import torch
    import gnn_model
    torch.manual_seed(0)
    net = gnn_model.build_gnn(36, 16, layer="sage", n_layers=0).eval()
    x = torch.randn(5, 36)
    ei_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ei_b = torch.zeros((2, 0), dtype=torch.long)   # sem arestas
    with torch.no_grad():
        ya = net(x, ei_a)
        yb = net(x, ei_b)
    # Com n_layers=0 a saída NÃO pode depender das arestas (baseline MLP).
    assert torch.allclose(ya, yb, atol=1e-6)
    assert ya.shape == (5, 3)


def test_gnn_l1_uses_edges():
    import torch
    import gnn_model
    torch.manual_seed(0)
    net = gnn_model.build_gnn(36, 16, layer="sage", n_layers=1).eval()
    x = torch.randn(5, 36)
    ei_a = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    ei_b = torch.zeros((2, 0), dtype=torch.long)
    with torch.no_grad():
        ya = net(x, ei_a)
        yb = net(x, ei_b)
    # Com n_layers>=1 e arestas presentes, a saída MUDA vs sem arestas.
    assert not torch.allclose(ya, yb, atol=1e-4)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py -k gnn -q'`
Expected: FAIL (`No module named 'gnn_model'`).

- [ ] **Step 3: Implementar `gnn_model.py`**

`src/scripts/partition_model/gnn_model.py`:
```python
#!/usr/bin/env python3
"""Approach B (Stage 1): GNN sobre o quadtree do superbloco, com camadas expressivas
do PyTorch Geometric (SAGE/GAT/GIN). Com n_layers=0 NÃO há camada de grafo -> por
construção um MLP independente por nó (o baseline da ablação estrutural). Adota-se
PyG para dar à estrutura o melhor tiro; a escolha da lib não é metodológica -- a
ablação n_layers=0 vs n_layers>=1 é."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GINConv, SAGEConv


def _make_conv(layer, hidden):
    if layer == "sage":
        return SAGEConv(hidden, hidden)
    if layer == "gat":
        return GATConv(hidden, hidden)
    if layer == "gin":
        return GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden)))
    raise ValueError("layer desconhecido: {}".format(layer))


class TreeGNN(nn.Module):
    def __init__(self, in_features=36, hidden=64, layer="sage", n_layers=2,
                 num_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
        self.convs = nn.ModuleList([_make_conv(layer, hidden)
                                    for _ in range(n_layers)])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x)
        for conv in self.convs:                 # vazio se n_layers=0 -> MLP puro
            h = torch.relu(conv(h, edge_index))
        return self.head(h)


def build_gnn(in_features=36, hidden=64, layer="sage", n_layers=2):
    return TreeGNN(in_features, hidden, layer, n_layers)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py -q'`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit + push**

```bash
git add src/scripts/partition_model/gnn_model.py src/scripts/partition_model/tests/test_graph.py
git commit -m "approachB: gnn_model.py (TreeGNN via PyG SAGE/GAT/GIN, n_layers=0==MLP) + testes"
git push
```

---

## Task 3: `train_gnn.py` — treina `L=0` e `L≥1` (idênticos exceto `--n-layers`)

**Files:**
- Create: `src/scripts/partition_model/train_gnn.py`
- Test: adicionar `test_gnn_train_shapes` em `tests/test_graph.py`

**Interfaces:**
- Consumes: `graph_data.build_graph_dataset/collate`, `gnn_model.build_gnn`, `data.discover_pkls/split_entries/assert_real_luma`, `torch`.
- Produces: bundle salvo em `results/models/gnn_L{n_layers}/gnn.pt` = `{"hidden":H, "layer":str, "n_layers":int, "state_dict":..., "num_features":36, "feature_set":"h9a", "head":"gnn3"}`. Função `train_gnn(graphs, hidden, layer, n_layers, device, epochs, lr, batch_sbs) -> (state_dict, final_loss)`.

- [ ] **Step 1: Escrever o teste de forma (sintético, rápido)**

Adicionar em `tests/test_graph.py`:
```python
def test_gnn_train_shapes():
    import numpy as np
    import train_gnn as tg
    rng = np.random.default_rng(0)
    # 8 grafos sintéticos de 5 nós (raiz 64 + quatro 32), rótulos aleatórios.
    graphs = []
    for _ in range(8):
        x = rng.standard_normal((5, 36)).astype("float32")
        y = rng.integers(0, 3, size=5).astype("int64")
        level = np.array([64, 32, 32, 32, 32], np.int64)
        ei = np.array([[0, 1, 2, 3, 4], [1, 0, 0, 0, 0]], np.int64)
        graphs.append({"x": x, "y": y, "level": level, "edge_index": ei})
    import torch
    sd, loss = tg.train_gnn(graphs, hidden=16, layer="sage", n_layers=1,
                            device=torch.device("cpu"), epochs=3, lr=1e-3,
                            batch_sbs=4)
    assert np.isfinite(loss)
    assert isinstance(sd, dict)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py::test_gnn_train_shapes -q'`
Expected: FAIL (`No module named 'train_gnn'`).

- [ ] **Step 3: Implementar `train_gnn.py`**

`src/scripts/partition_model/train_gnn.py`:
```python
#!/usr/bin/env python3
"""Approach B (Stage 1): treina o TreeGNN. Roda-se duas vezes com hiperparâmetros
IDÊNTICOS, mudando só n_layers (0 = baseline MLP independente, >=1 = com estrutura),
para a ablação controlada do Gate 1-B."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import data as datamod  # noqa: E402
import graph_data as gd  # noqa: E402
import gnn_model as gm  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def train_gnn(graphs, hidden, layer, n_layers, device, epochs, lr, batch_sbs):
    net = gm.build_gnn(36, hidden, layer, n_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    last = float("nan")
    for _ in range(epochs):
        order = np.random.permutation(len(graphs))
        for s in range(0, len(graphs), batch_sbs):
            batch = [graphs[i] for i in order[s:s + batch_sbs]]
            b = gd.collate(batch)
            x = torch.tensor(b["x"], device=device)
            y = torch.tensor(b["y"], device=device)
            ei = torch.tensor(b["edge_index"], device=device)
            opt.zero_grad()
            loss = loss_fn(net(x, ei), y)
            loss.backward()
            opt.step()
            last = float(loss.item())
    return net.state_dict(), last


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--n-layers", type=int, required=True,
                   help="0 = baseline MLP independente; >=1 = GNN com estrutura")
    p.add_argument("--layer", default="sage", choices=["sage", "gat", "gin"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-sbs", type=int, default=256)
    p.add_argument("--per-pkl", type=int, default=2000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, VAL_SEQS, TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    graphs = gd.build_graph_dataset(train_e, per_pkl=args.per_pkl or None)
    print("grafos de treino:", len(graphs), flush=True)
    sd, loss = train_gnn(graphs, args.hidden, args.layer, args.n_layers, device,
                         args.epochs, args.lr, args.batch_sbs)
    out_dir = args.out_dir or "/workspace/results/models/gnn_L{}".format(args.n_layers)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"hidden": args.hidden, "layer": args.layer,
                "n_layers": args.n_layers, "state_dict": sd, "num_features": 36,
                "feature_set": "h9a", "head": "gnn3"},
               os.path.join(out_dir, "gnn.pt"))
    print("[layer={} L={}] loss final={:.5f} -> {}".format(
        args.layer, args.n_layers, loss, os.path.join(out_dir, "gnn.pt")),
        flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Rodar o teste de forma (passar)**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 -m pytest tests/test_graph.py::test_gnn_train_shapes -q'`
Expected: PASS.

- [ ] **Step 5: Commit + push (código; o treino pesado é do controller)**

```bash
git add src/scripts/partition_model/train_gnn.py src/scripts/partition_model/tests/test_graph.py
git commit -m "approachB: train_gnn.py (treino do TreeGNN, ablacao por --n-layers) + smoke"
git push
```

**Nota ao controller:** após o merge desta tarefa, treinar os DOIS modelos com
hiperparâmetros idênticos (mesmo `--hidden/--epochs/--lr/--per-pkl/--batch-sbs`),
mudando só a estrutura, e guardar os logs:
`train_gnn.py --n-layers 0 --out-dir .../gnn_L0` (baseline MLP) e
`train_gnn.py --n-layers 2 --layer sage --out-dir .../gnn_L2` (GNN forte).
Commitar os dois `gnn.pt` (force-add; `results/models` é gitignored). Em caso de
nulo no Gate 1-B, varrer `--layer {sage,gat,gin}` e `--n-layers {1,2,3}` (Stage 1b).

---

## Task 4: `simulate_pruning.py` — modo `--gnn-bundle` (score conjunto)

**Files:**
- Modify: `src/scripts/partition_model/simulate_pruning.py`

**Interfaces:**
- Consumes: bundle `gnn.pt` (`head=="gnn3"`), `gnn_model.build_gnn`, `graph_data.sb_edges`.
- Produces: `score_with_gnn(sbs, bundle, device)` que, para cada superbloco, monta o grafo de `sbs[si]["nodes"]` (chaves `(dim,r,c)`), roda o GNN e anexa `nd["prob"] = softmax(logits)` (`[P(NONE),P(SPLIT),P(REST)]`). Arg `--gnn-bundle`; precedência de seleção como `--regret-bundle`.

- [ ] **Step 1: Ler os pontos de integração**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'true; grep -n "score_with_regret\|regret_bundle\|feature_set, mode\|def main" /workspace/src/scripts/partition_model/simulate_pruning.py'`
Objetivo: espelhar o ramo `--regret-bundle` (já existente) para o `--gnn-bundle`.

- [ ] **Step 2: Adicionar `score_with_gnn` e o wiring**

Adicionar (junto de `score_with_regret`):
```python
def score_with_gnn(sbs, bundle, device):
    """Approach B: probs conjuntas por nó de um TreeGNN. Monta o grafo de cada
    superbloco a partir das chaves de nós e anexa prob=softmax(logits)."""
    import torch
    import torch.nn.functional as F
    import gnn_model
    import graph_data
    net = gnn_model.build_gnn(bundle["num_features"], bundle["hidden"],
                              bundle["layer"], bundle["n_layers"])
    net.load_state_dict(bundle["state_dict"])
    net = net.to(device).eval()
    modeled = {d for d, _ in MODEL_LEVELS}
    for sb in sbs:
        keys = [k for k in sb["nodes"] if k[0] in modeled]
        if not keys:
            continue
        feats = np.stack([sb["nodes"][k]["feat"] for k in keys])
        e_list = graph_data.sb_edges(keys)
        ei = (torch.tensor(np.asarray(e_list).T, dtype=torch.long, device=device)
              if e_list else torch.zeros((2, 0), dtype=torch.long, device=device))
        with torch.no_grad():
            logits = net(torch.tensor(feats, dtype=torch.float32, device=device), ei)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        for k, p in zip(keys, probs):
            sb["nodes"][k]["prob"] = p
    return sbs
```
Wiring no `main`: adicionar `--gnn-bundle` (default None); quando setado, carregar o bundle, `assert bundle.get("head")=="gnn3"`, `feature_set = bundle.get("feature_set","h9a")`, `mode="gnn"`; no despacho de scoring, `elif mode=="gnn": sbs = score_with_gnn(sbs, bundle, device); model_tag = "GNN {} layer={} L={}".format(args.gnn_bundle, bundle["layer"], bundle["n_layers"])`. Espelhar exatamente a estrutura if/elif do `--regret-bundle` (precedência mutuamente exclusiva); não quebrar os caminhos existentes.

- [ ] **Step 3: Smoke (não é o gate; o gate é do controller)**

Run (após o controller ter treinado um `gnn_L2/gnn.pt`; se ainda não existir, PULAR este step e deixar o controller validar no Gate):
`MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 simulate_pruning.py --dataset-dir /workspace/results/dataset_h9 --gnn-bundle /workspace/results/models/gnn_L2/gnn.pt --val-seqs HoneyBee --tau-none 0.9 0.99 --limit 150 --out-csv /workspace/results/models/gnn_L2/smoke.csv'`
Expected: imprime a tabela de métricas sem traceback. Não commitar o smoke.csv.

- [ ] **Step 4: Commit + push**

```bash
git add src/scripts/partition_model/simulate_pruning.py
git commit -m "approachB: modo --gnn-bundle no simulador (score conjunto do TreeGNN)"
git push
```

---

## Task 5: `gate1_gnn.py` — ablação `L=0` vs `L≥1` por nó (macro-F1, SPLIT-recall)

**Files:**
- Create: `src/scripts/partition_model/gate1_gnn.py`

**Interfaces:**
- Consumes: `graph_data.build_graph_dataset/collate`, `gnn_model.build_gnn`, `data.discover_pkls/split_entries`, `train_gnn.VAL_SEQS`, `torch`, `sklearn`? (NÃO — evitar dependência; calcular macro-F1 e recall à mão com numpy).
- Produces: relatório impresso + `results/models/gnn_gate1.csv` com, por bundle (k0/k2) e por tamanho (64/32/16): `n, acc, macroF1, split_recall`.

- [ ] **Step 1: Implementar `gate1_gnn.py`**

`src/scripts/partition_model/gate1_gnn.py`:
```python
#!/usr/bin/env python3
"""Gate 1-B (Approach B): ablação controlada L=0 (MLP) vs L>=1 (GNN) na validação.
Para cada bundle, prediz as classes por nó no conjunto de validação e reporta, por
tamanho, acurácia, macro-F1 e SPLIT-recall. Se L>=1 não superar L=0 (nem no oráculo,
ver simulate_pruning --gnn-bundle), a estrutura não agrega -> teto informacional."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data as datamod  # noqa: E402
import graph_data as gd  # noqa: E402
import gnn_model as gm  # noqa: E402

VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def _macro_f1_and_split_recall(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    # SPLIT é a classe 1 (student.collapse_label).
    tp1 = np.sum((y_pred == 1) & (y_true == 1))
    fn1 = np.sum((y_pred != 1) & (y_true == 1))
    split_recall = tp1 / (tp1 + fn1) if (tp1 + fn1) else 0.0
    return float(np.mean(f1s)), float(split_recall)


def _predict(bundle, graphs, device):
    net = gm.build_gnn(bundle["num_features"], bundle["hidden"],
                       bundle["layer"], bundle["n_layers"])
    net.load_state_dict(bundle["state_dict"])
    net = net.to(device).eval()
    yt, yp, lv = [], [], []
    with torch.no_grad():
        for g in graphs:
            x = torch.tensor(g["x"], device=device)
            ei = torch.tensor(g["edge_index"], device=device)
            pred = F.softmax(net(x, ei), -1).argmax(-1).cpu().numpy()
            yt.append(g["y"]); yp.append(pred); lv.append(g["level"])
    return (np.concatenate(yt), np.concatenate(yp), np.concatenate(lv))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--bundles", nargs="+", required=True,
                   help="pares tag=caminho, ex.: k0=.../gnn_k0/gnn.pt k2=.../gnn_k2/gnn.pt")
    p.add_argument("--per-pkl", type=int, default=2000)
    p.add_argument("--out", default="/workspace/results/models/gnn_gate1.csv")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, val_e = datamod.split_entries(entries, VAL_SEQS, None)
    datamod.assert_real_luma(val_e)
    graphs = gd.build_graph_dataset(val_e, per_pkl=args.per_pkl or None)
    print("grafos de validação:", len(graphs), flush=True)

    rows = ["tag,dim,n,acc,macroF1,split_recall"]
    for spec in args.bundles:
        tag, path = spec.split("=", 1)
        bundle = torch.load(path, map_location=device)
        assert bundle.get("head") == "gnn3"
        yt, yp, lv = _predict(bundle, graphs, device)
        for dim in (64, 32, 16):
            m = lv == dim
            if not m.any():
                continue
            acc = float((yp[m] == yt[m]).mean())
            f1, sr = _macro_f1_and_split_recall(yt[m], yp[m])
            rows.append("{},{},{},{:.4f},{:.4f},{:.4f}".format(
                tag, dim, int(m.sum()), acc, f1, sr))
            print("[{} dim {:>2}] n={} acc={:.3f} macroF1={:.3f} "
                  "split_recall={:.3f}".format(tag, dim, int(m.sum()), acc, f1, sr),
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Gate 1-B ->", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 2: Confirmar import + `--help` (o gate completo é do controller)**

Run: `MSYS_NO_PATHCONV=1 docker exec av1_bench bash -lc 'cd /workspace/src/scripts/partition_model && /workspace/build/venv-ml/bin/python3 gate1_gnn.py --help && /workspace/build/venv-ml/bin/python3 -c "import gate1_gnn; print(\"ok\")"'`
Expected: help + `ok`, sem erro.

- [ ] **Step 3: Commit + push**

```bash
git add src/scripts/partition_model/gate1_gnn.py
git commit -m "approachB: gate1_gnn.py (ablacao L=0 vs L>=1 por no: macro-F1, SPLIT-recall)"
git push
```

**Nota ao controller (Gate 1-B — milestone):** com os dois bundles treinados,
rodar (1) `gate1_gnn.py --bundles L0=.../gnn_L0/gnn.pt L2=.../gnn_L2/gnn.pt` para
macro-F1/SPLIT-recall; e (2) o oráculo a SPLIT-lost casado para AMBOS via
`simulate_pruning.py --gnn-bundle .../gnn_L0/gnn.pt` e `--gnn-bundle .../gnn_L2/gnn.pt`
(mesmo grid de τ), comparando a redução de custo. **PASSA** se `L≥1` supera `L=0`
por margem relevante e consistente em SPLIT-recall E no oráculo; senão, antes de
declarar teto, **Stage 1b (varredura de robustez):** treinar `--layer {sage,gat,gin}`
× `--n-layers {1,2,3}` e re-rodar o gate — só um nulo consistente na melhor variante
confirma **teto informacional**. Adjudicar e registrar; decidir com o usuário sobre Stage 2.

---

## Task 6: Documentação do resultado

**Files:**
- Create: `docs/RESULTADOS_approachB.md`
- Modify: `docs/SINTESE_resultados_metodologia.md`, `docs/ANDAMENTO_tese.md`

- [ ] **Step 1: Escrever `docs/RESULTADOS_approachB.md`**

Documentar: pergunta (teto informacional vs artefato de independência), método (grafo do quadtree, TreeGNN via PyG, ablação `L=0` vs `L≥1` + varredura de robustez), Gate 1-B (macro-F1/SPLIT-recall + oráculo), o resultado obtido e a leitura (se FALHA: teto informacional confirmado; se PASSA: abre Stage 2). Padrão de `docs/RESULTADOS_solucao4.md`.

- [ ] **Step 2: Atualizar SINTESE (§5-ter Approach B) e ANDAMENTO**

Adicionar seção/entrada com o veredito do Gate 1-B e o que implica para a Conclusão 3.

- [ ] **Step 3: Commit + push**

```bash
git add docs/RESULTADOS_approachB.md docs/SINTESE_resultados_metodologia.md docs/ANDAMENTO_tese.md
git commit -m "approachB: documentar resultado do Gate 1-B (teste estrutural do teto)"
git push
```

---

## Notas de execução / ameaças à validade

- **Comparação justa:** `L=0` e `L≥1` DEVEM usar os mesmos `--hidden/--epochs/--lr/--per-pkl/--batch-sbs` e o mesmo split. Qualquer divergência invalida a atribuição à estrutura. O controller usa a MESMA linha de comando trocando só `--n-layers` (e `--layer` na varredura de robustez).
- **Não subdimensionar (risco metodológico real):** dar à estrutura o melhor tiro (PyG SAGE/GAT/GIN); um nulo só vira "teto informacional" após a varredura Stage 1b. A escolha PyG-vs-manual NÃO é metodológica.
- **Oráculo superestima o tempo (~5×):** aceitável no Stage 1 porque a comparação é RELATIVA (`L≥1` vs `L=0` sob o MESMO oráculo) — o viés cancela.
- **`L=0` compartilhado vs H9a por-tamanho:** o baseline primário é o gêmeo `L=0` (mesma arquitetura). O cross-check externo vs `student_h9a` é secundário (validade externa), não a atribuição.
- **Carga de dados:** `build_graph_dataset` sobre treino/val carrega os pkls (lento) → o controller roda treino e Gate 1-B em background, como nas fases anteriores.
- **Stage 2 é condicional** e fora deste plano: só se o Gate 1-B passar, com spec/plano próprios.
```
