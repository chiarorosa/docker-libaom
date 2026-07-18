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


def test_sb_edges_causal_only_from_decided():
    import graph_data
    # raiz 64 + quatro filhos 32 (raster TL=0,TR=1,BL=2,BR=3).
    keys = [(64, 0, 0), (32, 0, 0), (32, 0, 1), (32, 1, 0), (32, 1, 1)]
    E = set(graph_data.sb_edges_causal(keys))
    # todas as arestas sao dirigidas para o no; pai->filho presente:
    for j in (1, 2, 3, 4):
        assert (0, j) in E            # 64 -> cada 32
        assert (j, 0) not in E        # NUNCA filho -> pai (nao-causal)
    # irmao anterior (raster menor) -> posterior; nao o contrario:
    # (32,0,0)=raster0, (32,0,1)=raster1, (32,1,0)=raster2, (32,1,1)=raster3
    assert (1, 2) in E and (2, 1) not in E   # TL->TR sim, TR->TL nao
    assert (1, 4) in E and (4, 1) not in E   # TL->BR sim, BR->TL nao
    assert (3, 2) not in E                   # BL(raster2) nao recebe de (1,0)? check: (3)=(32,1,0) raster2, (2)=(32,0,1) raster1 -> (2,3) in E
    assert (2, 3) in E                       # TR(raster1) -> BL(raster2)
    # raiz nao tem arestas de entrada:
    assert not any(dst == 0 for _src, dst in E)
    # sem auto-aresta:
    assert not any(a == b for a, b in E)


def test_build_graph_dataset_causal_flag_smoke():
    import numpy as np
    import data as datamod, graph_data
    entries = datamod.discover_pkls("/workspace/results/dataset_h9")
    assert entries
    g_nc = graph_data.build_graph_dataset([entries[0]], per_pkl=10, causal=False)[0]
    g_c = graph_data.build_graph_dataset([entries[0]], per_pkl=10, causal=True)[0]
    # mesmas features/rotulos; arestas causais sao um subconjunto dirigido (<= nao-causal).
    assert g_c["x"].shape == g_nc["x"].shape
    assert g_c["edge_index"].shape[1] <= g_nc["edge_index"].shape[1]


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
