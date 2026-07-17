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
    # A soma da subárvore não muda (folha retangular vale seu próprio none_rdcost
    # como limite superior) -> regret_rel da raiz permanece 0.25.
    assert abs(out[(64, 0, 0)]["regret_rel"] - 0.25) < 1e-9
    # O próprio nó retangular colapsa: subtree_rd = seu rd -> regret 0, censurado.
    assert out[(32, 0, 0)]["regret_rel"] == 0.0
    assert out[(32, 0, 0)]["exact"] is False


def test_missing_none_rdcost_node_is_dropped():
    members, ctx = _split_sb(1000, [200, 200, 200, 200])
    ctx[0] = {"none_rdcost": 0}   # raiz sem NONE avaliado (sentinela)
    keys = {(n["dim"], n["r"], n["c"]) for n in regret.node_regrets(members, ctx)}
    assert (64, 0, 0) not in keys


def test_missing_split_child_falls_back_to_none_zero_regret():
    # Raiz SPLIT mas só 3 dos 4 filhos 32 existem (o 4º está ausente).
    # subtree_rd cai no fallback: usa o próprio rd da raiz (upper bound),
    # marcando a subárvore como censurada -> regret 0.
    m0, x0 = _mk(64, 0, 0, 3, 1000)   # 3 == SPLIT
    m1, x1 = _mk(32, 0, 0, 0, 200)    # 0 == NONE
    m2, x2 = _mk(32, 0, 1, 0, 200)
    m3, x3 = _mk(32, 1, 0, 0, 200)
    # (32, 1, 1) ausente de propósito.
    members = [m0, m1, m2, m3]
    ctx = [x0, x1, x2, x3]
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert out[(64, 0, 0)]["regret_rel"] == 0.0
    assert out[(64, 0, 0)]["exact"] is False


def test_nested_split_all_none_regret_matches_hand_calc():
    # Árvore de dois níveis, tudo NONE nas folhas:
    #   raiz (64,0,0) SPLIT, none_rdcost=1500
    #     filho (32,0,0) SPLIT, none_rdcost=600
    #       netos 16x4, cada NONE none_rdcost=100 -> subtree(32,0,0) = 400
    #     filhos (32,0,1),(32,1,0),(32,1,1) NONE, none_rdcost=200 cada
    #   subtree(raiz) = 400 + 200 + 200 + 200 = 1000
    #   regret_rel(raiz) = (1500-1000)/1000 = 0.5 ; exato (sem folha retangular)
    #   regret_rel(32,0,0) = (600-400)/400 = 0.5 ; exato
    root, x_root = _mk(64, 0, 0, 3, 1500)
    child00, x_child00 = _mk(32, 0, 0, 3, 600)
    child01, x_child01 = _mk(32, 0, 1, 0, 200)
    child10, x_child10 = _mk(32, 1, 0, 0, 200)
    child11, x_child11 = _mk(32, 1, 1, 0, 200)
    gc00, x_gc00 = _mk(16, 0, 0, 0, 100)
    gc01, x_gc01 = _mk(16, 0, 1, 0, 100)
    gc10, x_gc10 = _mk(16, 1, 0, 0, 100)
    gc11, x_gc11 = _mk(16, 1, 1, 0, 100)
    members = [root, child00, child01, child10, child11, gc00, gc01, gc10, gc11]
    ctx = [x_root, x_child00, x_child01, x_child10, x_child11,
           x_gc00, x_gc01, x_gc10, x_gc11]
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert abs(out[(64, 0, 0)]["regret_rel"] - 0.5) < 1e-9
    assert out[(64, 0, 0)]["exact"] is True
    assert abs(out[(32, 0, 0)]["regret_rel"] - 0.5) < 1e-9
    assert out[(32, 0, 0)]["exact"] is True


def test_nested_split_deep_rectangular_grandchild_propagates_censorship():
    # Mesma árvore do teste anterior, mas um dos netos 16 sob (32,0,0) vira
    # retangular (HORZ=1). A folha continua valendo seu próprio none_rdcost=100
    # (limite superior), então as somas não mudam -> regret_rel inalterado.
    # Mas a censura deve se propagar por dois níveis (OR ao subir a árvore).
    root, x_root = _mk(64, 0, 0, 3, 1500)
    child00, x_child00 = _mk(32, 0, 0, 3, 600)
    child01, x_child01 = _mk(32, 0, 1, 0, 200)
    child10, x_child10 = _mk(32, 1, 0, 0, 200)
    child11, x_child11 = _mk(32, 1, 1, 0, 200)
    gc00, x_gc00 = _mk(16, 0, 0, 1, 100)   # 1 == HORZ (retangular)
    gc01, x_gc01 = _mk(16, 0, 1, 0, 100)
    gc10, x_gc10 = _mk(16, 1, 0, 0, 100)
    gc11, x_gc11 = _mk(16, 1, 1, 0, 100)
    members = [root, child00, child01, child10, child11, gc00, gc01, gc10, gc11]
    ctx = [x_root, x_child00, x_child01, x_child10, x_child11,
           x_gc00, x_gc01, x_gc10, x_gc11]
    out = {(n["dim"], n["r"], n["c"]): n for n in regret.node_regrets(members, ctx)}
    assert abs(out[(64, 0, 0)]["regret_rel"] - 0.5) < 1e-9
    assert out[(64, 0, 0)]["exact"] is False


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
