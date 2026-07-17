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
