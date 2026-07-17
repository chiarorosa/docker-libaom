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
