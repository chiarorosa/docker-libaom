#!/usr/bin/env python3
"""Figura 1 do artigo ISCAS 2027 (NPL-AV1) — quem decide em cada ponto da
busca de um no, codificador nativo contra NPL-AV1, em codificacao All-Intra.

Tres colunas:
  esquerda — podadores APRENDIDOS nativos do libaom v3.10.0, no ponto em que
             agem; traco tracejado e texto esmaecido quando desligados em AI.
  centro   — a ordem fixa de avaliacao dos tipos de particao no no.
  direita  — os podadores do NPL-AV1, no ponto em que sao consultados.

Mesma infraestrutura do fluxograma do LASCAS (plot_pruner_flowchart_fig2.py):
largura de coluna do IEEEtran, texto em 8 pt, altura derivada das linhas,
PALETAS importadas do modulo do LASCAS para que as duas figuras nao divirjam.
Nao le artefato numerico: e um diagrama estrutural.

Fatos desenhados, conferidos em src/aom/av1/encoder (2026-10-03):
  ordem: av1_prune_partitions_before_search (5759) -> NONE (5829) ->
         av1_prune_after_none (5847) -> SPLIT (5857) -> RECT (5889) ->
         AB (5917) -> 4-way (5943)                      [partition_search.c]
  CNN intra: av1_intra_mode_cnn_partition, chamada antes da busca
         (partition_strategy.c:2435); ativa em AI a partir de cpu-used=1
         (speed_features.c, intra_cnn_based_part_prune_level=2).
  desligados em quadros intra (!frame_is_intra_only):
         av1_ml_predict_breakout (partition_search.c:4296),
         av1_ml_early_term_after_split (:4357), av1_ml_prune_rect_partition (:4373).
  ativos em AI (ml_prune_partition=1 em todos os presets, speed_features.c:339):
         av1_ml_prune_ab_partition (partition_strategy.c:2662),
         av1_ml_prune_4_partition (partition_search.c:4239).

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_npl_ladder_iscas_fig1.py

Saidas: figura1_escada_npl.pdf / _cinza.pdf (+ .png para conferencia)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pruner_insertion_fig2 import PALETAS  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_W_IN = 252.0 / 72.27   # largura de coluna do IEEEtran [conference]
PT = 8.0

# --- ritmo vertical (alturas em polegadas; a altura total e derivada) --------
LINHAS = [
    ("pad", 0.030),
    ("hdr", 0.150),
    ("gap", 0.040),
    ("p0",  0.440),   # ponto pre-busca
    ("gap", 0.045),
    ("b1",  0.150),   # NONE
    ("gap", 0.045),
    ("p1",  0.440),   # ponto pos-NONE
    ("gap", 0.045),
    ("b2",  0.150),   # SPLIT
    ("gap", 0.045),
    ("p2",  0.320),   # depois do SPLIT, antes das retangulares
    ("gap", 0.045),
    ("b3",  0.150),   # HORZ, VERT
    ("gap", 0.045),
    ("p3",  0.320),   # antes das estendidas
    ("gap", 0.045),
    ("b4",  0.150),   # AB, 4-way
    ("pad", 0.030),
]
FIG_H_IN = sum(h for _, h in LINHAS)

Y = {}
_c = 1.0
for _n, _h in LINHAS:
    _hn = _h / FIG_H_IN
    if _n not in ("pad", "gap"):
        Y[_n] = (_c - _hn / 2.0, _hn)
    _c -= _hn

# --- grade horizontal ----------------------------------------------------------
X_SPINE = 0.500
MEIA_CENTRO = 0.115        # meia-largura das caixas de avaliacao
X_ESQ = (0.004, 0.368)     # caixas dos podadores nativos
X_DIR = (0.632, 0.996)     # caixas do NPL-AV1

# --- textos --------------------------------------------------------------------
CENTRO = {"b1": "NONE", "b2": "SPLIT", "b3": "HORZ, VERT", "b4": "AB, 4-way"}

# (texto, ativo em AI?)
NATIVO = {
    "p0": ("CNN (cpu-used ≥ 1):\nforces or\nforbids SPLIT", True),
    "p1": ("breakout after NONE:\noff in AI", False),
    "p2": ("early termination and\nrect. pruning: off in AI", False),
    "p3": ("AB and 4-way\nnetworks: active in AI", True),
}
NPL = {
    "p0": "pre-search MLP:\nNONE only, SPLIT only,\nor no rect. types",
    "p1": "post-NONE MLPs:\nstop the node, or\nskip AB and 4-way",
}


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def caixa(x0, x1, yc, h, texto, borda, fundo, cor_txt, tracejada=False,
              peso="normal"):
        ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h,
                               facecolor=fundo, edgecolor=borda,
                               linewidth=0.8, zorder=4,
                               linestyle=(0, (2.2, 1.6)) if tracejada else "-"))
        ax.text((x0 + x1) / 2, yc, texto, ha="center", va="center",
                fontsize=PT, color=cor_txt, linespacing=1.15, zorder=5,
                weight=peso)

    def linha(x0, x1, y, tracejada=False, cor=None):
        ax.plot([x0, x1], [y, y], color=cor or p["ink_2"], linewidth=0.8,
                zorder=3, linestyle=(0, (2.2, 1.6)) if tracejada else "-")

    # cabecalho
    yh, _ = Y["hdr"]
    for x, t in (((X_ESQ[0] + X_ESQ[1]) / 2, "native (All-Intra)"),
                 (X_SPINE, "evaluation order"),
                 ((X_DIR[0] + X_DIR[1]) / 2, "NPL-AV1")):
        ax.text(x, yh, t, ha="center", va="center", fontsize=PT,
                color=p["ink"], style="italic")

    # eixo: do topo do primeiro ponto ao topo da ultima caixa
    y_top = Y["p0"][0]
    y_bot = Y["b4"][0] + Y["b4"][1] / 2
    ax.annotate("", xy=(X_SPINE, y_bot), xytext=(X_SPINE, y_top),
                arrowprops=dict(arrowstyle="-|>", color=p["ink_2"],
                                linewidth=0.8, shrinkA=0, shrinkB=0,
                                mutation_scale=7.0), zorder=2)

    # caixas de avaliacao, sobre o eixo
    for k, t in CENTRO.items():
        yc, h = Y[k]
        caixa(X_SPINE - MEIA_CENTRO, X_SPINE + MEIA_CENTRO, yc, h, t,
              p["ink_2"], p["surface"], p["ink"])

    # pontos de decisao
    for k in ("p0", "p1", "p2", "p3"):
        yc, h = Y[k]
        ax.plot([X_SPINE], [yc], marker="o", markersize=3.2,
                color=p["ink"], zorder=6)
        txt, ativo = NATIVO[k]
        caixa(X_ESQ[0], X_ESQ[1], yc, h - 0.02 / FIG_H_IN, txt,
              p["ink_2"] if ativo else p["muted"], p["surface"],
              p["ink"] if ativo else p["muted"], tracejada=not ativo)
        linha(X_ESQ[1], X_SPINE, yc, tracejada=not ativo,
              cor=p["ink_2"] if ativo else p["muted"])
        if k in NPL:
            caixa(X_DIR[0], X_DIR[1], yc, h - 0.02 / FIG_H_IN, NPL[k],
                  p["s1"], p["kept_f"], p["ink"])
            linha(X_SPINE, X_DIR[0], yc, cor=p["s1"])

    fig.savefig(out_path, facecolor=p["surface"])
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=p["surface"], dpi=400)
    plt.close(fig)
    print("  gravado: %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out-dir",
                    default=os.path.join(root, "results", "thesis", "figuras"))
    ap.add_argument("--variante", choices=sorted(PALETAS) + ["ambas"],
                    default="ambas")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sufixo = {"cor": "", "cinza": "_cinza"}
    alvos = sorted(PALETAS) if args.variante == "ambas" else [args.variante]
    for nome in alvos:
        draw(os.path.join(args.out_dir,
                          "figura1_escada_npl%s.pdf" % sufixo[nome]),
             PALETAS[nome])
    print("\nDimensoes finais: %.3f x %.3f in" % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
