#!/usr/bin/env python3
"""Figura 1 do artigo ISCAS 2027 (NPL-AV1) — fluxograma da busca de um no sob
a escada de podadores, em codificacao All-Intra.

Mesma gramatica visual do fluxograma do LASCAS (plot_pruner_flowchart_fig2.py):
losango = decisao, retangulo = acao ou processo, preenchimento = acao que
encerra o no, blocos de entrada com a soma dos atributos, trilhos de
reencontro, texto em 8 pt, largura de coluna do IEEEtran, altura derivada das
linhas, PALETAS importadas do modulo do LASCAS.

O CONTEUDO e proprio do ISCAS e deliberadamente distinto do LASCAS:
  - losango de topo "pruner of the preset": a escolha do podador que substitui a
    CNN nativa (nativo / pre-busca / nenhum, no degrau pos-NONE);
  - a cascata interna do pre-busca fica numa unica caixa de acao (no LASCAS ela
    e o centro da figura);
  - losango de parar o no apos NONE (pos-NONE), e o das estendidas;
  - o losango NATIVO das redes de AB e 4-way, ativas em AI.

Fatos desenhados, conferidos em src/aom/av1/encoder (2026-10-03):
  ordem: av1_prune_partitions_before_search (partition_search.c:5759) -> NONE
         (5829) -> av1_prune_after_none (5847) -> SPLIT (5857) -> RECT (5889)
         -> AB (5917) -> 4-way (5943).
  CNN intra: av1_intra_mode_cnn_partition (partition_strategy.c:2435), nivel 2
         em AI a partir de cpu-used=1: SPLIT forcado (NONE e retangulares
         desligados) ou SPLIT proibido.
  NONE pode nao ser avaliado: SPLIT forcado pela CNN ou pelo pre-busca
         (av1_set_square_split_only) -> caixa "NONE is evaluated, if allowed".
  pos-NONE, parar o no: av1_disable_all_splits (student_h9c_decide).
  pos-NONE, estendidas: h9d_skip_ext (partition_search.c:4113, 4181).
  redes nativas AB/4-way ativas em AI: av1_ml_prune_ab_partition
         (partition_strategy.c:2662), av1_ml_prune_4_partition
         (partition_search.c:4239); ml_prune_partition=1 (speed_features.c:339).

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
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pruner_insertion_fig2 import PALETAS, SPEC  # noqa: E402

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
PT_MATH = 10.0             # condicoes com subscrito, como no LASCAS

# --- ritmo vertical (polegadas; a altura total e derivada) ---------------------
LINHAS = [
    ("pad",   0.030),
    ("d0",    0.300),   # degrau do preset
    ("gap",   0.090),
    ("pre",   0.430),   # CNN (esq.) | MLP pre-busca (dir.)
    ("gap",   0.100),   # reencontro
    ("none",  0.250),
    ("gap",   0.060),
    ("e2",    0.150),   # entrada do pos-NONE
    ("gap",   0.085),
    ("cstop", 0.285),
    ("gap",   0.085),
    ("cext",  0.285),
    ("gap",   0.100),   # reencontro das estendidas
    ("mid",   0.250),
    ("gap",   0.085),
    ("cnat",  0.460),
    ("gap",   0.100),   # reencontro da poda nativa
    ("fim",   0.250),
    ("pad",   0.030),
]
FIG_H_IN = sum(h for _, h in LINHAS)

Y = {}
_c = 1.0
for _n, _h in LINHAS:
    _hn = _h / FIG_H_IN
    if _n not in ("pad", "gap"):
        Y[_n] = (_c - _hn / 2.0, _hn)
    _c -= _hn

# --- grade horizontal ------------------------------------------------------------
X_SPINE = 0.500
MEIA_LOS = 0.220           # meia-largura dos losangos de texto
MEIA_LOS_MAT = 0.180       # losangos de condicao (texto curto)
MEIA_PROC = 0.250          # meia-largura das caixas de processo
X_ESQ = (0.004, 0.290)     # caixa da CNN nativa
X_DIR = (0.710, 0.996)     # caixa do MLP pre-busca
X_ACAO = (0.780, 0.996)    # caixas de consequencia dos ramos "yes"
X_TAG = 0.004              # rotulo do degrau a esquerda dos losangos
X_ENT = (0.535, 0.996)     # blocos de entrada do pos-NONE

N_S1 = sum(n for _, n in SPEC["blocos_s1"])               # 36
N_EXTRA = SPEC["bloco_extra_s2"][1]                        # 3

CONDICOES = {
    "cstop": r"$p_{\mathrm{stop}} > \tau_{\mathrm{stop}}$",
    "cext":  r"$p_{\mathrm{ext}} < \theta_{\mathrm{size}}$",
}


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def seta(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=p["ink_2"],
                                    linewidth=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=7.0), zorder=4)

    def linha(pts):
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=p["ink_2"],
                linewidth=0.8, zorder=3, solid_capstyle="round",
                solid_joinstyle="round")

    def ponto(x, y):
        ax.plot([x], [y], marker="o", markersize=1.8, color=p["ink_2"],
                zorder=5)

    def rotulo(x, y, t, ha="center", italico=False, cor=None):
        ax.text(x, y, t, ha=ha, va="center", fontsize=PT,
                color=cor or p["ink_2"], zorder=5, linespacing=1.1,
                style="italic" if italico else "normal")

    def caixa(x0, x1, yc, h, texto, terminal=False, borda=None):
        ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h,
                               facecolor=p["kept_f"] if terminal else p["surface"],
                               edgecolor=borda or p["ink_2"], linewidth=0.8,
                               zorder=4))
        ax.text((x0 + x1) / 2, yc, texto, ha="center", va="center",
                fontsize=PT, color=p["ink"], linespacing=1.2, zorder=5)

    def losango(chave, texto, pt=PT, meia=MEIA_LOS):
        yc, h = Y[chave]
        ax.add_patch(Polygon([(X_SPINE, yc + h / 2), (X_SPINE + meia, yc),
                              (X_SPINE, yc - h / 2), (X_SPINE - meia, yc)],
                             closed=True, facecolor=p["surface"],
                             edgecolor=p["s1"], linewidth=0.9, zorder=4))
        ax.text(X_SPINE, yc, texto, ha="center", va="center", fontsize=pt,
                color=p["ink"], linespacing=1.1, zorder=5)
        return yc, h

    def processo(chave, texto):
        yc, h = Y[chave]
        caixa(X_SPINE - MEIA_PROC, X_SPINE + MEIA_PROC, yc, h, texto)
        return yc, h

    def nao(yc, h, y_alvo):
        seta(X_SPINE, yc - h / 2, X_SPINE, y_alvo)
        rotulo(X_SPINE - 0.014, yc - h / 2 - 0.34 * (yc - h / 2 - y_alvo),
               "no", ha="right")

    def sim(yc, texto, terminal=False, meia=MEIA_LOS):
        """Ramo "yes" pela direita; devolve a altura da caixa de acao."""
        seta(X_SPINE + meia, yc, X_ACAO[0], yc)
        rotulo((X_SPINE + meia + X_ACAO[0]) / 2, yc + 0.020, "yes")
        h = (0.300 if "\n" in texto else 0.190) / FIG_H_IN
        caixa(X_ACAO[0], X_ACAO[1], yc, h, texto, terminal=terminal)
        return h

    def reencontro(yc_acao, h_acao, y_merge):
        """Da base da caixa de acao desce ate a folga e volta ao eixo."""
        xm = (X_ACAO[0] + X_ACAO[1]) / 2
        linha([(xm, yc_acao - h_acao / 2), (xm, y_merge), (X_SPINE, y_merge)])
        ponto(X_SPINE, y_merge)

    def tag(yc, t):
        rotulo(X_TAG, yc, t, ha="left", italico=True, cor=p["muted"])

    # ---------------- degrau do preset --------------------------------------
    yd, hd = losango("d0", "pruner of\nthe preset")
    ypre, hpre = Y["pre"]
    xe = (X_ESQ[0] + X_ESQ[1]) / 2
    xd = (X_DIR[0] + X_DIR[1]) / 2
    # nativo: sai pelo vertice esquerdo
    linha([(X_SPINE - MEIA_LOS, yd), (xe, yd)])
    seta(xe, yd, xe, ypre + hpre / 2)
    rotulo((X_SPINE - MEIA_LOS + xe) / 2 + 0.010, yd + 0.020, "native")
    caixa(X_ESQ[0], X_ESQ[1], ypre, hpre,
          "CNN:\nforces or\nforbids SPLIT")
    # pre-busca: sai pelo vertice direito
    linha([(X_SPINE + MEIA_LOS, yd), (xd, yd)])
    seta(xd, yd, xd, ypre + hpre / 2)
    rotulo((X_SPINE + MEIA_LOS + xd) / 2 - 0.010, yd + 0.020, "pre-search")
    caixa(X_DIR[0], X_DIR[1], ypre, hpre,
          "MLP (%d): NONE\nonly, SPLIT only or\nno rectangular types" % N_S1,
          borda=p["s1"])
    # pos-NONE: segue pelo eixo, sem podador antes da busca
    ynone, hnone = Y["none"]
    y_merge0 = (ypre - hpre / 2 + ynone + hnone / 2) / 2
    seta(X_SPINE, yd - hd / 2, X_SPINE, ynone + hnone / 2)
    rotulo(X_SPINE - 0.014, (yd - hd / 2 + ypre) / 2 + 0.01, "post-NONE",
           ha="right")
    for x in (xe, xd):
        linha([(x, ypre - hpre / 2), (x, y_merge0), (X_SPINE, y_merge0)])
    ponto(X_SPINE, y_merge0)

    processo("none", "NONE is evaluated, if allowed:\n"
                     "rate, distortion and RD cost")

    # ---------------- entrada do pos-NONE ------------------------------------
    ye2, he2 = Y["e2"]
    folga = 0.012
    w = (X_ENT[1] - X_ENT[0] - folga) / 2
    centros = []
    for i, (t, destaque) in enumerate((("the same %d" % N_S1, False),
                                       ("%d RD of NONE" % N_EXTRA, True))):
        x0 = X_ENT[0] + i * (w + folga)
        ax.add_patch(Rectangle((x0, ye2 - he2 / 2), w, he2,
                               facecolor=p["s1"] if destaque else p["kept_f"],
                               edgecolor=p["s1"] if destaque else p["kept_e"],
                               linewidth=0.7, zorder=4))
        ax.text(x0 + w / 2, ye2, t, ha="center", va="center", fontsize=PT,
                color=p["surface"] if destaque else p["ink"], zorder=5)
        centros.append(x0 + w / 2)
    ycs, hcs = Y["cstop"]
    y_barra = (ye2 - he2 / 2 + ycs + hcs / 2) / 2
    for cx in centros:
        linha([(cx, ye2 - he2 / 2), (cx, y_barra)])
    linha([(X_SPINE, y_barra), (max(centros), y_barra)])
    ponto(X_SPINE, y_barra)
    seta(X_SPINE, ynone - hnone / 2, X_SPINE, ycs + hcs / 2)
    rotulo(X_SPINE - 0.014, (y_barra + ycs + hcs / 2) / 2,
           "%d" % (N_S1 + N_EXTRA), ha="right")

    # ---------------- parar o no ----------------------------------------------
    losango("cstop", CONDICOES["cstop"], pt=PT_MATH, meia=MEIA_LOS_MAT)
    tag(ycs, "post-NONE\noperating\npoint")
    sim(ycs, "node ends:\nno other type", terminal=True, meia=MEIA_LOS_MAT)
    yce, hce = Y["cext"]
    nao(ycs, hcs, yce + hce / 2)

    # ---------------- estendidas ----------------------------------------------
    losango("cext", CONDICOES["cext"], pt=PT_MATH, meia=MEIA_LOS_MAT)
    tag(yce, "extended\npruner on")
    h_a = sim(yce, "AB and\n4-way off", meia=MEIA_LOS_MAT)
    ymid, hmid = Y["mid"]
    nao(yce, hce, ymid + hmid / 2)
    reencontro(yce, h_a, (yce - hce / 2 + ymid + hmid / 2) / 2)

    processo("mid", "the enabled SPLIT, HORZ\nand VERT are evaluated")

    # ---------------- poda nativa de AB e 4-way -------------------------------
    ycn, hcn = losango("cnat", "native networks\nprune AB, 4-way")
    tag(ycn, "native, every\noperating\npoint")
    seta(X_SPINE, ymid - hmid / 2, X_SPINE, ycn + hcn / 2)
    h_n = sim(ycn, "pruned types\nare skipped")
    yfim, hfim = Y["fim"]
    nao(ycn, hcn, yfim + hfim / 2)
    reencontro(ycn, h_n, (ycn - hcn / 2 + yfim + hfim / 2) / 2)

    processo("fim", "the remaining AB and\n4-way types are evaluated")

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
