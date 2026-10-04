#!/usr/bin/env python3
"""Figura 1 do artigo ISCAS 2027 (NPL-AV1) — fluxograma da busca de um no em
codificacao All-Intra, com a PERTENCA de cada elemento explicita.

Mesma gramatica visual do fluxograma do LASCAS (plot_pruner_flowchart_fig2.py):
losango = decisao, retangulo = acao ou processo, blocos de entrada com a soma
dos atributos, trilhos de reencontro, texto em 8 pt, largura de coluna do
IEEEtran, altura derivada das linhas, PALETAS importadas do modulo do LASCAS.

Revisao de 2026-10-04 (pedido do usuario: a versao anterior nao deixava claro o
que e proposto, o que e nativo e que a CNN e desligada):
  - legenda de pertenca no topo: NPL-AV1 (preenchido, borda forte), libaom
    nativo (vazado, borda fina), nativo desligado no NPL-AV1 (tracejado);
  - os tres podadores propostos marcados (1), (2), (3) no ponto de insercao;
  - a CNN nativa aparece ao lado do ponto (1), tracejada e "disabled", ligada a
    ele por uma seta tracejada: o pre-busca ocupa o lugar dela;
  - saiu o losango "pruner of the preset": o ponto de operacao (quais
    podadores estao ligados) e configuracao, nao decisao por no; vai para a
    legenda do artigo e para o III-C.

Fatos desenhados, conferidos em src/aom/av1/encoder (2026-10-03):
  ordem: av1_prune_partitions_before_search (partition_search.c:5759) -> NONE
         (5829) -> av1_prune_after_none (5847) -> SPLIT (5857) -> RECT (5889)
         -> AB (5917) -> 4-way (5943).
  CNN intra: av1_intra_mode_cnn_partition (partition_strategy.c:2435), nivel 2
         em AI a partir de cpu-used=1; desligada por AV1_DISABLE_NATIVE_CNN.
  NONE pode nao ser avaliado (SPLIT forcado) -> "NONE is evaluated, if allowed".
  pos-NONE, parar o no: av1_disable_all_splits (student_h9c_decide).
  estendidas: h9d_skip_ext (partition_search.c:4113, 4181).
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
    ("leg",   0.140),   # legenda de pertenca
    ("gap",   0.060),
    ("start", 0.170),   # no em avaliacao
    ("gap",   0.050),
    ("e1",    0.150),   # entrada do pre-busca (a esquerda do eixo)
    ("gap",   0.090),
    ("c1",    0.450),   # (1) pre-busca | CNN nativa desligada a esquerda
    ("gap",   0.100),   # reencontro
    ("none",  0.250),
    ("gap",   0.060),
    ("e2",    0.150),   # entrada do pos-NONE e das estendidas
    ("gap",   0.085),
    ("cstop", 0.285),   # (2)
    ("gap",   0.085),
    ("cext",  0.285),   # (3)
    ("gap",   0.100),
    ("mid",   0.250),
    ("gap",   0.085),
    ("cnat",  0.420),   # redes nativas de AB e 4-way
    ("gap",   0.100),
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
X_SPINE = 0.520
MEIA_LOS = 0.190           # losangos de texto
MEIA_LOS_MAT = 0.170       # losangos de condicao
MEIA_PROC = 0.240          # caixas de processo nativas
X_CNN = (0.004, 0.275)     # CNN nativa desligada, ao lado do ponto (1)
X_ACAO = (0.755, 0.996)    # acoes dos ramos "yes"
X_ENT = (0.560, 0.996)     # blocos de entrada do pos-NONE (a direita)
X_ENT1 = (0.004, 0.470)    # blocos de entrada do pre-busca (a esquerda)
X_NOME = 0.004             # nome dos podadores (2) e (3), a esquerda

N_S1 = sum(n for _, n in SPEC["blocos_s1"])               # 36
N_EXTRA = SPEC["bloco_extra_s2"][1]                        # 3
TRACO = (0, (2.4, 1.6))


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # estilos de pertenca
    NPL = dict(face=p["kept_f"], edge=p["ink"], lw=1.2, ls="-", txt=p["ink"])
    NAT = dict(face=p["surface"], edge=p["ink_2"], lw=0.7, ls="-", txt=p["ink"])
    OFF = dict(face=p["surface"], edge=p["muted"], lw=0.8, ls=TRACO,
               txt=p["muted"])

    def seta(x0, y0, x1, y1, cor=None, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=cor or p["ink_2"],
                                    linewidth=0.8, linestyle=ls, shrinkA=0,
                                    shrinkB=0, mutation_scale=7.0), zorder=4)

    def linha(pts, cor=None):
        ax.plot([q[0] for q in pts], [q[1] for q in pts],
                color=cor or p["ink_2"], linewidth=0.8, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    def ponto(x, y):
        ax.plot([x], [y], marker="o", markersize=1.8, color=p["ink_2"],
                zorder=5)

    def rotulo(x, y, t, ha="center", cor=None, peso="normal"):
        ax.text(x, y, t, ha=ha, va="center", fontsize=PT,
                color=cor or p["ink_2"], zorder=6, linespacing=1.1,
                weight=peso)

    def caixa(x0, x1, yc, h, texto, st):
        ax.add_patch(Rectangle((x0, yc - h / 2), x1 - x0, h,
                               facecolor=st["face"], edgecolor=st["edge"],
                               linewidth=st["lw"], linestyle=st["ls"],
                               zorder=4))
        ax.text((x0 + x1) / 2, yc, texto, ha="center", va="center",
                fontsize=PT, color=st["txt"], linespacing=1.15, zorder=5)

    def losango(chave, texto, st, meia=MEIA_LOS, pt=PT):
        yc, h = Y[chave]
        ax.add_patch(Polygon([(X_SPINE, yc + h / 2), (X_SPINE + meia, yc),
                              (X_SPINE, yc - h / 2), (X_SPINE - meia, yc)],
                             closed=True, facecolor=st["face"],
                             edgecolor=st["edge"], linewidth=st["lw"],
                             zorder=4))
        ax.text(X_SPINE, yc, texto, ha="center", va="center", fontsize=pt,
                color=st["txt"], linespacing=1.1, zorder=5)
        return yc, h

    def marcador(x, y, n):
        ax.plot([x], [y], marker="o", markersize=10.0, color=p["ink"],
                markeredgecolor="none", zorder=7, clip_on=False)
        ax.text(x, y, n, ha="center", va="center", fontsize=PT,
                color=p["surface"], weight="bold", zorder=8)

    def processo(chave, texto):
        yc, h = Y[chave]
        caixa(X_SPINE - MEIA_PROC, X_SPINE + MEIA_PROC, yc, h, texto, NAT)
        return yc, h

    def nao(yc, h, y_alvo):
        seta(X_SPINE, yc - h / 2, X_SPINE, y_alvo)
        rotulo(X_SPINE - 0.014, yc - h / 2 - 0.34 * (yc - h / 2 - y_alvo),
               "no", ha="right")

    def sim(yc, texto, st, meia, folga_min=0.0):
        # folga_min afasta a caixa de acao do vertice quando o losango e largo,
        # para que o rotulo "yes" nao encoste nela (so o losango nativo precisa)
        x0 = max(X_ACAO[0], X_SPINE + meia + folga_min)
        seta(X_SPINE + meia, yc, x0, yc)
        rotulo((X_SPINE + meia + x0) / 2, yc + 0.020, "yes")
        nl = texto.count("\n") + 1
        h = (0.13 + 0.115 * nl) / FIG_H_IN
        caixa(x0, X_ACAO[1], yc, h, texto, st)
        return h

    def reencontro(yc_acao, h_acao, y_merge):
        xm = (X_ACAO[0] + X_ACAO[1]) / 2
        linha([(xm, yc_acao - h_acao / 2), (xm, y_merge), (X_SPINE, y_merge)])
        ponto(X_SPINE, y_merge)

    def entrada(chave, blocos, y_alvo_topo, total, xr=X_ENT):
        yc, h = Y[chave]
        folga = 0.010
        w = (xr[1] - xr[0] - folga * (len(blocos) - 1)) / len(blocos)
        cx = []
        for i, (t, destaque) in enumerate(blocos):
            x0 = xr[0] + i * (w + folga)
            ax.add_patch(Rectangle((x0, yc - h / 2), w, h,
                                   facecolor=p["ink"] if destaque else p["kept_f"],
                                   edgecolor=p["ink"] if destaque else p["kept_e"],
                                   linewidth=0.7, zorder=4))
            ax.text(x0 + w / 2, yc, t, ha="center", va="center", fontsize=PT,
                    color=p["surface"] if destaque else p["ink"], zorder=5)
            cx.append(x0 + w / 2)
        y_barra = (yc - h / 2 + y_alvo_topo) / 2
        for c in cx:
            linha([(c, yc - h / 2), (c, y_barra)])
        linha([(min(min(cx), X_SPINE), y_barra), (max(max(cx), X_SPINE), y_barra)])
        ponto(X_SPINE, y_barra)
        esq = max(cx) < X_SPINE
        rotulo(X_SPINE + (0.014 if esq else -0.014), (y_barra + y_alvo_topo) / 2,
               "%d" % total, ha="left" if esq else "right")

    # ---------------- legenda de pertenca ------------------------------------
    yl, hl = Y["leg"]
    itens = [(NPL, "NPL-AV1 (proposed)"), (NAT, "native libaom"),
             (OFF, "CNN, disabled")]
    xs = [0.004, 0.395, 0.700]
    for (st, t), x in zip(itens, xs):
        ax.add_patch(Rectangle((x, yl - hl * 0.35), 0.045, hl * 0.70,
                               facecolor=st["face"], edgecolor=st["edge"],
                               linewidth=st["lw"], linestyle=st["ls"],
                               zorder=4))
        rotulo(x + 0.058, yl, t, ha="left", cor=p["ink"])

    # ---------------- no e ponto (1): pre-busca --------------------------------
    ys, hs = processo("start", "node of 64, 32 or 16 samples")
    yc1, hc1 = Y["c1"]
    seta(X_SPINE, ys - hs / 2, X_SPINE, yc1 + hc1 / 2)
    entrada("e1", [("%d BC" % 24, False), ("%d NC" % 8, False),
                   ("%d CSP" % 4, False)], yc1 + hc1 / 2, N_S1, xr=X_ENT1)
    losango("c1", "pre-search\npruner acts", NPL)
    marcador(X_CNN[1] + 0.035, yc1 + hc1 * 0.38, "1")
    h_a1 = sim(yc1, "NONE only,\nSPLIT only or no\nrectangular types",
               NPL, MEIA_LOS)
    # CNN nativa, desligada no NPL-AV1, no mesmo ponto
    caixa(X_CNN[0], X_CNN[1], yc1, hc1 * 0.92,
          "native CNN\n(disabled): forces\nor forbids SPLIT", OFF)
    seta(X_CNN[1], yc1, X_SPINE - MEIA_LOS, yc1, cor=p["muted"], ls=TRACO)

    ynone, hnone = Y["none"]
    y_m1 = (yc1 - hc1 / 2 + ynone + hnone / 2) / 2
    nao(yc1, hc1, ynone + hnone / 2)
    reencontro(yc1, h_a1, y_m1)
    processo("none", "NONE is evaluated, if allowed:\n"
                     "rate, distortion and RD cost")

    # ---------------- pontos (2) e (3): pos-NONE e estendidas ----------------
    ycs, hcs = Y["cstop"]
    entrada("e2", [("the same %d" % N_S1, False),
                   ("%d RD of NONE" % N_EXTRA, True)], ycs + hcs / 2,
            N_S1 + N_EXTRA)
    seta(X_SPINE, ynone - hnone / 2, X_SPINE, ycs + hcs / 2)

    losango("cstop", r"$p_{\mathrm{stop}} > \tau_{\mathrm{stop}}$", NPL,
            meia=MEIA_LOS_MAT, pt=PT_MATH)
    marcador(X_NOME + 0.020, ycs, "2")
    rotulo(X_NOME + 0.052, ycs, "post-NONE\npruner", ha="left", cor=p["ink"])
    sim(ycs, "node ends", NPL, MEIA_LOS_MAT)
    yce, hce = Y["cext"]
    nao(ycs, hcs, yce + hce / 2)

    losango("cext", r"$p_{\mathrm{ext}} < \theta_{\mathrm{size}}$", NPL,
            meia=MEIA_LOS_MAT, pt=PT_MATH)
    marcador(X_NOME + 0.020, yce, "3")
    rotulo(X_NOME + 0.052, yce, "extended\npruner", ha="left", cor=p["ink"])
    h_a3 = sim(yce, "AB and\n4-way off", NPL, MEIA_LOS_MAT)
    ymid, hmid = Y["mid"]
    nao(yce, hce, ymid + hmid / 2)
    reencontro(yce, h_a3, (yce - hce / 2 + ymid + hmid / 2) / 2)
    processo("mid", "the enabled SPLIT, HORZ\nand VERT are evaluated")

    # ---------------- redes nativas de AB e 4-way -----------------------------
    ycn, hcn = losango("cnat", "native networks\nprune AB, 4-way", NAT,
                       meia=0.215)
    seta(X_SPINE, ymid - hmid / 2, X_SPINE, ycn + hcn / 2)
    h_n = sim(ycn, "pruned types\nare skipped", NAT, 0.215, folga_min=0.07)
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
