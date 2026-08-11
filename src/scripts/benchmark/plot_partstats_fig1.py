#!/usr/bin/env python3
"""Figura 1 do artigo LASCAS 2027 — decomposicao do custo de busca por familia
de candidato de particao.

Reaproveita `scan` e `pools` de analyze_partstats.py, de modo que a figura e a
Tabela I do artigo derivem do MESMO agregador e nao possam divergir.

Painel superior: barras empilhadas horizontais, uma por sequencia do conjunto de
teste reservado mais o agregado, com as quatro familias (NONE, retangulares, AB,
4-way) em % do tempo de busca LOCAL do no.
Painel inferior: o agregado desagregado nas SEIS formas estendidas individuais,
que e o que sustenta a afirmacao de que nenhuma forma isolada ultrapassa 7,22%.

CUIDADO METODOLOGICO herdado do analyze_partstats.py: a coluna PARTITION_SPLIT e
deliberadamente excluida, pois o seu temporizador engloba a recursao e
contabilizaria o trabalho dos nos-descendentes varias vezes. O denominador e,
portanto, os NOVE candidatos nao recursivos.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_partstats_fig1.py \
        --out-dir results/thesis/figuras

Saidas: figura1_custo_por_familia.pdf (vetorial, para o artigo)
        figura1_custo_por_familia.png (para conferencia visual)
        figura1_custo_por_familia_print.pdf (com textura, para impressao em cinza)
        figura1_dados.csv (os valores plotados, para auditoria)
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_partstats import scan, pools, NAMES, AB, FOURWAY, LOCAL  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# --- paleta categorica sobria, VALIDADA antes de ser adotada -----------------
# Quatro tons profundos — azul-aco, vinho, violeta e verde-mata — escolhidos por
# leitura academica sobria, e nao pelo tema de abertura (verde-agua e amarelo),
# que e mais vivo e cujos dois tons claros ficavam abaixo de 3:1 contra a
# superficie. A ordem NAO e ciclada: cada familia tem slot fixo, e a cor segue a
# entidade, nunca a sua posicao no ranque.
#
# LIMITE MEDIDO DA SOBRIEDADE: dessaturar mais nao e possivel. Ha um piso de
# croma (0,10 em OKLCH) abaixo do qual a cor le como cinza, e paletas mais
# apagadas — Tol muted, seaborn deep, cinza-ardosia — foram testadas e
# REPROVARAM nesse piso ou no de distincao. Esta e a versao mais sobria entre as
# que passam em todas as verificacoes.
#
# Validador (modo claro, superficie #fcfcfb, pares adjacentes):
#   faixa de luminosidade  PASSA  os quatro dentro de L 0,43-0,77
#   piso de croma          PASSA  os quatro >= 0,10
#   separacao sob DCV      PASSA  pior par adjacente dE 13,5 (protan); tritan 7,4
#   piso de visao normal   PASSA  pior par adjacente dE 21,1
#   contraste x superficie PASSA  os quatro >= 3:1 (nenhum exige alivio)
# Reproduzir:
#   node validate_palette.js "#25599f,#9b2d4f,#4a3aa7,#1a6b33" --mode light
SERIES = [
    # (rotulo em ingles, chave em pools(), cor slot, textura para impressao)
    ("PARTITION_NONE", "NONE", "#25599f", ""),
    ("Rectangular",    "RECT", "#9b2d4f", "///"),
    ("AB",             "AB",   "#4a3aa7", "\\\\\\"),
    ("4-way",          "4WAY", "#1a6b33", "..."),
]

SURFACE = "#fcfcfb"    # superficie do grafico (tambem o vao entre segmentos)
INK      = "#0b0b0b"   # tinta primaria
INK_2    = "#52514e"   # tinta secundaria
MUTED    = "#898781"   # eixos e rotulos
GRID     = "#e1e0d9"   # linha de grade, capilar
BASELINE = "#c3c2b7"

# Vao de 2 px entre segmentos empilhados, expresso em pontos (2 px @ 96 dpi).
GAP_PT = 1.5


def _rel_lum(hexcolor):
    """Luminancia relativa WCAG de uma cor hexadecimal."""
    def chan(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def label_ink(fill):
    """Escolhe entre tinta clara e escura pelo MAIOR contraste sobre o
    preenchimento — calculado, e nunca estimado a olho. A regra de alivio exige
    rotulo legivel, e branco sobre o amarelo do slot 4 fica em torno de 2,2:1."""
    lum = _rel_lum(fill)
    contraste_claro = 1.05 / (lum + 0.05)
    contraste_escuro = (lum + 0.05) / 0.05
    return SURFACE if contraste_claro >= contraste_escuro else INK


def num(valor, casas=1):
    """As figuras dos artigos sao em INGLES, logo o separador decimal e o ponto.
    O texto em portugues dos capitulos usa virgula; os dois nao se misturam."""
    return f"{valor:.{casas}f}"

# Rotulo da linha agregada — em ingles, como todo texto plotado.
AGG_LABEL = "Aggregate"

DEFAULT_INPUTS = [
    ("Jockey",    "results/benchmark/partstats/part_timing_t1.csv"),
    ("RaceNight", "results/benchmark/partstats_racenight/part_timing.csv"),
    ("RiverBank", "results/benchmark/partstats_riverbank/part_timing.csv"),
]


def collect(inputs):
    """Agrega cada CSV e devolve (por_sequencia, agregado_por_forma, n_total)."""
    rows = []
    grand = {i: 0 for i in range(10)}
    total_nodes = 0
    for label, path in inputs:
        if not os.path.exists(path):
            sys.exit(f"ERRO: arquivo de entrada ausente: {path}")
        n, times, _att, skipped, _lbs = scan(path)
        if skipped:
            print(f"  aviso: {skipped} linhas descartadas em {label} "
                  f"(linhas rasgadas ou malformadas)")
        P = pools(times)
        local = P["local"]
        if local == 0:
            sys.exit(f"ERRO: tempo local nulo em {label}; CSV vazio ou invalido?")
        rows.append((label, n, {k: 100.0 * P[k] / local for k in
                                ("NONE", "RECT", "AB", "4WAY", "ABFW")}))
        for i in range(10):
            grand[i] += times[i]
        total_nodes += n
        print(f"  {label:12s} {n:>9,} nos  AB+4way = {100.0*P['ABFW']/local:5.2f}%")

    G = pools(grand)
    glocal = G["local"]
    rows.append((AGG_LABEL, total_nodes,
                 {k: 100.0 * G[k] / glocal for k in
                  ("NONE", "RECT", "AB", "4WAY", "ABFW")}))

    shapes = [(NAMES[i], 100.0 * grand[i] / glocal,
               "AB" if i in AB else "4WAY") for i in AB + FOURWAY]
    shapes.sort(key=lambda t: t[1], reverse=True)
    return rows, shapes, total_nodes, glocal


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)


def draw(rows, shapes, out_path, textura=False, titulo_n=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.6), dpi=200,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55})
    fig.patch.set_facecolor(SURFACE)

    # ---------------- painel superior: familias por sequencia ---------------
    labels = [r[0] for r in rows]
    y = range(len(rows))
    left = [0.0] * len(rows)
    for nome, chave, cor, hatch in SERIES:
        vals = [r[2][chave] for r in rows]
        ax1.barh(list(y), vals, left=left, height=0.62,
                 color=cor, edgecolor=SURFACE, linewidth=GAP_PT,
                 hatch=(hatch if textura else None), zorder=3)
        # Rotulo direto dentro do segmento: a regra de alivio exige rotulo
        # visivel, pois aqua e amarelo ficam abaixo de 3:1 na superficie clara.
        tinta = label_ink(cor)
        for yi, (v, l0) in enumerate(zip(vals, left)):
            if v >= 7.0:
                ax1.text(l0 + v / 2, yi, num(v), ha="center", va="center",
                         fontsize=7.5, color=tinta, weight="bold", zorder=4)
        left = [a + b for a, b in zip(left, vals)]

    # A mensagem da figura: o total das estendidas, anotado no fim da barra.
    for yi, r in enumerate(rows):
        ax1.text(101.5, yi, f"AB+4-way  {num(r[2]['ABFW'])}%",
                 va="center", ha="left", fontsize=8, color=INK_2)

    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("% of node-local search time (nine non-recursive candidates)",
                   fontsize=8, color=INK_2)
    style_axis(ax1)
    for lbl in ax1.get_yticklabels():
        if lbl.get_text() == AGG_LABEL:
            lbl.set_weight("bold")

    handles = [Patch(facecolor=c, edgecolor=SURFACE, linewidth=0.8,
                     hatch=(h if textura else None), label=n)
               for n, _k, c, h in SERIES]
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.30),
               ncol=4, frameon=False, fontsize=8, labelcolor=INK_2,
               handlelength=1.4, handleheight=0.9, columnspacing=1.6)

    # ---------------- painel inferior: as seis formas estendidas ------------
    cor_por_familia = {"AB": SERIES[2][2], "4WAY": SERIES[3][2]}
    hatch_por_familia = {"AB": SERIES[2][3], "4WAY": SERIES[3][3]}
    nomes = [s[0] for s in shapes]
    vals = [s[1] for s in shapes]
    cores = [cor_por_familia[s[2]] for s in shapes]
    y2 = range(len(shapes))
    for yi, (v, c, s) in enumerate(zip(vals, cores, shapes)):
        ax2.barh(yi, v, height=0.6, color=c, edgecolor=SURFACE,
                 linewidth=0.8, zorder=3,
                 hatch=(hatch_por_familia[s[2]] if textura else None))
        ax2.text(v + 0.25, yi, f"{num(v, 2)}%", va="center", ha="left",
                 fontsize=7.5, color=INK_2)

    maior = max(vals)
    ax2.set_yticks(list(y2))
    ax2.set_yticklabels(nomes)
    ax2.invert_yaxis()
    ax2.set_xlim(0, maior * 1.35)
    ax2.set_xlabel("% of node-local search time, per individual extended shape "
                   f"(largest single shape: {num(maior, 2)}%)",
                   fontsize=8, color=INK_2)
    style_axis(ax2)

    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", bbox_inches="tight",
                    facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print(f"  gravado: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/thesis/figuras")
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="pares rotulo=caminho; omitido usa os tres de teste")
    args = ap.parse_args()

    inputs = DEFAULT_INPUTS
    if args.inputs:
        inputs = [tuple(a.split("=", 1)) for a in args.inputs]

    os.makedirs(args.out_dir, exist_ok=True)
    print("Lendo part_timing (pode levar ~1 min, sao ~110 MB no total):")
    rows, shapes, n_total, glocal = collect(inputs)
    print(f"\n  agregado: {n_total:,} nos, tempo local {glocal/1e6:.1f} s")

    base = os.path.join(args.out_dir, "figura1_custo_por_familia")
    draw(rows, shapes, base + ".pdf", textura=False)
    draw(rows, shapes, base + "_print.pdf", textura=True)

    # Vista tabular dos valores plotados: torna a figura auditavel.
    csv_path = os.path.join(args.out_dir, "figura1_dados.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "category", "nodes", "pct_node_local_time"])
        for label, n, p in rows:
            for nome, chave, _c, _h in SERIES:
                w.writerow(["families", f"{label}/{nome}", n, f"{p[chave]:.2f}"])
            w.writerow(["families", f"{label}/AB+4-way", n, f"{p['ABFW']:.2f}"])
        for nome, v, fam in shapes:
            w.writerow(["extended_shapes", nome, n_total, f"{v:.2f}"])
    print(f"  gravado: {csv_path}")

    print("\nConferir antes de usar no artigo: os valores do agregado devem "
          "reproduzir a Tabela I (NONE 30,1 / RECT 35,6 / AB 20,4 / 4-way 13,9 "
          "/ AB+4-way 34,3) e a maior forma isolada deve ser 7,22%.")


if __name__ == "__main__":
    main()
