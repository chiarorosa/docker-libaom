#!/usr/bin/env python3
"""Figura 1 do artigo LASCAS 2027 — as dez formas de particao avaliadas em um no
da arvore do AV1, agrupadas nas quatro familias usadas no artigo e anotadas com
a parcela do tempo de busca local que cada familia consome.

Nao confundir com plot_partstats_fig1.py, que desenha a MESMA decomposicao de
custo como barras empilhadas por sequencia e e a figura da tese. Aqui o objeto
sao as formas: e o que o orientador pediu para a figura 1 do artigo, e e o que
a prosa nao consegue mostrar.

Os percentuais NAO sao redigitados. Sao lidos de results/thesis/figuras/
figura1_dados.csv, que e a saida auditavel do agregador analyze_partstats.py, e
conferidos contra os valores publicados no artigo. Se algum divergir, o script
aborta em vez de gerar uma figura que contradiz o texto.

CUIDADO METODOLOGICO, herdado do agregador: PARTITION_SPLIT fica FORA do
denominador, porque o seu cronometro engloba a recursao e contabilizaria o
trabalho dos nos descendentes varias vezes. O denominador sao os NOVE candidatos
nao recursivos.

A nota de rodape precisa dizer isso sem virar enigma. Nao basta anunciar que o
SPLIT foi excluido: o leitor conclui que dividir e de graca. O custo dele
EXISTE, so que aparece nos filhos, contabilizado como o tempo local DELES. Dai a
redacao "SPLIT's cost is counted at its sub-blocks", que diz onde a conta foi
parar, e nao apenas que ela saiu daqui.

Paleta monocromatica, e nao por sobriedade apenas: um diagrama de formas nao
carrega variavel categorica alguma, entao o matiz nao codificaria nada. As
subdivisoes sao mostradas por blocos preenchidos e afastados entre si, que e o
que se le num quadrado de 0,185 polegada, e nao por linhas de divisao.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_partition_shapes_fig1.py --out-dir results/thesis/figuras

Saidas: figura1_formas_particao.pdf (vetorial, 1:1 na largura de coluna)
        figura1_formas_particao.png (para conferencia visual)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_W_IN = 252.0 / 72.27   # \columnwidth do IEEEtran [conference], em polegadas

# --- paleta monocromatica ----------------------------------------------------
SURFACE = "#ffffff"
INK     = "#0b0b0b"
INK_2   = "#3f3e3b"
MUTED   = "#6e6d68"
BLOCO_F = "#dedcd5"   # preenchimento de um sub-bloco
BLOCO_E = "#8d8b85"   # contorno de um sub-bloco
NO_E    = "#0b0b0b"   # contorno do no inteiro

# --- geometria das dez formas ------------------------------------------------
# Cada forma e a lista dos seus sub-blocos em coordenadas do no, (x, y, w, h)
# com origem no canto inferior esquerdo. A convencao dos nomes A e B e a do
# libaom: HORZ_A divide a metade DE CIMA, HORZ_B divide a de baixo; VERT_A
# divide a metade DA ESQUERDA, VERT_B a da direita.
FORMAS = {
    "NONE":   [(0, 0, 1, 1)],
    "SPLIT":  [(0, .5, .5, .5), (.5, .5, .5, .5), (0, 0, .5, .5), (.5, 0, .5, .5)],
    "HORZ":   [(0, .5, 1, .5), (0, 0, 1, .5)],
    "VERT":   [(0, 0, .5, 1), (.5, 0, .5, 1)],
    "HORZ_A": [(0, .5, .5, .5), (.5, .5, .5, .5), (0, 0, 1, .5)],
    "HORZ_B": [(0, .5, 1, .5), (0, 0, .5, .5), (.5, 0, .5, .5)],
    "VERT_A": [(0, .5, .5, .5), (0, 0, .5, .5), (.5, 0, .5, 1)],
    "VERT_B": [(0, 0, .5, 1), (.5, .5, .5, .5), (.5, 0, .5, .5)],
    "HORZ_4": [(0, .75, 1, .25), (0, .5, 1, .25), (0, .25, 1, .25), (0, 0, 1, .25)],
    "VERT_4": [(0, 0, .25, 1), (.25, 0, .25, 1), (.5, 0, .25, 1), (.75, 0, .25, 1)],
}

# (rotulo da familia, chave do agregado, formas da linha)
LINHAS = [
    ("structural",  "PARTITION_NONE", ["NONE", "SPLIT"]),
    ("rectangular", "Rectangular",    ["HORZ", "VERT"]),
    ("AB",          "AB",             ["HORZ_A", "HORZ_B", "VERT_A", "VERT_B"]),
    ("4-way",       "4-way",          ["HORZ_4", "VERT_4"]),
]

# Valores publicados na Secao II. O CSV manda; isto e so a tranca.
ESPERADO = {"PARTITION_NONE": 30.1, "Rectangular": 35.6, "AB": 20.4,
            "4-way": 13.9, "AB+4-way": 34.3}
MAIOR_ESTENDIDA = 7.22
TOLERANCIA = 0.06

# --- ritmo, em polegadas -----------------------------------------------------
PAD_TOP    = 0.030
LADO       = 0.185         # lado do quadrado que representa um no
GAP_LINHA  = 0.052
GAP_NOME   = 0.026         # do quadrado ao nome da forma
GAP_NOTA   = 0.045
H_NOTA     = 0.070
PAD_BOT    = 0.025
FOLGA_BLOCO = 0.011        # afastamento entre sub-blocos vizinhos

X_ROTULO   = 0.030         # coluna do nome da familia
X_FORMAS   = 0.837         # inicio da primeira coluna de formas
PASSO      = (COL_W_IN - X_FORMAS - 0.030) / 4.0
X_NOTA     = 2.050         # folga a direita da linha 4-way, que so tem 2 formas

FIG_H_IN = (PAD_TOP + 4 * LADO + 3 * GAP_LINHA + GAP_NOTA + H_NOTA + PAD_BOT)


def ler_percentuais(csv_path):
    """Le o agregado do artefato do analyze_partstats.py e confere contra o que
    o artigo publica. Divergiu, aborta: uma figura que contradiz o texto e pior
    do que nenhuma figura."""
    agregado, estendidas = {}, {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            v = float(r["pct_node_local_time"])
            if r["panel"] == "families" and r["category"].startswith("Aggregate/"):
                agregado[r["category"].split("/", 1)[1]] = v
            elif r["panel"] == "extended_shapes":
                estendidas[r["category"]] = v

    for chave, esperado in ESPERADO.items():
        obtido = agregado.get(chave)
        if obtido is None:
            raise SystemExit("ABORTADO: %s ausente em %s" % (chave, csv_path))
        if abs(obtido - esperado) > TOLERANCIA:
            raise SystemExit(
                "ABORTADO: %s vale %.2f no artefato e %.1f no artigo"
                % (chave, obtido, esperado))
    maior = max(estendidas.values())
    if abs(maior - MAIOR_ESTENDIDA) > TOLERANCIA:
        raise SystemExit(
            "ABORTADO: a maior forma estendida vale %.2f no artefato e %.2f no"
            " artigo" % (maior, MAIOR_ESTENDIDA))
    print("  conferido: agregado e maior forma estendida reproduzem a Secao II")
    return agregado, maior


def draw(out_path, agregado, maior):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    def nx(v_in):
        return v_in / COL_W_IN

    def ny(v_in):
        return v_in / FIG_H_IN

    def desenha_forma(nome, x0_in, y0_in):
        """O no como contorno, e os sub-blocos preenchidos e afastados entre si.
        O afastamento e o que torna a subdivisao legivel neste tamanho: uma
        linha de divisao de 0,4 pt dentro de um quadrado de 0,185 polegada
        desaparece na impressao."""
        ax.add_patch(Rectangle((nx(x0_in), ny(y0_in)), nx(LADO), ny(LADO),
                               facecolor="none", edgecolor=NO_E,
                               linewidth=0.55, zorder=4))
        for bx, by, bw, bh in FORMAS[nome]:
            rx = x0_in + bx * LADO + FOLGA_BLOCO
            ry = y0_in + by * LADO + FOLGA_BLOCO
            rw = bw * LADO - 2 * FOLGA_BLOCO
            rh = bh * LADO - 2 * FOLGA_BLOCO
            ax.add_patch(Rectangle((nx(rx), ny(ry)), nx(rw), ny(rh),
                                   facecolor=BLOCO_F, edgecolor=BLOCO_E,
                                   linewidth=0.4, zorder=5))

    y_topo = FIG_H_IN - PAD_TOP
    for rotulo, chave, formas in LINHAS:
        y0 = y_topo - LADO
        yc = y0 + LADO / 2.0

        ax.text(nx(X_ROTULO), ny(yc), rotulo, ha="left", va="center",
                fontsize=5.4, color=INK, zorder=4)
        ax.text(nx(X_FORMAS - 0.055), ny(yc), "%.1f%%" % agregado[chave],
                ha="right",
                va="center", fontsize=5.6, color=INK, zorder=4)

        for i, nome in enumerate(formas):
            x0 = X_FORMAS + i * PASSO
            desenha_forma(nome, x0, y0)
            ax.text(nx(x0 + LADO + GAP_NOME), ny(yc), nome, ha="left",
                    va="center", fontsize=4.9, color=INK_2, zorder=6)

        # A folga da ultima linha recebe a leitura conjunta das duas familias
        # estendidas, que e o argumento da Secao II.
        if chave == "4-way":
            # A nota comeca onde a segunda forma termina, e nao numa coluna
            # da grade: a linha 4-way so ocupa duas colunas, e alinhar a nota
            # a coluna 3 desperdicaria meia polegada de folga util.
            x_nota = X_NOTA
            ax.text(nx(x_nota), ny(yc + 0.042),
                    "AB and 4-way are the extended", ha="left", va="center",
                    fontsize=4.8, color=INK_2, zorder=4)
            ax.text(nx(x_nota), ny(yc - 0.042),
                    "partitions: %.1f%%, none above %.2f%%"
                    % (agregado["AB+4-way"], maior),
                    ha="left", va="center", fontsize=4.8, color=INK_2, zorder=4)

        y_topo = y0 - GAP_LINHA

    ax.text(nx(X_ROTULO), ny(PAD_BOT + H_NOTA / 2),
            "share of the time a node spends on its own candidates;"
            " SPLIT's cost is counted at its sub-blocks", ha="left", va="center",
            fontsize=4.9, color=MUTED, zorder=4)

    fig.savefig(out_path, facecolor=SURFACE)
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=SURFACE, dpi=400)
    plt.close(fig)
    print("  gravado: %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    padrao = os.path.join(root, "results", "thesis", "figuras")
    ap.add_argument("--out-dir", default=padrao)
    ap.add_argument("--dados", default=os.path.join(padrao, "figura1_dados.csv"),
                    help="saida auditavel de plot_partstats_fig1.py")
    args = ap.parse_args()

    if not os.path.exists(args.dados):
        raise SystemExit(
            "ABORTADO: %s nao existe. Gere-o antes com\n"
            "  build/venv-ml/bin/python src/scripts/benchmark/"
            "plot_partstats_fig1.py" % args.dados)
    agregado, maior = ler_percentuais(args.dados)

    os.makedirs(args.out_dir, exist_ok=True)
    draw(os.path.join(args.out_dir, "figura1_formas_particao.pdf"), agregado,
         maior)
    print("\nDimensoes finais: %.3f x %.3f in (altura derivada do ritmo)"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
