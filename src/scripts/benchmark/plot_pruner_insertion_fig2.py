#!/usr/bin/env python3
"""Figura 2 do artigo LASCAS 2027 — pontos de insercao dos dois podadores na
busca de um no da arvore de particionamento do AV1, e o banco de redes por nivel
atras de cada um deles.

Diferente das figuras 1 e 3, esta nao plota medicao: e um diagrama estrutural.
Por isso NAO le artefato numerico, e sim declara em SPEC os fatos que desenha,
cada um com a sua procedencia no texto do artigo, para que uma mudanca de
arquitetura obrigue a mexer aqui e nao passe despercebida.

Faixa superior: a busca de um no na ordem de avaliacao nativa, da esquerda para
a direita, com os dois podadores marcados como os UNICOS estagios inseridos. O
podador de pre-busca decide antes de qualquer avaliacao; o pos-NONE decide
depois que o custo do candidato indiviso ja foi medido.

Faixa inferior: o que ha atras de cada ponto de insercao — o vetor de atributos,
o banco de tres redes selecionadas pelo tamanho do no, e a topologia.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_pruner_insertion_fig2.py \
        --out-dir results/thesis/figuras

Saidas: figura2_pontos_insercao.pdf (vetorial, 1:1 na largura de coluna do
                                     IEEEtran, para \\includegraphics)
        figura2_pontos_insercao.png (para conferencia visual)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --- tipografia: identica a da figura 3 --------------------------------------
# O IEEEtran compoe em Times; STIXGeneral e a serif de metrica Times que
# acompanha o matplotlib. fonttype 42 embute como TrueType, pois o Type 3
# padrao costuma ser recusado pelo PDF eXpress do IEEE.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_W_IN = 252.0 / 72.27   # \columnwidth do IEEEtran [conference], em polegadas
FIG_H_IN = 1.42            # teto acordado com o enquadramento de 4+1 paginas

# --- paleta: os mesmos slots das figuras 1 e 3 -------------------------------
C_S1    = "#25599f"   # azul-aco — podador de pre-busca
C_S2    = "#4a3aa7"   # violeta  — podador pos-NONE
SURFACE = "#ffffff"
INK     = "#0b0b0b"
INK_2   = "#52514e"
MUTED   = "#898781"
NATIVE  = "#b9b7ae"   # contorno dos estagios nativos, que a solucao nao toca
TINT_1  = "#e6edf6"   # preenchimento tenue dos blocos inseridos
TINT_2  = "#eae7f5"

# --- os fatos desenhados, com procedencia ------------------------------------
# Ordem de avaliacao dentro do no: Secao II do artigo, "NONE first, then the
# quad-tree split, then the rectangular shapes, and finally the extended
# partitions". Topologias e contagem de acoes: Secoes III-C e III-D.
SPEC = {
    "ordem_nativa":   ["NONE", "SPLIT", "HORZ\nVERT", "AB +\n4-way"],
    "topologia_s1":   "36–64–32–3",
    "topologia_s2":   "39–64–32–2",
    "niveis":         ["64", "32", "16"],
    "nivel_terminal": "8",
    # Alcance maximo de cada podador, Secao III-D. O de pre-busca chega ao
    # limite quando compromete o no como NONE, o que corta a subarvore inteira;
    # o pos-NONE nunca passa da familia estendida.
    "alcance_s1":     "reach: up to the whole subtree",
    "alcance_s2":     "reach: the AB and 4-way family only",
}


def draw(out_path):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    def distintivo(x, y, num, cor):
        """Distintivo numerado. Desenhado como MARCADOR, e nao como Circle em
        coordenadas de dados: o eixo tem aspecto desigual, e um circulo em
        coordenadas de dados sairia elipse."""
        ax.plot([x], [y], marker="o", markersize=7.0, color=cor,
                markeredgecolor="none", zorder=5, clip_on=False)
        ax.text(x, y, num, ha="center", va="center", fontsize=4.8,
                color=SURFACE, weight="bold", zorder=6)

    # ---------------- faixa superior: o percurso de um no -------------------
    # Larguras somam 0,85; as cinco folgas de 0,028 recebem as setas.
    Y0, H = 0.715, 0.245
    caixas = [
        # (x0, largura, rotulo, cor da borda, preenchimento, cor do texto, inserido)
        (0.005, 0.175, "pre-search\npruner", C_S1, TINT_1, C_S1, True),
        (0.208, 0.115, "NONE",               NATIVE, SURFACE, INK, False),
        (0.351, 0.175, "post-NONE\npruner",  C_S2, TINT_2, C_S2, True),
        (0.554, 0.130, "SPLIT",              NATIVE, SURFACE, INK, False),
        (0.712, 0.125, "HORZ\nVERT",         NATIVE, SURFACE, INK, False),
        (0.865, 0.130, "AB +\n4-way",        NATIVE, SURFACE, INK, False),
    ]

    for x0, w, rotulo, borda, fundo, tinta, inserido in caixas:
        ax.add_patch(Rectangle((x0, Y0), w, H, facecolor=fundo,
                               edgecolor=borda,
                               linewidth=(0.8 if inserido else 0.6), zorder=3))
        ax.text(x0 + w / 2, Y0 + H / 2, rotulo, ha="center", va="center",
                fontsize=6.2, color=tinta, linespacing=1.2,
                weight=("bold" if inserido else "normal"), zorder=4)

    # Setas entre estagios consecutivos, na folga de 0,028.
    for (x0, w, *_), (x1, *_) in zip(caixas, caixas[1:]):
        ax.annotate("", xy=(x1 - 0.003, Y0 + H / 2),
                    xytext=(x0 + w + 0.003, Y0 + H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                    linewidth=0.6, shrinkA=0, shrinkB=0,
                                    mutation_scale=4.5), zorder=3)

    # A recursao vai dentro da propria caixa do SPLIT: um arco de retorno aqui
    # invadiria a faixa inferior, e a informacao cabe em uma palavra.
    ax.text(0.554 + 0.130 / 2, Y0 + 0.052, "(recurses)", ha="center",
            va="center", fontsize=4.8, color=MUTED, style="italic", zorder=4)

    # Distintivo no canto superior DIREITO: a esquerda ele encostaria no rotulo
    # de duas linhas, que e centralizado e ocupa quase toda a largura da caixa.
    for (x0, w, *_), num, cor in ((caixas[0], "1", C_S1), (caixas[2], "2", C_S2)):
        distintivo(x0 + w - 0.020, Y0 + H - 0.048, num, cor)

    # ---------------- faixa inferior: o que ha atras de cada ponto ----------
    PY0, PH = 0.075, 0.575
    paineis = [
        (0.005, C_S1, "1", "36 attributes, causal context only",
         "topology " + SPEC["topologia_s1"] + ", three actions",
         SPEC["alcance_s1"]),
        (0.513, C_S2, "2", "39 attributes, adding the RD of NONE",
         "topology " + SPEC["topologia_s2"] + ", two actions",
         SPEC["alcance_s2"]),
    ]
    LINHAS_Y = (0.560, 0.435, 0.310, 0.180)

    for px, cor, num, linha1, linha3, linha4 in paineis:
        # Regua colorida na borda esquerda: e ela, com o distintivo, que liga o
        # painel ao bloco inserido la em cima, sem um conector atravessando a
        # figura.
        ax.add_patch(Rectangle((px, PY0), 0.006, PH, facecolor=cor,
                               edgecolor="none", zorder=3))
        distintivo(px + 0.030, LINHAS_Y[0], num, cor)

        tx = px + 0.058
        ax.text(tx, LINHAS_Y[0], linha1, ha="left", va="center",
                fontsize=5.8, color=INK, zorder=4)

        # Banco de redes: uma por tamanho de no, desenhado como banco.
        ax.text(tx, LINHAS_Y[1], "networks by node size:", ha="left",
                va="center", fontsize=5.8, color=INK_2, zorder=4)
        bx = px + 0.325
        for nivel in SPEC["niveis"]:
            ax.add_patch(Rectangle((bx, LINHAS_Y[1] - 0.048), 0.044, 0.096,
                                   facecolor=SURFACE, edgecolor=cor,
                                   linewidth=0.6, zorder=4))
            ax.text(bx + 0.022, LINHAS_Y[1], nivel, ha="center", va="center",
                    fontsize=5.4, color=cor, zorder=5)
            bx += 0.051

        ax.text(tx, LINHAS_Y[2], linha3, ha="left", va="center",
                fontsize=5.8, color=INK_2, zorder=4)
        ax.text(tx, LINHAS_Y[3], linha4, ha="left", va="center",
                fontsize=5.8, color=cor, zorder=4)

    ax.text(0.5, 0.012,
            f"the {SPEC['nivel_terminal']}$\\times${SPEC['nivel_terminal']}"
            " level is terminal and carries no network",
            ha="center", va="bottom", fontsize=5.2, color=MUTED,
            style="italic", zorder=4)

    fig.savefig(out_path, facecolor=SURFACE)
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=SURFACE, dpi=400)
    plt.close(fig)
    print(f"  gravado: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out-dir",
                    default=os.path.join(root, "results", "thesis", "figuras"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    draw(os.path.join(args.out_dir, "figura2_pontos_insercao.pdf"))
    print(f"\nDimensoes finais: {COL_W_IN:.3f} x {FIG_H_IN:.2f} in, "
          f"para \\includegraphics[width=\\columnwidth]{{figura2_pontos_insercao}}")


if __name__ == "__main__":
    main()
