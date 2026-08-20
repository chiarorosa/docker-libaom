#!/usr/bin/env python3
"""Figura 2 do artigo LASCAS 2027 — pontos de insercao dos dois podadores na
busca de um no da arvore de particionamento do AV1, e a rede implantada atras
de cada um deles.

Diferente das figuras 1 e 3, esta nao plota medicao: e um diagrama estrutural.
Por isso NAO le artefato numerico, e sim declara em SPEC os fatos que desenha,
cada um com a sua procedencia no texto do artigo, para que uma mudanca de
arquitetura obrigue a mexer aqui e nao passe despercebida.

Faixa superior: a busca de um no na ordem de avaliacao nativa, da esquerda para
a direita, com os dois podadores marcados como os UNICOS estagios inseridos.

Faixa inferior, um cartao por ponto de insercao:
  - o vetor de entrada como BARRA SEGMENTADA, em escala de atributos, de modo
    que a barra do estagio 2 seja literalmente a do estagio 1 mais um segmento
    destacado. E o que torna visivel o empilhamento, e por que o segundo
    estagio nao pode agir antes: o segmento que ele acrescenta so existe depois
    que NONE foi medido;
  - a topologia e o banco de tres redes selecionadas pelo tamanho do no;
  - as saidas COM O LIMIAR QUE AS GOVERNA, que e a politica implantada e o
    conteudo mais reproduzivel do trabalho;
  - o alcance maximo do podador, que sustenta o argumento de complementaridade.

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
FIG_H_IN = 1.54

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

# Rampa monocromatica de cada acento, para os segmentos do vetor de entrada. Os
# tres primeiros segmentos usam a rampa; o quarto, exclusivo do estagio 2, usa o
# acento cheio, para que a diferenca entre os dois vetores salte a vista.
RAMPA_1 = ["#dbe4f2", "#c2d0e8", "#a9bcde"]
RAMPA_2 = ["#e0dcf0", "#cbc5e6", "#b6aedb"]

# --- os fatos desenhados, com procedencia ------------------------------------
# Ordem de avaliacao dentro do no: Secao II, "NONE first, then the quad-tree
# split, then the rectangular shapes, and finally the extended partitions".
# Composicao do vetor de atributos: Secao III-A. Topologia, contagem de
# parametros e dobra da padronizacao: Secao III-C. Acoes e limiares: Secao III-D.
SPEC = {
    "niveis":         ["64", "32", "16"],
    "nivel_terminal": "8",
    "parametros":     "27,759",
    # (rotulo, numero de atributos) na ordem em que o vetor os carrega.
    "blocos_s1":      [("luma descriptors", 24), ("causal context", 8),
                       ("qp and position", 4)],
    "bloco_extra_s2": ("log RD of NONE", 3),
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
                markeredgecolor="none", zorder=6, clip_on=False)
        ax.text(x, y, num, ha="center", va="center", fontsize=4.8,
                color=SURFACE, weight="bold", zorder=7)

    # ---------------- faixa superior: o percurso de um no -------------------
    # Larguras somam 0,85; as cinco folgas de 0,028 recebem as setas.
    Y0, H = 0.765, 0.225
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
                fontsize=6.6, color=tinta, linespacing=1.2,
                weight=("bold" if inserido else "normal"), zorder=4)

    for (x0, w, *_), (x1, *_) in zip(caixas, caixas[1:]):
        ax.annotate("", xy=(x1 - 0.003, Y0 + H / 2),
                    xytext=(x0 + w + 0.003, Y0 + H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED,
                                    linewidth=0.6, shrinkA=0, shrinkB=0,
                                    mutation_scale=4.5), zorder=3)

    # A recursao vai dentro da propria caixa do SPLIT: um arco de retorno aqui
    # invadiria a faixa inferior, e a informacao cabe em uma palavra.
    ax.text(0.554 + 0.130 / 2, Y0 + 0.046, "(recurses)", ha="center",
            va="center", fontsize=4.8, color=MUTED, style="italic", zorder=4)

    # Distintivo no canto superior DIREITO: a esquerda ele encostaria no rotulo
    # de duas linhas, que e centralizado e ocupa quase toda a largura da caixa.
    for (x0, w, *_), num, cor in ((caixas[0], "1", C_S1), (caixas[2], "2", C_S2)):
        distintivo(x0 + w - 0.020, Y0 + H - 0.042, num, cor)

    # ---------------- faixa inferior: a rede atras de cada ponto ------------
    PY0, PH = 0.305, 0.430          # o cartao
    BAR_Y0, BAR_H = 0.668, 0.052    # barra do vetor de entrada
    Y_LEGENDA = 0.628
    Y_TOPOLOGIA = 0.545
    Y_CHIPS, CHIP_H = 0.446, 0.100
    Y_ALCANCE = 0.330
    # Escala unica para as duas barras: 0,0093 de largura por atributo. E ela
    # que faz os tres primeiros segmentos coincidirem entre os dois cartoes.
    POR_ATRIBUTO = 0.0093

    paineis = [
        dict(px=0.005, cor=C_S1, rampa=RAMPA_1, num="1",
             blocos=SPEC["blocos_s1"], extra=None,
             legenda="24 luma $\\cdot$ 8 causal context $\\cdot$ 4 qp/position",
             topologia="36–64–32–3",
             chips=[("NONE", r"$p\,{\geq}\,\tau_{\mathrm{none}}$"),
                    ("SPLIT", r"$p\,{\geq}\,\tau_{\mathrm{split}}$"),
                    ("rest", r"$p\,{<}\,\tau_{\mathrm{rest}}$")],
             alcance="reach: up to the whole subtree"),
        dict(px=0.513, cor=C_S2, rampa=RAMPA_2, num="2",
             blocos=SPEC["blocos_s1"], extra=SPEC["bloco_extra_s2"],
             legenda="the same 36, plus 3 log RD terms of NONE",
             topologia="39–64–32–2",
             chips=[("skip AB + 4-way", r"$p\,{<}\,\theta$ per node size"),
                    ("evaluate them", "otherwise")],
             alcance="reach: the AB and 4-way family only"),
    ]

    for p in paineis:
        px, cor = p["px"], p["cor"]
        # Regua colorida na borda esquerda: e ela, com o distintivo, que liga o
        # cartao ao bloco inserido la em cima, sem um conector atravessando a
        # figura.
        ax.add_patch(Rectangle((px, PY0), 0.006, PH, facecolor=cor,
                               edgecolor="none", zorder=3))
        tx = px + 0.058

        # -- vetor de entrada, em escala de atributos ---------------------
        bx = tx
        for (_, n), tom in zip(p["blocos"], p["rampa"]):
            ax.add_patch(Rectangle((bx, BAR_Y0), n * POR_ATRIBUTO, BAR_H,
                                   facecolor=tom, edgecolor=SURFACE,
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
        if p["extra"] is not None:
            _, n = p["extra"]
            ax.add_patch(Rectangle((bx, BAR_Y0), n * POR_ATRIBUTO, BAR_H,
                                   facecolor=cor, edgecolor=SURFACE,
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
        ax.text(bx + 0.012, BAR_Y0 + BAR_H / 2,
                f"= {sum(n for _, n in p['blocos']) + (p['extra'][1] if p['extra'] else 0)}",
                ha="left", va="center", fontsize=5.2, color=INK, zorder=5)
        ax.text(tx, Y_LEGENDA, p["legenda"], ha="left", va="center",
                fontsize=5.0, color=MUTED, zorder=4)
        distintivo(px + 0.030, BAR_Y0 + BAR_H / 2, p["num"], cor)

        # -- topologia e banco de redes por nivel -------------------------
        ax.text(tx, Y_TOPOLOGIA, p["topologia"], ha="left", va="center",
                fontsize=5.8, color=INK, zorder=4)
        ax.text(px + 0.190, Y_TOPOLOGIA, "by node size:", ha="left",
                va="center", fontsize=5.2, color=INK_2, zorder=4)
        bx = px + 0.344
        for nivel in SPEC["niveis"]:
            ax.add_patch(Rectangle((bx, Y_TOPOLOGIA - 0.042), 0.038, 0.084,
                                   facecolor=SURFACE, edgecolor=cor,
                                   linewidth=0.6, zorder=4))
            ax.text(bx + 0.019, Y_TOPOLOGIA, nivel, ha="center", va="center",
                    fontsize=5.2, color=cor, zorder=5)
            bx += 0.044

        # -- saidas, cada uma com o limiar que a governa ------------------
        n_chips = len(p["chips"])
        folga = 0.020
        largura = (0.424 - folga * (n_chips - 1)) / n_chips
        cx = tx
        for rotulo, porta in p["chips"]:
            ax.add_patch(Rectangle((cx, Y_CHIPS - CHIP_H / 2), largura, CHIP_H,
                                   facecolor=SURFACE, edgecolor=cor,
                                   linewidth=0.5, zorder=4))
            ax.text(cx + largura / 2, Y_CHIPS + 0.020, rotulo, ha="center",
                    va="center", fontsize=5.2, color=INK, zorder=5)
            ax.text(cx + largura / 2, Y_CHIPS - 0.022, porta, ha="center",
                    va="center", fontsize=4.6, color=cor, zorder=5)
            cx += largura + folga

        ax.text(tx, Y_ALCANCE, p["alcance"], ha="left", va="center",
                fontsize=5.8, color=cor, zorder=4)

    # ---------------- notas comuns aos dois cartoes -------------------------
    # A dobra da padronizacao e a razao de o codificador alimentar atributos
    # crus e de nao ter sido preciso escrever inferencia nova em C.
    ax.text(0.5, 0.195, "standardization is folded into the first linear layer,"
            " so the encoder feeds raw attributes", ha="center", va="center",
            fontsize=5.0, color=INK_2, zorder=4)
    ax.text(0.5, 0.072, f"{SPEC['parametros']} parameters across the six"
            " deployed networks $\\cdot$ the "
            f"{SPEC['nivel_terminal']}$\\times${SPEC['nivel_terminal']}"
            " level is terminal and carries no network",
            ha="center", va="center", fontsize=4.6, color=MUTED,
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
