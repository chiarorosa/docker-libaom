#!/usr/bin/env python3
"""Figura 2 do artigo LASCAS 2027 — pontos de insercao dos dois podadores na
busca de um no da arvore de particionamento do AV1, e o efeito de cada decisao
sobre os estagios daquele no.

Diferente das figuras 1 e 3, esta nao plota medicao: e um diagrama estrutural.
Por isso NAO le artefato numerico, e sim declara em SPEC e em BLOCOS os fatos
que desenha, cada um com a sua procedencia, para que uma mudanca de arquitetura
ou de politica obrigue a mexer aqui e nao passe despercebida.

Composicao, em tres partes empilhadas sobre UM UNICO eixo horizontal:

  1. CABECALHO — a ordem nativa de avaliacao dentro de um no, da esquerda para a
     direita, com os dois podadores marcados como travas estreitas nos pontos em
     que sao inseridos: uma antes de NONE, outra logo depois de NONE.

  2. MATRIZ DE COBERTURA — uma faixa por acao da politica. A faixa declara a
     esquerda a condicao sobre a saida da rede e marca, coluna a coluna, se o
     estagio segue sendo avaliado ou se foi removido. As colunas da matriz sao
     EXATAMENTE as colunas do cabecalho, de modo que o alcance de cada podador
     se leia por alinhamento vertical, sem seta nem conector. E o alinhamento
     que torna geometrica a complementaridade defendida no texto: o corte do
     segundo estagio e um subconjunto estrito do corte do primeiro.

  3. LINHA DE ESPECIFICACAO de cada podador — o vetor de entrada como barra
     segmentada em escala de atributos, de modo que a barra do estagio 2 seja
     literalmente a do estagio 1 mais um segmento destacado. E o que torna
     visivel o empilhamento, e por que o segundo estagio nao pode agir antes: o
     segmento que ele acrescenta so existe depois que NONE foi medido.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_pruner_insertion_fig2.py --out-dir results/thesis/figuras

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
    "hatch.linewidth": 0.35,
})

COL_W_IN = 252.0 / 72.27   # \columnwidth do IEEEtran [conference], em polegadas
FIG_H_IN = 1.54            # teto medido: 1,62 empurra o Agradecimento a pag. 5

# --- paleta: os mesmos slots das figuras 1 e 3 -------------------------------
C_S1    = "#25599f"   # azul-aco — podador de pre-busca
C_S2    = "#4a3aa7"   # violeta  — podador pos-NONE
SURFACE = "#ffffff"
INK     = "#0b0b0b"
INK_2   = "#52514e"
MUTED   = "#898781"
NATIVE  = "#b9b7ae"   # contorno dos estagios nativos, que a solucao nao toca
KEPT_F  = "#dcdad2"   # celula ainda avaliada: preenchida, neutra
KEPT_E  = "#b9b7ae"

# Rampa monocromatica de cada acento, para os segmentos do vetor de entrada. Os
# tres primeiros segmentos usam a rampa; o quarto, exclusivo do estagio 2, usa o
# acento cheio, para que a diferenca entre os dois vetores salte a vista.
RAMPA_1 = ["#dbe4f2", "#c2d0e8", "#a9bcde"]
RAMPA_2 = ["#e0dcf0", "#cbc5e6", "#b6aedb"]

# --- os fatos desenhados, com procedencia ------------------------------------
# Ordem de avaliacao dentro do no: Secao II. Composicao do vetor: Secao III-A.
# Topologia e contagem de parametros: Secao III-C. Acoes e limiares: Secao
# III-D, cuja semantica foi conferida contra as chamadas de
# src/aom/av1/encoder/partition_strategy.c e as definicoes de encodeframe_utils.h:
#   probs[0] > tau_none  -> av1_disable_all_splits    (so NONE pode ser avaliado)
#   probs[1] > tau_split -> av1_set_square_split_only (NONE off, so o split)
#   probs[2] < tau_rest  -> av1_disable_rect_partitions (AB/4-way caem junto)
#   estagio 2: probs[0] < theta -> marca o no para pular AB/4-way apenas.
SPEC = {
    "niveis":         "16, 32 and 64",
    "nivel_terminal": "8",
    "parametros":     "27,759",
    # (rotulo, numero de atributos) na ordem em que o vetor os carrega.
    "blocos_s1":      [("luma descriptors", 24), ("causal context", 8),
                       ("qp and position", 4)],
    "bloco_extra_s2": ("log RD of NONE", 3),
}

# --- grade horizontal, compartilhada pelo cabecalho e pela matriz ------------
X_MATRIZ = 0.278           # inicio da regiao de colunas; a esquerda, os rotulos

# (rotulo, x0, x1). As folgas entre colunas ficam em branco tambem nas faixas,
# o que mantem a grade rigida e faz cada celula contar como uma unidade.
COLUNAS = [
    ("NONE",       0.313, 0.469),
    ("SPLIT",      0.509, 0.665),
    ("HORZ, VERT", 0.676, 0.838),
    # A ultima coluna para em 0,996: um traco de 0,45 pt centrado em x = 1,0
    # perderia metade da largura fora da figura.
    ("AB, 4-way",  0.849, 0.996),
]
# Travas: estreitas, sobre o proprio percurso, entre os estagios nativos.
TRAVAS = [("1", 0.278, 0.308, C_S1), ("2", 0.474, 0.504, C_S2)]

SPINE_Y0, SPINE_H = 0.906, 0.094
FAIXA_H = 0.066            # altura de uma faixa da matriz
PASSO   = 0.086            # de centro a centro de faixas consecutivas
POR_ATRIBUTO = 0.006       # escala unica das duas barras de entrada

# Um bloco por podador: identidade, acento, y da linha de especificacao, y da
# primeira faixa, e as acoes. Cada acao e (condicao, [avaliado?] por coluna).
BLOCOS = [
    dict(num="1", nome="pre-search pruner", cor=C_S1, rampa=RAMPA_1,
         y_spec=0.855, y_faixa=0.779,
         blocos=SPEC["blocos_s1"], extra=None,
         nota="no RD evidence yet", topologia="36 – 64 – 32 – 3",
         acoes=[
             (r"$p_{\mathrm{none}} > \tau_{\mathrm{none}}$",   [1, 0, 0, 0]),
             (r"$p_{\mathrm{split}} > \tau_{\mathrm{split}}$", [0, 1, 0, 0]),
             (r"$p_{\mathrm{rest}} < \tau_{\mathrm{rest}}$",   [1, 1, 0, 0]),
             ("otherwise", [1, 1, 1, 1]),
         ]),
    dict(num="2", nome="post-NONE pruner", cor=C_S2, rampa=RAMPA_2,
         y_spec=0.408, y_faixa=0.330,
         blocos=SPEC["blocos_s1"], extra=SPEC["bloco_extra_s2"],
         nota="+3 log RD from NONE", topologia="39 – 64 – 32 – 2",
         acoes=[
             (r"$p_{\mathrm{ext}} < \theta_{\mathrm{size}}$", [1, 1, 1, 0]),
         ]),
]


def draw(out_path):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)

    def distintivo(x, y, num, cor, tam=6.6):
        """Distintivo numerado. Desenhado como MARCADOR, e nao como Circle em
        coordenadas de dados: o eixo tem aspecto desigual, e um circulo em
        coordenadas de dados sairia elipse."""
        ax.plot([x], [y], marker="o", markersize=tam, color=cor,
                markeredgecolor="none", zorder=6, clip_on=False)
        ax.text(x, y, num, ha="center", va="center", fontsize=4.6,
                color=SURFACE, weight="bold", zorder=7)

    def regua(y, x0=0.0, x1=1.0, cor="#d9d7ce", lw=0.5):
        ax.plot([x0, x1], [y, y], color=cor, linewidth=lw, zorder=2,
                solid_capstyle="butt")

    # ---------------- cabecalho: a ordem nativa dentro de um no -------------
    ax.text(0.014, SPINE_Y0 + SPINE_H / 2 + 0.022, "search order",
            ha="left", va="center", fontsize=5.2, color=INK_2, zorder=4)
    ax.text(0.014, SPINE_Y0 + SPINE_H / 2 - 0.026, "within a node",
            ha="left", va="center", fontsize=5.2, color=INK_2, zorder=4)
    ax.annotate("", xy=(X_MATRIZ - 0.004, SPINE_Y0 + SPINE_H / 2),
                xytext=(0.196, SPINE_Y0 + SPINE_H / 2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.6,
                                shrinkA=0, shrinkB=0, mutation_scale=4.5),
                zorder=3)

    for rotulo, x0, x1 in COLUNAS:
        ax.add_patch(Rectangle((x0, SPINE_Y0), x1 - x0, SPINE_H,
                               facecolor=SURFACE, edgecolor=NATIVE,
                               linewidth=0.6, zorder=3))
        ax.text((x0 + x1) / 2, SPINE_Y0 + SPINE_H / 2, rotulo, ha="center",
                va="center", fontsize=6.0, color=INK, zorder=4)

    # Sem chevron entre as caixas: nas folgas de 0,011 o glifo sai menor que o
    # proprio traco da caixa e le como sujeira. A seta da esquerda, o rotulo
    # "search order" e a ordem das travas ja fixam o sentido da leitura.
    for num, x0, x1, cor in TRAVAS:
        ax.add_patch(Rectangle((x0, SPINE_Y0), x1 - x0, SPINE_H,
                               facecolor=cor, edgecolor="none", zorder=4))
        ax.text((x0 + x1) / 2, SPINE_Y0 + SPINE_H / 2, num, ha="center",
                va="center", fontsize=5.0, color=SURFACE, weight="bold",
                zorder=5)

    regua(SPINE_Y0 - 0.014, X_MATRIZ, 0.996, cor="#c9c7be", lw=0.6)

    # ---------------- um bloco por podador ----------------------------------
    for b in BLOCOS:
        cor = b["cor"]

        # -- linha de especificacao: identidade, vetor de entrada, topologia --
        y = b["y_spec"]
        distintivo(0.016, y, b["num"], cor)
        ax.text(0.034, y, b["nome"], ha="left", va="center", fontsize=6.0,
                color=cor, weight="bold", zorder=4)

        bx = X_MATRIZ
        for (_, n), tom in zip(b["blocos"], b["rampa"]):
            ax.add_patch(Rectangle((bx, y - 0.021), n * POR_ATRIBUTO, 0.042,
                                   facecolor=tom, edgecolor=SURFACE,
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
        total = sum(n for _, n in b["blocos"])
        if b["extra"] is not None:
            n = b["extra"][1]
            ax.add_patch(Rectangle((bx, y - 0.021), n * POR_ATRIBUTO, 0.042,
                                   facecolor=cor, edgecolor=SURFACE,
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
            total += n
        ax.text(bx + 0.011, y, "= %d" % total, ha="left", va="center",
                fontsize=5.6, color=INK, zorder=5)
        ax.text(0.575, y, b["nota"], ha="left", va="center", fontsize=5.0,
                color=(cor if b["extra"] is not None else MUTED), zorder=4)
        ax.text(0.996, y, b["topologia"], ha="right", va="center",
                fontsize=5.6, color=INK_2, zorder=4)

        # -- matriz de cobertura: uma faixa por acao da politica -------------
        for i, (condicao, mantidos) in enumerate(b["acoes"]):
            yc = b["y_faixa"] - i * PASSO
            ax.text(0.034, yc, condicao, ha="left", va="center", fontsize=5.8,
                    color=INK, zorder=4)
            for (_, x0, x1), vivo in zip(COLUNAS, mantidos):
                if vivo:
                    ax.add_patch(Rectangle((x0, yc - FAIXA_H / 2), x1 - x0,
                                           FAIXA_H, facecolor=KEPT_F,
                                           edgecolor=KEPT_E, linewidth=0.4,
                                           zorder=3))
                else:
                    ax.add_patch(Rectangle((x0, yc - FAIXA_H / 2), x1 - x0,
                                           FAIXA_H, facecolor=SURFACE,
                                           edgecolor=cor, linewidth=0.45,
                                           hatch="///", zorder=3))

    regua(0.455, 0.0, 0.996, cor="#e3e1d9", lw=0.5)
    regua(0.262, 0.0, 0.996, cor="#c9c7be", lw=0.6)

    # ---------------- legenda ----------------------------------------------
    ax.add_patch(Rectangle((0.014, 0.176), 0.030, 0.038, facecolor=KEPT_F,
                           edgecolor=KEPT_E, linewidth=0.4, zorder=4))
    ax.text(0.052, 0.195, "still evaluated", ha="left", va="center",
            fontsize=5.0, color=INK_2, zorder=4)
    ax.add_patch(Rectangle((0.226, 0.176), 0.030, 0.038, facecolor=SURFACE,
                           edgecolor=INK_2, linewidth=0.45, hatch="///",
                           zorder=4))
    ax.text(0.264, 0.195, "removed by the pruner", ha="left", va="center",
            fontsize=5.0, color=INK_2, zorder=4)

    bx = 0.014
    chave = list(zip(SPEC["blocos_s1"], RAMPA_1)) + \
        [(SPEC["bloco_extra_s2"], C_S2)]
    for (rotulo, n), tom in chave:
        ax.add_patch(Rectangle((bx, 0.091), 0.022, 0.036, facecolor=tom,
                               edgecolor=SURFACE, linewidth=0.4, zorder=4))
        ax.text(bx + 0.028, 0.109, "%d %s" % (n, rotulo), ha="left",
                va="center", fontsize=4.8, color=INK_2, zorder=4)
        bx += 0.028 + (len(rotulo) + 3) * 0.0107

    ax.text(0.014, 0.026,
            "one network per node size, %s pixels; the %s$\\times$%s level is"
            " terminal $\\cdot$ %s parameters in total"
            % (SPEC["niveis"], SPEC["nivel_terminal"], SPEC["nivel_terminal"],
               SPEC["parametros"]),
            ha="left", va="center", fontsize=4.7, color=MUTED,
            style="italic", zorder=4)

    fig.savefig(out_path, facecolor=SURFACE)
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=SURFACE, dpi=400)
    plt.close(fig)
    print("  gravado: %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out-dir",
                    default=os.path.join(root, "results", "thesis", "figuras"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    draw(os.path.join(args.out_dir, "figura2_pontos_insercao.pdf"))
    print("\nDimensoes finais: %.3f x %.2f in, para "
          "\\includegraphics[width=\\columnwidth]{figura2_pontos_insercao}"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
