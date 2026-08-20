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

Duas paletas, MESMA GEOMETRIA, para que a escolha seja comparada sem que nada
mais mude entre as versoes:

  cor   — os acentos das figuras 1 e 3, azul-aco e violeta;
  cinza — preto e escalas de cinza, para impressao monocromatica. Os dois
          podadores deixam de ser separados por matiz (nao sobreviveria ao
          cinza) e passam a ser separados pelo distintivo numerado e pelo bloco
          em que estao, que e a informacao que de fato os distingue.

Em ambas, a celula removida e preenchida, e nao hachurada: no tamanho final a
hachura de 0,35 pt quase desaparece, sobretudo na amostra da legenda.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_pruner_insertion_fig2.py --out-dir results/thesis/figuras

Saidas: figura2_pontos_insercao.pdf       (variante de cor, a do artigo)
        figura2_pontos_insercao_cinza.pdf (variante monocromatica)
        e um .png de cada, para conferencia visual.
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
FIG_H_IN = 1.54            # teto medido: 1,62 empurra o Agradecimento a pag. 5

# --- as duas paletas ---------------------------------------------------------
# Chaves comuns:
#   s1, s2    acento de cada podador (distintivo, trava, nome, nota, segmento
#             extra do vetor);
#   kept_*    celula ainda avaliada;
#   cut_*     celula removida, uma cor por podador na variante de cor e uma so
#             na monocromatica, onde a identidade vem do bloco e do distintivo;
#   rampa     os tres primeiros segmentos do vetor de entrada, IDENTICA nos
#             dois podadores: sao literalmente os mesmos 36 atributos, e so o
#             quarto segmento, em s2, distingue o vetor do segundo estagio.
PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        native="#b9b7ae", regua="#c9c7be", regua_leve="#e3e1d9",
        s1="#25599f", s2="#4a3aa7",
        kept_f="#dcdad2", kept_e="#b9b7ae",
        cut_f={"s1": "#8ba5cd", "s2": "#9d94ca"},
        cut_e={"s1": "#25599f", "s2": "#4a3aa7"},
        cut_legenda="#94a0c0",
        rampa=["#dbe4f2", "#c2d0e8", "#a9bcde"],
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        native="#a8a7a1", regua="#bcbab3", regua_leve="#e0ded7",
        s1="#141413", s2="#141413",
        kept_f="#e3e1db", kept_e="#b5b3ac",
        cut_f={"s1": "#6b6a65", "s2": "#6b6a65"},
        cut_e={"s1": "#6b6a65", "s2": "#6b6a65"},
        cut_legenda="#6b6a65",
        rampa=["#e2e1dc", "#c7c5be", "#a9a7a0"],
    ),
}

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
TRAVAS = [("1", "s1", 0.278, 0.308), ("2", "s2", 0.474, 0.504)]

SPINE_Y0, SPINE_H = 0.906, 0.094
FAIXA_H = 0.066            # altura de uma faixa da matriz
PASSO   = 0.086            # de centro a centro de faixas consecutivas
POR_ATRIBUTO = 0.006       # escala unica das duas barras de entrada

# Um bloco por podador: identidade, chave de acento, y da linha de
# especificacao, y da primeira faixa, e as acoes. Cada acao e
# (condicao, [avaliado?] por coluna).
BLOCOS = [
    dict(num="1", nome="pre-search pruner", chave="s1",
         y_spec=0.855, y_faixa=0.779,
         blocos=SPEC["blocos_s1"], extra=None,
         nota="no RD evidence yet", topologia="36 – 64 – 32 – 3",
         acoes=[
             (r"$p_{\mathrm{none}} > \tau_{\mathrm{none}}$",   [1, 0, 0, 0]),
             (r"$p_{\mathrm{split}} > \tau_{\mathrm{split}}$", [0, 1, 0, 0]),
             (r"$p_{\mathrm{rest}} < \tau_{\mathrm{rest}}$",   [1, 1, 0, 0]),
             ("otherwise", [1, 1, 1, 1]),
         ]),
    dict(num="2", nome="post-NONE pruner", chave="s2",
         y_spec=0.408, y_faixa=0.330,
         blocos=SPEC["blocos_s1"], extra=SPEC["bloco_extra_s2"],
         nota="+3 log RD from NONE", topologia="39 – 64 – 32 – 2",
         acoes=[
             (r"$p_{\mathrm{ext}} < \theta_{\mathrm{size}}$", [1, 1, 1, 0]),
         ]),
]


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(p["surface"])

    def distintivo(x, y, num, cor, tam=6.6):
        """Distintivo numerado. Desenhado como MARCADOR, e nao como Circle em
        coordenadas de dados: o eixo tem aspecto desigual, e um circulo em
        coordenadas de dados sairia elipse."""
        ax.plot([x], [y], marker="o", markersize=tam, color=cor,
                markeredgecolor="none", zorder=6, clip_on=False)
        ax.text(x, y, num, ha="center", va="center", fontsize=4.6,
                color=p["surface"], weight="bold", zorder=7)

    def regua(y, x0, x1, cor, lw):
        ax.plot([x0, x1], [y, y], color=cor, linewidth=lw, zorder=2,
                solid_capstyle="butt")

    # ---------------- cabecalho: a ordem nativa dentro de um no -------------
    ax.text(0.014, SPINE_Y0 + SPINE_H / 2 + 0.022, "search order",
            ha="left", va="center", fontsize=5.2, color=p["ink_2"], zorder=4)
    ax.text(0.014, SPINE_Y0 + SPINE_H / 2 - 0.026, "within a node",
            ha="left", va="center", fontsize=5.2, color=p["ink_2"], zorder=4)
    ax.annotate("", xy=(X_MATRIZ - 0.004, SPINE_Y0 + SPINE_H / 2),
                xytext=(0.196, SPINE_Y0 + SPINE_H / 2),
                arrowprops=dict(arrowstyle="-|>", color=p["muted"],
                                linewidth=0.6, shrinkA=0, shrinkB=0,
                                mutation_scale=4.5), zorder=3)

    for rotulo, x0, x1 in COLUNAS:
        ax.add_patch(Rectangle((x0, SPINE_Y0), x1 - x0, SPINE_H,
                               facecolor=p["surface"], edgecolor=p["native"],
                               linewidth=0.6, zorder=3))
        ax.text((x0 + x1) / 2, SPINE_Y0 + SPINE_H / 2, rotulo, ha="center",
                va="center", fontsize=6.0, color=p["ink"], zorder=4)

    # Sem chevron entre as caixas: nas folgas de 0,011 o glifo sai menor que o
    # proprio traco da caixa e le como sujeira. A seta da esquerda, o rotulo
    # "search order" e a ordem das travas ja fixam o sentido da leitura.
    for num, chave, x0, x1 in TRAVAS:
        ax.add_patch(Rectangle((x0, SPINE_Y0), x1 - x0, SPINE_H,
                               facecolor=p[chave], edgecolor="none", zorder=4))
        ax.text((x0 + x1) / 2, SPINE_Y0 + SPINE_H / 2, num, ha="center",
                va="center", fontsize=5.0, color=p["surface"], weight="bold",
                zorder=5)

    regua(SPINE_Y0 - 0.014, X_MATRIZ, 0.996, p["regua"], 0.6)

    # ---------------- um bloco por podador ----------------------------------
    for b in BLOCOS:
        chave = b["chave"]
        cor = p[chave]
        rampa = p["rampa"]

        # -- linha de especificacao: identidade, vetor de entrada, topologia --
        y = b["y_spec"]
        distintivo(0.016, y, b["num"], cor)
        ax.text(0.034, y, b["nome"], ha="left", va="center", fontsize=6.0,
                color=cor, weight="bold", zorder=4)

        bx = X_MATRIZ
        for (_, n), tom in zip(b["blocos"], rampa):
            ax.add_patch(Rectangle((bx, y - 0.021), n * POR_ATRIBUTO, 0.042,
                                   facecolor=tom, edgecolor=p["surface"],
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
        total = sum(n for _, n in b["blocos"])
        if b["extra"] is not None:
            n = b["extra"][1]
            ax.add_patch(Rectangle((bx, y - 0.021), n * POR_ATRIBUTO, 0.042,
                                   facecolor=cor, edgecolor=p["surface"],
                                   linewidth=0.5, zorder=4))
            bx += n * POR_ATRIBUTO
            total += n
        ax.text(bx + 0.011, y, "= %d" % total, ha="left", va="center",
                fontsize=5.6, color=p["ink"], zorder=5)
        ax.text(0.575, y, b["nota"], ha="left", va="center", fontsize=5.0,
                color=(cor if b["extra"] is not None else p["muted"]), zorder=4)
        ax.text(0.996, y, b["topologia"], ha="right", va="center",
                fontsize=5.6, color=p["ink_2"], zorder=4)

        # -- matriz de cobertura: uma faixa por acao da politica -------------
        for i, (condicao, mantidos) in enumerate(b["acoes"]):
            yc = b["y_faixa"] - i * PASSO
            ax.text(0.034, yc, condicao, ha="left", va="center", fontsize=5.8,
                    color=p["ink"], zorder=4)
            for (_, x0, x1), vivo in zip(COLUNAS, mantidos):
                face = p["kept_f"] if vivo else p["cut_f"][chave]
                borda = p["kept_e"] if vivo else p["cut_e"][chave]
                ax.add_patch(Rectangle((x0, yc - FAIXA_H / 2), x1 - x0,
                                       FAIXA_H, facecolor=face,
                                       edgecolor=borda, linewidth=0.45,
                                       zorder=3))

    regua(0.455, 0.0, 0.996, p["regua_leve"], 0.5)
    regua(0.262, 0.0, 0.996, p["regua"], 0.6)

    # ---------------- legenda ----------------------------------------------
    # Amostras maiores que as antigas e preenchidas: a hachura de 0,35 pt
    # sumia justamente aqui, num retangulo de 0,10 x 0,06 polegada.
    ax.add_patch(Rectangle((0.014, 0.172), 0.038, 0.046,
                           facecolor=p["kept_f"], edgecolor=p["kept_e"],
                           linewidth=0.45, zorder=4))
    ax.text(0.060, 0.195, "still evaluated", ha="left", va="center",
            fontsize=5.2, color=p["ink_2"], zorder=4)
    ax.add_patch(Rectangle((0.238, 0.172), 0.038, 0.046,
                           facecolor=p["cut_legenda"], edgecolor=p["ink_2"],
                           linewidth=0.45, zorder=4))
    ax.text(0.284, 0.195, "removed by the pruner", ha="left", va="center",
            fontsize=5.2, color=p["ink_2"], zorder=4)

    bx = 0.014
    chave_vetor = list(zip(SPEC["blocos_s1"], p["rampa"])) + \
        [(SPEC["bloco_extra_s2"], p["s2"])]
    for (rotulo, n), tom in chave_vetor:
        ax.add_patch(Rectangle((bx, 0.089), 0.024, 0.038, facecolor=tom,
                               edgecolor=p["surface"], linewidth=0.4, zorder=4))
        ax.text(bx + 0.030, 0.108, "%d %s" % (n, rotulo), ha="left",
                va="center", fontsize=4.9, color=p["ink_2"], zorder=4)
        bx += 0.030 + (len(rotulo) + 3) * 0.0109

    # Rodape: 5,4 pt em cinza escuro e sem italico. O italico de 4,7 pt em
    # cinza claro da versao anterior ficava abaixo do piso de legibilidade.
    ax.text(0.014, 0.030,
            "one network per node size — %s pixels; %s$\\times$%s is terminal"
            " $\\cdot$ %s parameters"
            % (SPEC["niveis"], SPEC["nivel_terminal"], SPEC["nivel_terminal"],
               SPEC["parametros"]),
            ha="left", va="center", fontsize=5.4, color=p["ink_2"], zorder=4)

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
                          "figura2_pontos_insercao%s.pdf" % sufixo[nome]),
             PALETAS[nome])
    print("\nDimensoes finais: %.3f x %.2f in, para "
          "\\includegraphics[width=\\columnwidth]{figura2_pontos_insercao}"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
