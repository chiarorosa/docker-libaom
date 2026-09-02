#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 — linha do tempo de disponibilidade de
informacao dentro da busca de um no da arvore de particionamento do AV1.

Como a figura 2 do LASCAS, esta NAO plota medicao: e um diagrama estrutural.
Por isso nao le artefato numerico, e sim declara em SPEC os fatos que desenha,
cada um com a sua procedencia, para que uma mudanca de instrumentacao ou de
composicao do vetor obrigue a mexer aqui e nao passe despercebida.

COMPOSICAO NOVA, e nao a da figura 2 do LASCAS. Aquela e uma matriz de cobertura
de acoes de poda; esta e uma linha do tempo de DISPONIBILIDADE. O eixo horizontal
e o percurso da busca dentro de um unico no, com tres instantes marcados, e cada
faixa e um item de informacao que comeca exatamente no instante em que passa a
existir:

  t0  antes da busca do no        — luminancia, quantizacao, posicao e a
                                    vizinhanca causal ja decidida;
  t1  depois de PARTITION_NONE    — taxa, distorcao e custo RD do candidato
                                    indiviso;
  t2  depois da recursao          — custo RD da subarvore otima.

E o alinhamento vertical que faz o argumento do artigo sem precisar de seta: as
tres faixas dos blocos A, B e C comecam todas em t0, ou seja, o podador pre-busca
as recebe de graca; a faixa do custo de NONE so comeca em t1, o que a torna
estruturalmente indisponivel a ele; e a faixa da subarvore otima, que e o alvo
que a perda de otimalidade mede, so comeca em t2.

Duas paletas, MESMA GEOMETRIA, como no gerador da figura 2 do LASCAS:
  cor   — acento azul-aco para o instante atacado;
  cinza — preto e escalas de cinza, para impressao monocromatica; o instante
          atacado deixa de ser separado por matiz e passa a se-lo pelo tracado
          cheio contra o pontilhado dos demais.

PISO TIPOGRAFICO DE 9 PT. O template do ICASSP 2027 exige corpo nao menor que
nove pontos em TODA a pagina, e nao so no texto corrido. Duas consequencias
governam a geometria abaixo:

  1. a figura e desenhada exatamente na largura de coluna do spconf.sty
     (\\textwidth 178 mm com \\columnsep 6 mm, logo \\columnwidth = 86 mm), para
     que \\includegraphics[width=\\columnwidth] fique em escala 1:1 e os 9 pt
     declarados aqui sejam 9 pt medidos na pagina;
  2. a 9 pt os rotulos nao cabem mais em uma linha na coluna estreita, entao
     passam a duas linhas e a figura fica mais alta. A COMPOSICAO NAO MUDA —
     rotulos a esquerda, regua de instantes a direita, linhas verticais
     continuas carregando o alinhamento —, mudam a quebra de linha e a altura.

Toda a geometria e declarada em PONTOS e convertida em fracao no fim, porque o
que precisa ser conferido contra a norma esta em pontos, e nao em fracao.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \
        src/scripts/benchmark/plot_information_timeline_icassp_fig1.py \
        --out-dir results/thesis/figuras

Saidas: figura1_linha_do_tempo.pdf       (variante de cor)
        figura1_linha_do_tempo_cinza.pdf (variante monocromatica)
        e um .png de cada, para conferencia visual.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- tipografia: identica a das figuras do LASCAS ----------------------------
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

# Largura de coluna do spconf.sty (ICASSP): 86 mm. Desenhando nesta largura, o
# \includegraphics[width=\columnwidth] fica em escala 1:1 e o corpo declarado
# aqui e o corpo medido na pagina.
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt PostScript
H_PT = 185.0                       # altura imposta pelos rotulos de 9 pt
COL_W_IN = W_PT / 72.0
FIG_H_IN = H_PT / 72.0

# Piso do template. Nenhum texto desta figura fica abaixo disto.
PT = 9.0

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#c9c7be", regua_leve="#e3e1d9",
        alvo="#25599f",                       # o instante que o artigo ataca
        barra_livre="#c2d0e8", borda_livre="#25599f",
        barra_tarde="#dcdad2", borda_tarde="#a8a7a1",
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#bcbab3", regua_leve="#e0ded7",
        alvo="#141413",
        barra_livre="#b5b3ac", borda_livre="#141413",
        barra_tarde="#e3e1db", borda_tarde="#a8a7a1",
    ),
}

# --- os fatos desenhados, com procedencia ------------------------------------
# Ordem de avaliacao e instantes: Secao II do artigo, conferida contra
# av1_rd_pick_partition() em src/aom/av1/encoder/partition_strategy.c.
# Composicao e contagem de colunas dos blocos: Secao III-A, espelhando
# src/scripts/partition_model/features.py (FEATURE_NAMES, RPP_SUBSETS).
# O bloco de taxa/distorcao/custo de NONE existe no vetor completo da tese
# (indices 38..40) e NAO e consumido por nenhuma representacao deste artigo:
# aparece aqui para mostrar POR QUE nao poderia ser.
# A quebra em duas linhas nao e enfeite: a 9 pt, "RD cost of the best partition"
# em linha unica ocuparia metade da largura da coluna e nao sobraria regua.
SPEC = {
    # O rotulo vai como texto simples: o matplotlib nao e LaTeX, e escapar o
    # sublinhado imprimiria a propria barra invertida.
    "instantes": [
        ("before the\nsearch", 0.055, "left"),
        ("after\nNONE",        0.520, "center"),
        ("after\nrecursion",   0.855, "center"),
    ],
    # (rotulo, indice do instante em que a informacao passa a existir)
    "faixas": [
        ("block A\n24 columns", 0),
        ("block B\n8 columns",  0),
        ("block C\n4 columns",  0),
        ("RD cost\nof NONE",    1),
        ("RD cost of the\nbest partition", 2),
    ],
}

# --- geometria, em pontos ----------------------------------------------------
MARGEM_X = 3.0             # respiro nas duas bordas laterais
LARG_ROTULO = 62.0         # coluna dos rotulos de faixa, a esquerda
X0_PT = MARGEM_X + LARG_ROTULO + 6.0   # inicio da regua de instantes
X1_PT = W_PT - MARGEM_X                # fim da regua
Y_EIXO_PT = 154.0          # linha do tempo
Y_TOPO_FAIXA_PT = 129.0    # centro da primeira faixa
# O passo e o rotulo de duas linhas (cerca de 20 pt) mais 6 pt de respiro: e o
# rotulo, e nao a barra, que dita a altura da linha depois do salto para 9 pt.
PASSO_PT = 26.0
FAIXA_H_PT = 11.0
Y_NOTA_PT = 8.0

X0, X1 = X0_PT / W_PT, X1_PT / W_PT
Y_EIXO = Y_EIXO_PT / H_PT
Y_TOPO_FAIXA = Y_TOPO_FAIXA_PT / H_PT
PASSO = PASSO_PT / H_PT
FAIXA_H = FAIXA_H_PT / H_PT


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(p["surface"])

    xs = [X0 + t * (X1 - X0) for _, t, _ha in SPEC["instantes"]]
    x_rotulo = MARGEM_X / W_PT

    # ---------------- eixo do tempo dentro do no ----------------------------
    # "search of one node" fica sobre a coluna de rotulos, alinhado a seta.
    ax.text(x_rotulo, Y_EIXO, "search of\none node", ha="left", va="center",
            fontsize=PT, color=p["ink_2"], linespacing=1.12, zorder=4)
    ax.annotate("", xy=(X1 + 2.0 / W_PT, Y_EIXO),
                xytext=(X0 - 8.0 / W_PT, Y_EIXO),
                arrowprops=dict(arrowstyle="-|>", color=p["muted"],
                                linewidth=0.55, shrinkA=0, shrinkB=0))

    # Os tres instantes. O primeiro e o que o artigo ataca: tracado cheio e
    # acento; os outros dois, pontilhados e neutros. Na variante monocromatica
    # e essa diferenca de tracado, e nao o matiz, que carrega a distincao.
    y_base = Y_TOPO_FAIXA - (len(SPEC["faixas"]) - 1) * PASSO - 9.0 / H_PT
    for i, ((rotulo, _, ha), x) in enumerate(zip(SPEC["instantes"], xs)):
        alvo = (i == 0)
        cor = p["alvo"] if alvo else p["muted"]
        ax.plot([x, x], [y_base, Y_EIXO], color=cor,
                linewidth=0.85 if alvo else 0.45, zorder=3,
                linestyle="-" if alvo else (0, (1.4, 1.4)))
        ax.plot([x], [Y_EIXO], marker="o", markersize=3.0 if alvo else 2.0,
                color=cor, markeredgecolor="none", zorder=5, clip_on=False)
        ax.text(x, Y_EIXO + 6.0 / H_PT, rotulo, ha=ha, va="bottom",
                fontsize=PT, color=p["ink"] if alvo else p["ink_2"],
                weight="bold" if alvo else "normal", linespacing=1.12,
                zorder=6)

    # ---------------- faixas de disponibilidade -----------------------------
    for k, (rotulo, inst) in enumerate(SPEC["faixas"]):
        y = Y_TOPO_FAIXA - k * PASSO
        cedo = (inst == 0)
        ax.text(x_rotulo, y, rotulo, ha="left", va="center", fontsize=PT,
                color=p["ink"] if cedo else p["ink_2"], linespacing=1.12,
                zorder=4)
        # trilho apagado antes de existir, barra cheia a partir do instante
        ax.plot([X0, X1], [y, y], color=p["regua_leve"], linewidth=0.4,
                zorder=1, solid_capstyle="butt")
        ax.add_patch(plt.Rectangle(
            (xs[inst], y - FAIXA_H / 2), X1 - xs[inst], FAIXA_H,
            facecolor=p["barra_livre"] if cedo else p["barra_tarde"],
            edgecolor=p["borda_livre"] if cedo else p["borda_tarde"],
            linewidth=0.4, zorder=2))

    # ---------------- nota de leitura ---------------------------------------
    # Encurtada em relacao a versao de 5,8 pt: a 9 pt a frase inteira nao cabe
    # na largura da coluna. O que saiu ("each band is") esta na legenda do .tex.
    ax.text(x_rotulo, Y_NOTA_PT / H_PT,
            "shaded from the instant the encoder holds it",
            ha="left", va="center", fontsize=PT, color=p["ink_2"], zorder=4)

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
                          "figura1_linha_do_tempo%s.pdf" % sufixo[nome]),
             PALETAS[nome])
    print("\nDimensoes finais: %.3f x %.2f in, para "
          "\\includegraphics[width=\\columnwidth]{figura1_linha_do_tempo}"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
