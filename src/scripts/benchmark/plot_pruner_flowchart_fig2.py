#!/usr/bin/env python3
"""Figura 2 do artigo LASCAS 2027, VARIANTE FLUXOGRAMA — a politica dos dois
podadores desenhada como cascata de condicionais.

Alternativa a matriz de cobertura de plot_pruner_insertion_fig2.py, para ser
comparada com ela. As duas dizem a mesma coisa e privilegiam leituras
diferentes:

  matriz     — mostra o ALCANCE. Cada acao vira uma faixa e o corte do segundo
               estagio se le como subconjunto estrito do corte do primeiro, por
               alinhamento vertical. Nao mostra a ordem em que as condicoes sao
               testadas nem que a cascata e um if/else if.
  fluxograma — mostra o CONTROLE. A cascata aparece como cascata, a saida
               antecipada do compromisso com o split aparece como terminal, e a
               dependencia do segundo estagio fica explicita: ele so pode ser
               consultado depois da caixa em que NONE e medido, porque e dessa
               caixa que vem o bloco de tres atributos que ele acrescenta.
               Em compensacao, o alcance de cada acao volta a ser texto.

E o formato pedido pelo padrao de escrita do grupo (CLAUDE.md, B.7:
"figura de fluxograma + caminhada textual passo a passo por ela").

Como as figuras 1 e 3, e como a variante matriz, NAO le artefato numerico: e um
diagrama estrutural. Importa SPEC e PALETAS do modulo da matriz para que a
composicao do vetor de atributos e a paleta nao possam divergir entre as duas
versoes da mesma figura.

Semantica das condicoes conferida contra src/aom/av1/encoder/partition_strategy.c
e as definicoes de encodeframe_utils.h:
  probs[0] > tau_none  -> av1_disable_all_splits    (so NONE pode ser avaliado)
  probs[1] > tau_split -> av1_set_square_split_only (NONE off, so o split)
  probs[2] < tau_rest  -> av1_disable_rect_partitions (AB/4-way caem junto)
  estagio 2: probs[0] < theta -> marca o no para pular AB/4-way apenas.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_pruner_flowchart_fig2.py --out-dir results/thesis/figuras

Saidas: figura2_fluxograma.pdf (vetorial, 1:1 na largura de coluna do IEEEtran)
        figura2_fluxograma.png (para conferencia visual)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

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

COL_W_IN = 252.0 / 72.27   # lumnwidth do IEEEtran [conference], em polegadas
PT = 9.0                   # piso tipografico do artigo: nada abaixo disto

# --- ritmo vertical ----------------------------------------------------------
# A altura da figura e DERIVADA das linhas, e nao arbitrada: assim mexer em uma
# caixa nao desalinha o resto nem exige reajustar um numero solto la em cima.
# Alturas em polegadas.
LINHAS = [
    ("pad",  0.050),
    ("e1",   0.200),   # blocos de entrada do estagio 1
    ("gap",  0.105),   # junta + seta para o primeiro losango
    ("c1",   0.340),   # p_none  > tau_none
    ("gap",  0.075),
    ("c2",   0.340),   # p_split > tau_split
    ("gap",  0.075),
    ("c3",   0.340),   # p_rest  < tau_rest
    ("gap",  0.125),   # aqui volta o trilho de reencontro do estagio 1
    ("none", 0.330),   # NONE e avaliado, duas linhas
    ("gap",  0.065),
    ("e2",   0.200),   # blocos de entrada do estagio 2
    ("gap",  0.105),
    ("c4",   0.340),   # p_ext < theta
    ("gap",  0.125),   # aqui volta o trilho do estagio 2
    ("fim",  0.335),   # duas linhas
    ("pad",  0.040),
]
FIG_H_IN = sum(h for _, h in LINHAS)

Y = {}
_cursor = 1.0
for _nome, _h in LINHAS:
    _hn = _h / FIG_H_IN
    if _nome not in ("pad", "gap"):
        Y[_nome] = (_cursor - _hn / 2.0, _hn)   # (centro, altura normalizada)
    _cursor -= _hn

# --- grade horizontal --------------------------------------------------------
# O eixo corre encostado a esquerda porque as linhas de losango nao tem nada
# naquela margem: so as linhas de entrada precisam da coluna do nome. Isso
# devolve quase um quinto de largura as caixas de consequencia, que eram o
# ponto em que o texto encostava na borda.
X_SPINE   = 0.265          # eixo dos losangos e das caixas de processo
MEIA_LOS  = 0.200          # meia-largura do losango
MEIA_PROC = 0.215          # meia-largura da caixa de processo
X_ACAO_0  = 0.558          # caixas de consequencia, a direita do ramo "yes"
X_ACAO_1  = 0.944
X_TRILHO  = 0.974          # trilho vertical de reencontro
X_NOME    = 0.020          # coluna do nome do podador, nas linhas de entrada
X_ENT_0   = 0.280          # primeiro bloco de entrada
X_ENT_1   = 0.996

# Nomes curtos dos blocos do vetor: a contagem continua vindo de SPEC, para
# nao divergir da Secao III-A, mas o rotulo encurta. Com o piso de 9 pt, tres
# caixas lado a lado numa coluna de 3,49 in dao cerca de 17 caracteres cada.
CURTO = {"luma descriptors": "luma", "causal context": "context",
         "qp and position": "qp, pos"}

# --- textos ------------------------------------------------------------------
# Blocos do vetor de atributos: vem de SPEC, para nao divergirem da Secao III-A.
ENTRADAS_1 = ["%d %s" % (n, CURTO[r]) for r, n in SPEC["blocos_s1"]]
ENTRADA_2_EXTRA = "%d %s" % (SPEC["bloco_extra_s2"][1],
                             SPEC["bloco_extra_s2"][0])

CONDICOES = {
    "c1": r"$p_{\mathrm{none}} > \tau_{\mathrm{none}}$",
    "c2": r"$p_{\mathrm{split}} > \tau_{\mathrm{split}}$",
    "c3": r"$p_{\mathrm{rest}} < \tau_{\mathrm{rest}}$",
    "c4": r"$p_{\mathrm{ext}} < \theta_{\mathrm{size}}$",
}
# As consequencias nomeiam as proprias formas em vez de parafrasea-las por
# familia ("the rectangular candidates", "the extended ones"). Encurta as
# frases quase pela metade, usa o vocabulario da Secao II e, sobretudo, torna
# cada consequencia conferivel: o leitor conta as formas que sobraram.
ACOES = {
    "c1": "only NONE is evaluated",
    "c2": "only SPLIT: NONE and\nthe other shapes off",
    "c3": "HORZ and VERT off,\nAB and 4-way with them",
    "c4": "AB and 4-way off",
}


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(p["surface"])

    def seta(x0, y0, x1, y1, cor=None, cabeca=True):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>" if cabeca else "-",
                                    color=cor or p["ink_2"], linewidth=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7.0),
                    zorder=4)

    def linha(pontos, cor=None):
        xs = [q[0] for q in pontos]
        ys = [q[1] for q in pontos]
        ax.plot(xs, ys, color=cor or p["ink_2"], linewidth=0.8, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")

    def losango(chave, cor):
        yc, h = Y[chave]
        ax.add_patch(Polygon([(X_SPINE, yc + h / 2), (X_SPINE + MEIA_LOS, yc),
                              (X_SPINE, yc - h / 2), (X_SPINE - MEIA_LOS, yc)],
                             closed=True, facecolor=p["surface"],
                             edgecolor=cor, linewidth=0.9, zorder=4))
        ax.text(X_SPINE, yc, CONDICOES[chave], ha="center", va="center",
                fontsize=PT, color=p["ink"], zorder=5)
        return yc, h

    def caixa_acao(chave, yc, terminal=False, cor=None):
        """Consequencia do ramo "yes". Terminal (cantos arredondados) marca o
        ramo que ENCERRA a busca deste no: o compromisso com o split e o unico
        que dispensa a avaliacao de NONE, e por isso nao reencontra o fluxo."""
        h = (0.335 if "\n" in ACOES[chave] else 0.215) / FIG_H_IN
        w = X_ACAO_1 - X_ACAO_0
        if terminal:
            ax.add_patch(FancyBboxPatch((X_ACAO_0, yc - h / 2), w, h,
                                        boxstyle="round,pad=0,rounding_size=0.022",
                                        facecolor=p["kept_f"], edgecolor=cor,
                                        linewidth=0.8, zorder=4))
        else:
            ax.add_patch(Rectangle((X_ACAO_0, yc - h / 2), w, h,
                                   facecolor=p["surface"], edgecolor=cor,
                                   linewidth=0.8, zorder=4))
        ax.text((X_ACAO_0 + X_ACAO_1) / 2, yc, ACOES[chave], ha="center",
                va="center", fontsize=PT, color=p["ink"], linespacing=1.25,
                zorder=5)
        return h

    def caixa_processo(chave, texto, cor):
        yc, h = Y[chave]
        ax.add_patch(Rectangle((X_SPINE - MEIA_PROC, yc - h / 2),
                               2 * MEIA_PROC, h, facecolor=p["surface"],
                               edgecolor=cor, linewidth=0.9, zorder=4))
        ax.text(X_SPINE, yc, texto, ha="center", va="center", fontsize=PT,
                color=p["ink"], linespacing=1.2, zorder=5)
        return yc, h

    def fila_entrada(chave, nome, num, cor, blocos):
        """Linha de entrada: identidade do podador a esquerda, blocos do vetor
        de atributos a direita, e a junta que os leva ao primeiro losango."""
        yc, h = Y[chave]
        # Numero e nome na MESMA linha de base. Com o piso de 9 pt, o arranjo
        # anterior — circulo acima, nome abaixo — pedia meia polegada de altura
        # na coluna da esquerda e invadia as folgas das linhas vizinhas.
        ax.plot([X_NOME + 0.020], [yc], marker="o", markersize=13.0,
                color=cor, markeredgecolor="none", zorder=6, clip_on=False)
        ax.text(X_NOME + 0.020, yc, num, ha="center", va="center",
                fontsize=PT, color=p["surface"], weight="bold", zorder=7)
        ax.text(X_NOME + 0.055, yc, nome, ha="left", va="center", fontsize=PT,
                color=cor, weight="bold", zorder=5)

        n = len(blocos)
        folga = 0.012
        w = (X_ENT_1 - X_ENT_0 - folga * (n - 1)) / n
        centros = []
        for i, (rotulo, destaque) in enumerate(blocos):
            x0 = X_ENT_0 + i * (w + folga)
            ax.add_patch(Rectangle((x0, yc - h / 2), w, h,
                                   facecolor=(cor if destaque else p["kept_f"]),
                                   edgecolor=(cor if destaque else p["kept_e"]),
                                   linewidth=0.7, zorder=4))
            ax.text(x0 + w / 2, yc, rotulo, ha="center", va="center",
                    fontsize=PT,
                    color=(p["surface"] if destaque else p["ink"]), zorder=5)
            centros.append(x0 + w / 2)
        return yc, h, centros

    def junta(yc_ent, h_ent, centros, yc_alvo, h_alvo, rotulo):
        """Desce de cada bloco de entrada ate uma barra comum e leva a barra ao
        losango seguinte, com o total de atributos anotado na descida."""
        y_barra = (yc_ent - h_ent / 2 + yc_alvo + h_alvo / 2) / 2
        for cx in centros:
            linha([(cx, yc_ent - h_ent / 2), (cx, y_barra)])
        # A barra tem de alcancar o eixo: os blocos de entrada comecam a
        # direita dele, e sem esta extensao a seta de descida ficaria solta.
        linha([(min(min(centros), X_SPINE), y_barra),
               (max(centros), y_barra)])
        seta(X_SPINE, y_barra, X_SPINE, yc_alvo + h_alvo / 2)
        ax.text(X_SPINE - 0.012, (y_barra + yc_alvo + h_alvo / 2) / 2, rotulo,
                ha="right", va="center", fontsize=PT, color=p["ink_2"],
                zorder=5)

    def ramo_sim(chave, yc, h, cor, terminal=False):
        """Ramo "yes": sai pelo vertice direito do losango para a caixa de
        consequencia. Devolve a altura da caixa, para o trilho."""
        seta(X_SPINE + MEIA_LOS, yc, X_ACAO_0, yc)
        ax.text((X_SPINE + MEIA_LOS + X_ACAO_0) / 2, yc + 0.022, "yes",
                ha="center", va="center", fontsize=PT, color=p["ink_2"],
                zorder=5)
        return caixa_acao(chave, yc, terminal=terminal, cor=cor)

    def ramo_nao(yc, h, y_prox_topo):
        seta(X_SPINE, yc - h / 2, X_SPINE, y_prox_topo)
        # O rotulo fica no terco superior do segmento, e nao no meio: o meio
        # e onde o trilho reencontra o eixo, e os dois colidiriam.
        ax.text(X_SPINE - 0.012,
                yc - h / 2 - 0.34 * (yc - h / 2 - y_prox_topo), "no",
                ha="right", va="center", fontsize=PT, color=p["ink_2"],
                zorder=5)

    def trilho(y_saidas, y_merge, y_alvo_topo):
        """Reencontro dos ramos "yes" que nao encerram o no: um unico trilho a
        direita, descendo ate a folga acima da caixa de destino e voltando ao
        eixo. Tres linhas de retorno separadas custariam mais tinta do que
        informacao."""
        for ys in y_saidas:
            linha([(X_ACAO_1, ys), (X_TRILHO, ys)])
        linha([(X_TRILHO, max(y_saidas)), (X_TRILHO, y_merge)])
        linha([(X_TRILHO, y_merge), (X_SPINE, y_merge)])
        ax.plot([X_SPINE], [y_merge], marker="o", markersize=1.8,
                color=p["ink_2"], zorder=5)

    c1, c2 = p["s1"], p["s2"]

    # ---------------- estagio 1 ---------------------------------------------
    y_e1, h_e1, cx1 = fila_entrada(
        "e1", "pre-search", "1", c1,
        [(r, False) for r in ENTRADAS_1])
    yc1, hc1 = Y["c1"]
    junta(y_e1, h_e1, cx1, yc1, hc1, "36")

    y_saidas = []
    losango("c1", c1)
    h_a1 = ramo_sim("c1", yc1, hc1, c1)
    y_saidas.append(yc1)
    yc2, hc2 = Y["c2"]
    ramo_nao(yc1, hc1, yc2 + hc2 / 2)

    losango("c2", c1)
    ramo_sim("c2", yc2, hc2, c1, terminal=True)   # encerra: NONE nao e avaliado
    yc3, hc3 = Y["c3"]
    ramo_nao(yc2, hc2, yc3 + hc3 / 2)

    losango("c3", c1)
    ramo_sim("c3", yc3, hc3, c1)
    y_saidas.append(yc3)

    y_none, h_none = Y["none"]
    ramo_nao(yc3, hc3, y_none + h_none / 2)
    y_merge = (yc3 - hc3 / 2 + y_none + h_none / 2) / 2
    trilho(y_saidas, y_merge, y_none + h_none / 2)

    caixa_processo("none", "NONE is evaluated: rate,\n"
                           "distortion and RD cost", p["ink_2"])

    # ---------------- estagio 2 ---------------------------------------------
    y_e2, h_e2, cx2 = fila_entrada(
        "e2", "post-NONE", "2", c2,
        [("the same 36", False), (ENTRADA_2_EXTRA, True)])
    seta(X_SPINE, y_none - h_none / 2, X_SPINE, y_e2 + h_e2 / 2)
    yc4, hc4 = Y["c4"]
    junta(y_e2, h_e2, cx2, yc4, hc4, "39")

    losango("c4", c2)
    ramo_sim("c4", yc4, hc4, c2)
    y_fim, h_fim = Y["fim"]
    ramo_nao(yc4, hc4, y_fim + h_fim / 2)
    y_merge2 = (yc4 - hc4 / 2 + y_fim + h_fim / 2) / 2
    trilho([yc4], y_merge2, y_fim + h_fim / 2)

    caixa_processo("fim", "the enabled candidates\n"
                          "are evaluated", p["ink_2"])

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
                          "figura2_fluxograma%s.pdf" % sufixo[nome]),
             PALETAS[nome])
    print("\nDimensoes finais: %.3f x %.3f in "
          "(altura derivada das linhas, nao arbitrada)" % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
