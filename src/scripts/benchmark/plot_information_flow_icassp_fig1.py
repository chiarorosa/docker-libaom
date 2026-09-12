#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 — o fluxo da busca de particionamento intra do
AV1 num no, e o ponto de terminacao antecipada PRE-BUSCA que este trabalho ataca.

Diagrama ESTRUTURAL: nao le artefato numerico, declara em ETAPAS e FAIXAS os
fatos que desenha, e a trava `conferir()` reconfere cada expressao contra o
proprio .tex do artigo, parando se divergirem. A terminologia do artigo e
normativa; a figura se ajusta a ela, nunca o contrario.

Composicao, tres partes sobre UMA UNICA grade de colunas:

  1. CABECALHO — a ordem nativa de avaliacao dentro de um no, da esquerda para a
     direita, com o ponto de insercao do preditor marcado como distintivo
     numerado ANTES da primeira coluna. E a posicao do distintivo, e nao um
     rotulo, que diz que a decisao antecede toda medicao de RD do no.

  2. DUAS FAIXAS DE DISPONIBILIDADE, coluna a coluna, nas MESMAS colunas do
     cabecalho. A de cima, a informacao pre-busca, esta cheia desde antes da
     primeira coluna; a de baixo, a informacao de RD do no, comeca vazia e so se
     preenche nos instantes em que cada custo passa a existir. A assimetria
     vertical entre as duas faixas E o argumento da Secao II: o preditor so pode
     consumir a faixa de cima.

  3. FAIXA DA DECISAO — o alcance da terminacao antecipada, marcado nas mesmas
     colunas: a primeira segue selecionada, as demais sao removidas junto com os
     seus descendentes recursivos. Mostrado por alinhamento, sem seta e sem
     frase: o que a versao anterior dizia em "Skip remaining partition
     alternatives and recursive descendants" aqui se le na coluna.

Por que nao ha caixa nem seta de fluxograma: a versao anterior gastava duas
colunas de pagina com sentencas que repetiam o corpo do artigo. Cada rotulo aqui
carrega o NOME da etapa; o que a etapa faz esta na Secao II.

Duas paletas, MESMA GEOMETRIA, como nos demais geradores do grupo:
  cor   — azul-aco no acento da decisao e no que ela remove;
  cinza — a mesma geometria em luminancia, para impressao monocromatica.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \\
        src/scripts/benchmark/plot_information_flow_icassp_fig1.py \\
        --out-dir results/thesis/figuras

Saidas: figura1_fluxo_informacao.pdf       (variante de cor, a do artigo)
        figura1_fluxo_informacao_cinza.pdf (variante monocromatica)
        e um .png de cada, para conferencia visual.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --- tipografia: identica a da figura 2 do artigo ----------------------------
# O spconf.sty compoe em Times; STIXGeneral e a serif de metrica Times que
# acompanha o matplotlib. fonttype 42 embute como TrueType, pois o Type 3 padrao
# costuma ser recusado pelo PDF eXpress do IEEE.
# O monoespacado tem de ser o MESMO que o \texttt do artigo compoe, que e o
# NimbusMonL do texlive. O pacote so traz o OTF, e o matplotlib embutiria CFF
# declarando TrueType -- mismatch que o PDF eXpress reprova; por isso usa-se a
# conversao para TrueType feita por `otf2ttf_nimbus_mono.py`, com nome proprio.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "font.monospace": ["Nimbus Mono PS TT", "DejaVu Sans Mono"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Largura de coluna exata do spconf.sty (\textwidth 178 mm, \columnsep 6 mm),
# de modo que \includegraphics[width=\columnwidth] fique 1:1 e os corpos
# declarados aqui sejam os corpos medidos na pagina.
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt
H_PT = 122.0
COL_W_IN, FIG_H_IN = W_PT / 72.0, H_PT / 72.0

# Corpos na escala do exemplar do grupo (a figura 2 do LASCAS compoe em 5,2 pt):
# tres colunas em 86 mm nao comportam 8 pt sem o rotulo vazar da celula, e os
# guias do ICASSP nao impoem piso -- exigem fonte embutida e subsetada, nao
# tamanho minimo. Mexer no texto sem reconferir aqui faz o rotulo vazar.
PT = 7.0        # corpo dos rotulos de etapa e de faixa
PT_MIN = 6.2    # corpo das notas, dos tokens \texttt e da legenda
PT_MONO = 5.8   # tokens em Courier, mais largos que a serif no mesmo corpo
PT_BADGE = 5.0

# --- as duas paletas ---------------------------------------------------------
PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#c9c7be",
        acento="#25599f",
        tem_f="#dcdad2", tem_e="#b9b7ae",      # informacao ja disponivel
        nao_f="#f4f3ef", nao_e="#d8d6cd",      # ainda nao produzida
        cut_f="#8ba5cd", cut_e="#25599f",      # removido pela decisao
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#bcbab3",
        acento="#141413",
        tem_f="#e3e1db", tem_e="#b5b3ac",
        nao_f="#f5f4f1", nao_e="#d6d4cd",
        cut_f="#6b6a65", cut_e="#3f3e3b",
    ),
}

# --- os fatos desenhados, com procedencia ------------------------------------
# Cada rotulo ocorre no corpo do artigo. A Secao II fixa a ordem de avaliacao e o
# instante em que cada custo passa a existir; a Secao IV-A fixa a contagem de
# candidatos e o fato de PARTITION_SPLIT ser representado pela avaliacao
# recursiva dos seus filhos.
# (texto da linha, e se a linha e monoespacada como o \texttt da pagina)
ETAPAS = [
    [("PARTITION_NONE", True)],
    [("PARTITION_SPLIT", True), ("and descendants", False)],
    [("remaining partition", False), ("candidates", False)],
]

# Faixas de disponibilidade. `celulas` diz, por coluna, se a informacao ja
# existe naquele ponto da busca.
FAIXAS = [
    dict(rot=["pre-search", "information"], celulas=[1, 1, 1],
         nota="source/block information, coding parameters,"
              " causal-neighbor partitions",
         marcas=[]),
    dict(rot=["current-node RD", "information"], celulas=[0, 1, 1],
         nota=None,
         marcas=[(1, r"$J_{\mathrm{none}}$"), (2, r"$J^{*}$")]),
]

# Alcance da decisao: 1 segue no fluxo, 0 e removida com os seus descendentes.
DECISAO = dict(rot="terminate as", mono_rot="PARTITION_NONE",
               celulas=[1, 0, 0])

# --- grade horizontal, compartilhada pelo cabecalho e pelas faixas -----------
X_ROT = 3.0                 # inicio da coluna de rotulos, a esquerda
X_GRADE = 56.0              # inicio da regiao de colunas
X_BADGE = 50.5              # distintivo, entre o rotulo e a primeira coluna
GAP = 4.0
COLS = []
_larg = (W_PT - X_GRADE - 2 * GAP - 1.0) / 3.0
for _i in range(3):
    _x0 = X_GRADE + _i * (_larg + GAP)
    COLS.append((_x0, _x0 + _larg))

Y_SPINE = 114.0             # eixo da ordem de busca
Y_CAB = 92.0               # base das caixas de etapa
H_CAB = 17.0
H_FAIXA = 10.0
Y_FAIXA = [68.0, 44.0]      # base de cada faixa de disponibilidade
Y_DEC = 20.0                # base da faixa da decisao
Y_LEG = 6.5


def conferir(tex_path):
    """Trava de vocabulario contra o .tex do artigo.

    A figura nao pode introduzir termo que o artigo nao use. Uma divergencia
    significa que o texto mudou e que a figura ficou para tras."""
    if not os.path.exists(tex_path):
        print("  aviso: %s nao encontrado, trava de vocabulario nao rodou"
              % tex_path)
        return
    with open(tex_path, encoding="utf-8") as f:
        tex = f.read()
    plano = tex.replace("\\_", "_").replace("\\mathrm", "")
    plano = re.sub(r"\\[a-zA-Z]+", " ", plano)
    plano = re.sub(r"[{}$]", "", plano)
    plano = re.sub(r"\s+", " ", plano).lower()
    exigidos = ["pre-search", "source/block information", "coding parameters",
                "causal neighbor", "current-node rd search", "partition_none",
                "partition_split", "partition candidates",
                "recursive descendants", "j_none", "j^*"]
    faltam = [e for e in exigidos if e not in plano]
    if faltam:
        sys.stderr.write("Figura 1 usa termo ausente do artigo:\n  "
                         + "\n  ".join(faltam)
                         + "\nAjuste a figura ao vocabulario do .tex.\n")
        raise SystemExit(1)
    print("  trava de vocabulario: %d expressoes conferidas" % len(exigidos))


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H_PT)
    ax.axis("off")
    ax.set_facecolor(p["surface"])

    def txt(x, y, s, size=PT, cor=None, ha="left", va="center",
            family="serif", peso="normal", z=5):
        ax.text(x, y, s, fontsize=size, color=cor or p["ink"], ha=ha, va=va,
                family=family, fontweight=peso, zorder=z)

    def celula(x0, x1, y0, h, face, edge, lw=0.5, z=3):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=face,
                               edgecolor=edge, linewidth=lw, zorder=z))

    def distintivo(x, y):
        ax.plot([x], [y], marker="o", markersize=7.4, color=p["acento"],
                markeredgecolor="none", zorder=6, clip_on=False)
        txt(x, y, "1", size=PT_BADGE, cor=p["surface"], ha="center",
            peso="bold", z=7)

    # ---- cabecalho: ordem nativa de avaliacao dentro de um no --------------
    txt(X_ROT, Y_SPINE + 4.0, "search order", size=PT_MIN, cor=p["ink_2"])
    txt(X_ROT, Y_SPINE - 4.0, "within a node", size=PT_MIN, cor=p["ink_2"])
    ax.annotate("", xy=(W_PT - 1.0, Y_SPINE), xytext=(X_BADGE + 5.5, Y_SPINE),
                arrowprops=dict(arrowstyle="-|>", color=p["muted"],
                                linewidth=0.6, shrinkA=0, shrinkB=0,
                                mutation_scale=5), zorder=2)
    # O distintivo vem ANTES da primeira coluna: e a posicao, e nao um rotulo,
    # que diz que a decisao antecede toda medicao de RD do no.
    distintivo(X_BADGE, Y_SPINE)

    for (x0, x1), linhas in zip(COLS, ETAPAS):
        celula(x0, x1, Y_CAB, H_CAB, p["surface"], p["regua"], lw=0.6, z=3)
        n = len(linhas)
        for i, (s, mono) in enumerate(linhas):
            yy = Y_CAB + H_CAB / 2 + (n - 1) * 4.8 / 2 - i * 4.8
            txt((x0 + x1) / 2, yy, s, size=PT_MONO if mono else PT_MIN,
                ha="center", family="monospace" if mono else "serif")

    # ---- faixas de disponibilidade ----------------------------------------
    for faixa, y in zip(FAIXAS, Y_FAIXA):
        for i, r in enumerate(faixa["rot"]):
            txt(X_ROT, y + H_FAIXA / 2 + (4.0 if i == 0 else -4.0), r)
        # A faixa pre-busca comeca ANTES da primeira coluna: e o que mostra que
        # ela ja existe no instante do distintivo.
        if all(faixa["celulas"]):
            celula(X_BADGE - 3.5, COLS[0][0], y, H_FAIXA, p["tem_f"],
                   p["tem_e"])
        for (x0, x1), tem in zip(COLS, faixa["celulas"]):
            celula(x0, x1, y, H_FAIXA,
                   p["tem_f"] if tem else p["nao_f"],
                   p["tem_e"] if tem else p["nao_e"])
        for col, simbolo in faixa["marcas"]:
            x0, x1 = COLS[col]
            txt((x0 + x1) / 2, y + H_FAIXA / 2, simbolo, ha="center")
        if faixa["nota"]:
            txt(X_ROT, y - 5.0, faixa["nota"], size=PT_MIN, cor=p["muted"])

    # ---- faixa da decisao --------------------------------------------------
    txt(X_ROT, Y_DEC + H_FAIXA / 2 + 4.0, DECISAO["rot"])
    txt(X_ROT, Y_DEC + H_FAIXA / 2 - 4.0, DECISAO["mono_rot"], size=PT_MONO,
        family="monospace")
    for (x0, x1), mantida in zip(COLS, DECISAO["celulas"]):
        celula(x0, x1, Y_DEC, H_FAIXA,
               p["tem_f"] if mantida else p["cut_f"],
               p["tem_e"] if mantida else p["cut_e"])
        txt((x0 + x1) / 2, Y_DEC + H_FAIXA / 2,
            "selected" if mantida else "removed", size=PT_MIN, ha="center",
            cor=p["ink_2"] if mantida else p["surface"])

    # ---- legenda -----------------------------------------------------------
    distintivo(X_ROT + 3.2, Y_LEG)
    txt(X_ROT + 9.5, Y_LEG,
        "pre-search early-termination point considered in this work",
        size=PT_MIN, cor=p["ink_2"])

    fig.savefig(out_path, facecolor=p["surface"])
    fig.savefig(out_path.replace(".pdf", ".png"), facecolor=p["surface"],
                dpi=400)
    plt.close(fig)
    print("  gravado: %s  (%.3f x %.2f in)" % (out_path, COL_W_IN, FIG_H_IN))


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/workspace/results/thesis/figuras")
    ap.add_argument("--tex", default="/workspace/results/thesis/"
                                     "IEEE_Conference_Template/ICASSP/"
                                     "paper_daniel.tex")
    ap.add_argument("--variante", choices=sorted(PALETAS) + ["ambas"],
                    default="ambas")
    args = ap.parse_args(argv)
    conferir(args.tex)
    os.makedirs(args.out_dir, exist_ok=True)
    nomes = sorted(PALETAS) if args.variante == "ambas" else [args.variante]
    for nome in nomes:
        sufixo = "" if nome == "cor" else "_cinza"
        draw(os.path.join(args.out_dir,
                          "figura1_fluxo_informacao%s.pdf" % sufixo),
             PALETAS[nome])


if __name__ == "__main__":
    main(sys.argv[1:])
