#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 - o no de decisao do particionamento intra do
AV1: o que existe antes da busca, a decisao antecipada, e o que a busca gera.

Diagrama ESTRUTURAL, como a figura 2 do LASCAS: nao le artefato numerico, e sim
declara em blocos os fatos que desenha, cada um com a sua procedencia no .tex, de
modo que uma mudanca de terminologia no artigo obrigue a mexer aqui e nao passe
despercebida. A trava `conferir()` reconfere o vocabulario contra o proprio
`paper.tex` e para se divergirem.

Tres decisoes de composicao, nesta ordem de importancia:

  ECONOMIA DE TEXTO. Cada caixa carrega o NOME da etapa, nao a sua descricao: o
  que a etapa faz esta no corpo do artigo, e repeti-lo na figura so competia com
  ele. Onde o artigo tem simbolo, a figura usa o simbolo - $J_\\mathrm{none}$ e
  $J^{*}$, da equacao (1) -, que e o que permite marcar o instante em que cada
  custo passa a existir sem gastar uma frase.

  VOCABULARIO DO ARTIGO. Todo rotulo e expressao que ja ocorre no corpo:
  "pre-search information", "current-node RD search", "partition candidates",
  "best partition", os tokens \\texttt e os dois simbolos de custo. A figura nao
  introduz termo proprio - "rectangular" e "extended", por exemplo, saem porque
  nao aparecem no .tex.

  ESCALA DE CINZA. Sem matiz: o desenho e estrutural, e a unica hierarquia que
  precisa marcar - a caixa da decisao contra as demais - se resolve por peso de
  traco e por preenchimento. Assim ele sobrevive a impressao monocromatica sem
  variante separada e sem depender de cor para significar.

Coluna simples (86 mm), proporcao 1,3:1. A versao anterior espalhava o mesmo
conteudo em `figure*` de 178 mm: ocupava 4,5 polegadas-coluna de pagina contra
as 2,6 desta, e a horizontalidade vinha das frases dentro das caixas, nao da
informacao.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \\
        src/scripts/benchmark/plot_information_flow_icassp_fig1.py \\
        --out-dir results/thesis/figuras

Saida: figura1_fluxo_informacao.pdf, e um .png para conferencia visual.
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

# --- tipografia: identica a da figura 2 do artigo ----------------------------
# O spconf.sty compoe em Times; STIXGeneral e a serif de metrica Times que
# acompanha o matplotlib. fonttype 42 embute como TrueType, pois o Type 3 padrao
# costuma ser recusado pelo PDF eXpress do IEEE.
# O monoespacado tem de ser o MESMO que o \texttt do artigo compoe, que e o
# NimbusMonL (clone do Courier) do texlive. O pacote fonts-urw-base35 so traz o
# OTF, e o matplotlib embute CFF declarando TrueType - mismatch que o PDF
# eXpress reprova; por isso a fonte usada aqui e a conversao para TrueType feita
# por `otf2ttf_nimbus_mono.py`, com nome proprio. Sem ela, a lista degrada para
# o DejaVu Sans Mono, que embute limpo mas nao casa com o \texttt da pagina.
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
# de modo que \includegraphics[width=\columnwidth] fique em escala 1:1 e os
# corpos declarados aqui sejam os corpos medidos na pagina.
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt PostScript
COL_W_IN = W_PT / 72.0

PT = 8.5        # corpo das caixas
LINHA = 10.0    # entrelinha
PADB = 3.5      # respiro interno das caixas
PADG = 5.0      # respiro interno do grupo tracejado
MARGEM = 1.5
SETA = 8.0      # comprimento das setas entre etapas
SETA_RAMO = 15.0  # a bifurcacao, que precisa de espaco para o cotovelo

# --- as colunas do desenho, em pontos ----------------------------------------
# Somam W_PT. Cada largura foi fixada pela linha mais longa que a coluna carrega,
# medida por `caber()` na hora de desenhar: mexer no texto sem mexer aqui para a
# execucao em vez de produzir um PDF com texto vazando.
W_SIM = 94.0            # o desfecho do ramo "yes"
W_GRUPO = 104.0         # a busca do no, com as suas etapas
W_MARCA = 30.0          # a coluna dos simbolos de custo
GAP_SIM_GRP = 7.0
GAP_GRP_MARCA = 5.0
W_DECISAO = 132.0

# --- os fatos desenhados, com procedencia ------------------------------------
# Todo rotulo abaixo ocorre no proprio paper.tex. A terminologia do artigo e
# normativa: esta figura se ajusta a ela, nunca o contrario. Convencao de linha:
# uma lista de trechos (texto, familia), para que um token em Courier possa
# dividir a linha com o texto em Times, como no \texttt da pagina.
TX = "monospace"
SF = "serif"

ENTRADA = {
    "titulo": "Pre-search information",
    "linhas": [[("source/block information, coding parameters,", SF)],
               [("partitions of causal neighbors", SF)]],
}
DECISAO = [("Terminate as ", SF), ("PARTITION_NONE", TX), ("?", SF)]
SIM = [[("PARTITION_NONE", TX)],
       [("selected; candidates", SF)],
       [("and descendants skipped", SF)]]
GRUPO_TITULO = "Current-node RD search"
# (linha da etapa, simbolo do custo que passa a existir nela)
ETAPAS = [
    ([("PARTITION_NONE", TX)], "$J_\\mathrm{none}$"),
    ([("PARTITION_SPLIT", TX)], None),
    ([("remaining candidates", SF)], None),
    ([("best partition", SF)], "$J^{*}$"),
]
RAMOS = {"sim": "yes", "nao": "no"}

# --- paleta acromatica -------------------------------------------------------
# Sem matiz. A decisao se distingue por traco mais grosso e fundo preenchido; o
# grupo da busca, por borda tracejada; as marcas de custo, por corpo em italico
# matematico. Nada aqui significa por cor.
P = dict(
    surface="#ffffff",
    ink="#0b0b0b",          # texto das caixas
    ink_2="#3f3e3b",        # rotulos de ramo e simbolos
    borda="#4a4945",        # borda das caixas de acao
    forte="#141413",        # borda da decisao
    fundo="#e6e4df",        # fundo da decisao
    tracejado="#8c8a84",    # grupo e conectores
)


def conferir(tex_path):
    """Trava de vocabulario contra o paper.tex.

    A figura nao pode introduzir termo que o artigo nao use. Cada expressao
    abaixo e desenhada e tem de aparecer no .tex; uma divergencia significa que
    o texto mudou e que a figura ficou para tras."""
    if not os.path.exists(tex_path):
        print("  aviso: %s nao encontrado, trava de vocabulario nao rodou"
              % tex_path)
        return
    with open(tex_path, encoding="utf-8") as f:
        tex = f.read()
    # Normaliza a marcacao para comparar so o texto: "\texttt{PARTITION\_NONE}"
    # vira "PARTITION_NONE" e "$J_{\mathrm{none}}$" vira "J_none".
    plano = tex.replace("\\_", "_").replace("\\mathrm", "")
    plano = re.sub(r"\\[a-zA-Z]+", " ", plano)
    plano = re.sub(r"[{}$]", "", plano)
    plano = re.sub(r"\s+", " ", plano)
    exigidos = [
        "pre-search information",
        "source/block information",
        "coding parameters",
        "causal neighbor",
        "current-node RD search",
        "PARTITION_NONE",
        "PARTITION_SPLIT",
        "partition candidates",
        "recursive descendants",
        "best partition",
        "J_none",
        "J^*",
    ]
    faltando = [e for e in exigidos if e.lower() not in plano.lower()]
    if faltando:
        sys.stderr.write(
            "Figura 1 e paper.tex divergem; ausentes no .tex:\n  "
            + "\n  ".join(faltando)
            + "\nAtualize a figura para a terminologia do artigo.\n")
        raise SystemExit(1)
    print("  trava de vocabulario: as %d expressoes batem com o paper.tex"
          % len(exigidos))


def altura(n_linhas, pad=PADB):
    return n_linhas * LINHA + 2 * pad


class Tela(object):
    """Eixo unico em pontos PostScript, origem no canto inferior esquerdo."""

    def __init__(self, altura_pt):
        self.h = altura_pt
        self.fig = plt.figure(figsize=(COL_W_IN, altura_pt / 72.0), dpi=400)
        self.fig.patch.set_facecolor(P["surface"])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W_PT)
        self.ax.set_ylim(0, altura_pt)
        self.ax.set_axis_off()
        self.ax.set_facecolor(P["surface"])
        self.fig.canvas.draw()
        self.rend = self.fig.canvas.get_renderer()

    # -- medicao ------------------------------------------------------------
    def largura(self, s, **kw):
        t = self.ax.text(0, 0, s, alpha=0, **kw)
        bb = t.get_window_extent(self.rend)
        t.remove()
        return bb.width * 72.0 / self.fig.dpi

    def caber(self, linha, limite, onde, peso="normal"):
        """Trava de transbordo: nenhum texto pode estourar a sua caixa."""
        w = sum(self.largura(s, family=f, fontsize=PT, fontweight=peso)
                for s, f in linha)
        if w > limite:
            sys.stderr.write(
                "Texto estoura a caixa em %s: %.1f pt para %.1f pt disponiveis"
                "\n  %r\n" % (onde, w, limite, "".join(s for s, _f in linha)))
            raise SystemExit(1)
        return w

    # -- primitivas ---------------------------------------------------------
    def caixa(self, x0, y0, x1, y1, borda, fill, lw=0.5, dash=None, r=1.8,
              zorder=2):
        pat = FancyBboxPatch(
            (x0 + r, y0 + r), x1 - x0 - 2 * r, y1 - y0 - 2 * r,
            boxstyle="round,pad=%f,rounding_size=%f" % (r, r),
            linewidth=lw, edgecolor=borda, facecolor=fill, zorder=zorder)
        if dash is not None:
            pat.set_linestyle((0, dash))
        self.ax.add_patch(pat)

    def texto(self, x, y, s, ha="center", cor=None, familia=None,
              peso="normal", size=PT):
        kw = {"family": familia} if familia else {}
        self.ax.text(x, y, s, ha=ha, va="baseline", fontsize=size,
                     color=cor or P["ink"], fontweight=peso, zorder=5, **kw)

    def linha(self, x, y, partes, ha="center", cor=None, peso="normal"):
        """Uma linha com trechos em Times e trechos em Courier."""
        larguras = [self.largura(s, family=f, fontsize=PT, fontweight=peso)
                    for s, f in partes]
        total = sum(larguras)
        cur = {"left": x, "right": x - total, "center": x - total / 2.0}[ha]
        for (s, f), w in zip(partes, larguras):
            self.texto(cur, y, s, ha="left", cor=cor, familia=f, peso=peso)
            cur += w
        return total

    def bloco(self, xc, y_topo, linhas, cor=None):
        y = y_topo - PADB - LINHA + 2.6
        for ln in linhas:
            self.linha(xc, y, ln, ha="center", cor=cor)
            y -= LINHA

    def seta(self, x0, y0, x1, y1, cor, lw=0.6, tracejada=False, cabeca=True):
        pat = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>" if cabeca else "-",
            mutation_scale=4.5, linewidth=lw, color=cor,
            shrinkA=0, shrinkB=0, zorder=4)
        if tracejada:
            pat.set_linestyle((0, (1.8, 1.6)))
        self.ax.add_patch(pat)


def desenhar(out_path):
    # --- alturas, somadas antes de abrir a figura ---------------------------
    h_entrada = altura(1 + len(ENTRADA["linhas"]))
    h_decisao = altura(1)
    h_sim = altura(len(SIM))
    h_etapa = altura(1)
    h_tit_grupo = LINHA + 1.0
    h_ramo = (h_tit_grupo + len(ETAPAS) * h_etapa
              + (len(ETAPAS) - 1) * SETA + 2 * PADG)

    H = (MARGEM + h_entrada + SETA + h_decisao + SETA_RAMO + h_ramo + MARGEM)

    t = Tela(H)
    x0, x1 = MARGEM, W_PT - MARGEM
    xc = (x0 + x1) / 2.0

    # --- 1. o que ja existe antes da busca ----------------------------------
    y_top = H - MARGEM
    t.caixa(x0, y_top - h_entrada, x1, y_top, P["borda"], P["surface"])
    t.caber([(ENTRADA["titulo"], SF)], x1 - x0 - 2 * PADB, "entrada",
            peso="bold")
    t.texto(xc, y_top - PADB - LINHA + 2.6, ENTRADA["titulo"], cor=P["forte"],
            peso="bold")
    y = y_top - PADB - 2 * LINHA + 2.6
    for ln in ENTRADA["linhas"]:
        t.caber(ln, x1 - x0 - 2 * PADB, "entrada")
        t.linha(xc, y, ln, ha="center")
        y -= LINHA

    # --- 2. a decisao antecipada --------------------------------------------
    y_dec_top = y_top - h_entrada - SETA
    t.seta(xc, y_dec_top + SETA, xc, y_dec_top, P["ink_2"])
    xd0, xd1 = xc - W_DECISAO / 2.0, xc + W_DECISAO / 2.0
    t.caixa(xd0, y_dec_top - h_decisao, xd1, y_dec_top, P["forte"], P["fundo"],
            lw=0.9)
    t.caber(DECISAO, W_DECISAO - 2 * PADB, "decisao")
    t.linha(xc, y_dec_top - PADB - LINHA + 2.6, DECISAO, ha="center")

    # --- 3. os dois ramos ---------------------------------------------------
    y_ramo = y_dec_top - h_decisao - SETA_RAMO
    x_sim0 = x0
    x_grp0 = x_sim0 + W_SIM + GAP_SIM_GRP
    x_grp1 = x_grp0 + W_GRUPO
    x_marca = x_grp1 + GAP_GRP_MARCA
    assert abs((x_marca + W_MARCA + MARGEM) - W_PT) < 1.5, \
        "as colunas nao somam a largura da coluna do artigo"
    xs_c = x_sim0 + W_SIM / 2.0
    xg_c = (x_grp0 + x_grp1) / 2.0

    # A bifurcacao e ortogonal, e nao diagonal: um tronco curto, uma travessa,
    # e uma descida por ramo. E o roteamento que nao cruza rotulo nenhum.
    y_b = y_dec_top - h_decisao
    y_trave = y_b - SETA_RAMO * 0.45
    t.seta(xc, y_b, xc, y_trave, P["ink_2"], cabeca=False)
    t.seta(xs_c, y_trave, xg_c, y_trave, P["ink_2"], cabeca=False)
    # O desfecho do "yes" e uma caixa so, muito mais baixa do que o grupo do
    # "no"; centrada contra ele, o desenho fica equilibrado, e alinhada pelo
    # topo sobraria um vazio de meia figura no canto inferior esquerdo.
    y_sim_top = y_ramo - (h_ramo - h_sim) / 2.0
    t.seta(xs_c, y_trave, xs_c, y_sim_top, P["ink_2"])
    t.seta(xg_c, y_trave, xg_c, y_ramo, P["ink_2"])
    t.texto(xs_c + 2.5, y_trave + 2.4, RAMOS["sim"], ha="left", cor=P["ink_2"])
    t.texto(xg_c - 2.5, y_trave + 2.4, RAMOS["nao"], ha="right", cor=P["ink_2"])

    # ramo "yes": o no termina, e a figura nao precisa dizer mais nada
    t.caixa(x_sim0, y_sim_top - h_sim, x_sim0 + W_SIM, y_sim_top, P["borda"],
            P["surface"])
    for ln in SIM:
        t.caber(ln, W_SIM - 2 * PADB, "ramo yes")
    t.bloco(xs_c, y_sim_top, SIM)

    # ramo "no": a busca do no, etapa a etapa, com o custo que cada uma produz.
    # O rotulo do grupo vive DENTRO do tracejado, para que a seta do ramo pouse
    # na borda da caixa e nao atravesse texto.
    t.caixa(x_grp0, y_ramo - h_ramo, x_grp1, y_ramo, P["tracejado"],
            P["surface"], lw=0.6, dash=(2.6, 2.0), zorder=1)
    t.caber([(GRUPO_TITULO, SF)], W_GRUPO - 2 * PADG, "titulo do grupo",
            peso="bold")
    t.texto(xg_c, y_ramo - PADG - LINHA + 3.0, GRUPO_TITULO, cor=P["forte"],
            peso="bold")
    y = y_ramo - PADG - h_tit_grupo
    for i, (ln, marca) in enumerate(ETAPAS):
        t.caber(ln, W_GRUPO - 2 * PADG - 2 * PADB, "etapa %d" % (i + 1))
        t.caixa(x_grp0 + PADG, y - h_etapa, x_grp1 - PADG, y, P["borda"],
                P["surface"], zorder=3)
        t.linha(xg_c, y - PADB - LINHA + 2.6, ln, ha="center")
        if marca is not None:
            yc = y - h_etapa / 2.0
            t.seta(x_grp1 - PADG, yc, x_marca, yc, P["tracejado"],
                   tracejada=True)
            t.texto(x_marca + 2.0, yc - 2.9, marca, ha="left", cor=P["ink_2"])
        y -= h_etapa
        if i < len(ETAPAS) - 1:
            t.seta(xg_c, y, xg_c, y - SETA, P["ink_2"])
            y -= SETA

    t.fig.savefig(out_path, facecolor=P["surface"])
    t.fig.savefig(out_path[:-4] + ".png", facecolor=P["surface"], dpi=400)
    plt.close(t.fig)
    print("  gravado: %s  (%.3f x %.2f in)" % (out_path, COL_W_IN, H / 72.0))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--tex", default=os.path.join(
        root, "results", "thesis", "IEEE_Conference_Template", "ICASSP",
        "paper.tex"))
    ap.add_argument("--out-dir", default=os.path.join(
        root, "results", "thesis", "figuras"))
    args = ap.parse_args()

    conferir(args.tex)
    os.makedirs(args.out_dir, exist_ok=True)
    desenhar(os.path.join(args.out_dir, "figura1_fluxo_informacao.pdf"))


if __name__ == "__main__":
    main()
