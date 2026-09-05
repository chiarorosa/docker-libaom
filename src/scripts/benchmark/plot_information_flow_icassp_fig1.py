#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 - fluxo da busca de particionamento intra do
AV1 num no, com o ponto de decisao antecipada e a informacao disponivel em cada
instante.

Diagrama ESTRUTURAL, como a figura 2 do LASCAS: nao le artefato numerico, e sim
declara em blocos os fatos que desenha, cada um com a sua procedencia no .tex, de
modo que uma mudanca de terminologia no artigo obrigue a mexer aqui e nao passe
despercebida. A trava `conferir()` reconfere o vocabulario contra o proprio
`paper.tex` e para se divergirem.

TOPOLOGIA: e a do rascunho em TikZ que originou a figura, preservada caixa por
caixa - entrada a esquerda, decisao ao centro, ramo "Yes" descendo em duas
acoes, ramo "No" a direita com os quatro estagios nativos dentro de um retangulo
tracejado, e as duas notas laterais tracejadas ligadas aos estagios em que uma
grandeza de custo passa a existir. O que muda em relacao ao rascunho e a
execucao: tipografia do proprio artigo (Times no corpo, Courier nos tokens
\\texttt), paleta sobria do grupo e largura declarada em pontos.

Por que DUAS COLUNAS (`figure*`, `width=\\textwidth`): o desenho e largo por
natureza - sao tres colunas de conteudo lado a lado mais a coluna de notas.
Espremido em 86 mm ele so caberia empilhado e com cerca de 3,9 in de altura,
45% da coluna; em 178 mm cabe em 2,7 in, com a mesma area de pagina e a leitura
da esquerda para a direita que o fluxo pede.

Duas paletas, MESMA GEOMETRIA, como nos demais geradores:
  cor   - o azul-aco #25599f dos acentos das figuras do grupo, no lugar do azul
          do rascunho;
  cinza - o mesmo desenho em luminancia, para impressao monocromatica.

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

# Largura de texto exata do spconf.sty (\textwidth 178 mm), de modo que
# \includegraphics[width=\textwidth] fique em escala 1:1 e os corpos declarados
# aqui sejam os corpos medidos na pagina.
W_PT = 178.0 / 25.4 * 72.0         # 504,57 pt PostScript
FIG_W_IN = W_PT / 72.0

PT = 8.5        # corpo das caixas
PT_TT = 8.5     # o mesmo corpo, na fonte do \texttt da pagina
LINHA = 10.0    # entrelinha
PADB = 5.0      # respiro interno das caixas
PADG = 6.0      # respiro interno do grupo tracejado
MARGEM = 1.5

# --- as colunas do desenho, em pontos ----------------------------------------
# Somam W_PT. Cada largura foi fixada pela linha mais longa que a coluna carrega,
# medida com `caber()` na hora de desenhar: mexer no texto sem mexer aqui para a
# execucao em vez de produzir um PDF com texto vazando.
W_ENTRADA = 122.0
W_DECISAO = 100.0
W_GRUPO = 143.0
W_NOTA = 98.5
GAP_ENT_DEC = 14.0      # a seta da entrada para a decisao
GAP_DEC_GRP = 14.0      # a seta "No", da decisao para o primeiro estagio
GAP_GRP_NOTA = 10.0     # os tracejados que ligam estagio e nota
W_SIM = 110.0           # as acoes do ramo "Yes", mais largas que a decisao

# --- os fatos desenhados, com procedencia ------------------------------------
# Todo texto abaixo vem do proprio paper.tex, Secao II ("AV1 Intra Partition
# Search and Early-Termination Point"). A terminologia do artigo e normativa:
# esta figura se ajusta a ela, nunca o contrario.
#
# Convencao de linha: uma lista de trechos (texto, familia), para que um token
# em Courier possa dividir a linha com o texto em Times, como no \texttt.
TX = "monospace"
SF = "serif"

ENTRADA = {
    "titulo": ["Available before the", "current-node RD search"],
    "itens": [["Source/block information"],
              ["Coding parameters"],
              ["Previously coded causal-", "neighbor partitions"]],
}
DECISAO = {
    "titulo": "Early decision",
    "linhas": [[("Terminate as", SF)], [("PARTITION_NONE?", TX)]],
}
SIM = [
    [[("Select", SF)], [("PARTITION_NONE", TX)]],
    [[("Skip remaining partition", SF)],
     [("alternatives and their", SF)],
     [("recursive descendants", SF)]],
]
GRUPO_TITULO = "Generated during the current-node search"
ESTAGIOS = [
    ([[("Evaluate ", SF), ("PARTITION_NONE", TX)]],
     [[("$J_\\mathrm{none}$ becomes", SF)], [("available here", SF)]]),
    ([[("Evaluate ", SF), ("PARTITION_SPLIT", TX)],
      [("and recurse into children", SF)]], None),
    ([[("Evaluate rectangular and", SF)], [("extended partitions", SF)]], None),
    ([[("Select best partition", SF)]],
     [[("Best-partition RD cost", SF)], [("becomes available only", SF)],
      [("after the full search", SF)]]),
]
RAMOS = {"sim": "Yes", "nao": "No"}

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        borda="#4a4945", acento="#25599f", acento_fill="#eef3fa",
        acento_leve="#7d9cc4", nota_borda="#6e6d68",
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        borda="#3f3e3b", acento="#141413", acento_fill="#eeece7",
        acento_leve="#8c8a84", nota_borda="#6e6d68",
    ),
}


def conferir(tex_path):
    """Trava de vocabulario contra o paper.tex.

    A figura nao pode introduzir termo que o artigo nao use. Cada expressao
    abaixo tem de aparecer no .tex; uma divergencia significa que a Secao II
    mudou e que a figura ficou para tras."""
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
        "current-node RD search",
        "source/block information",
        "coding parameters",
        "causal neighboring blocks",
        "PARTITION_NONE",
        "PARTITION_SPLIT",
        "recursive descendants",
        "J_none",
        "the RD cost of the best partition",
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

    def __init__(self, altura_pt, p):
        self.h = altura_pt
        self.p = p
        self.fig = plt.figure(figsize=(FIG_W_IN, altura_pt / 72.0), dpi=400)
        self.fig.patch.set_facecolor(p["surface"])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W_PT)
        self.ax.set_ylim(0, altura_pt)
        self.ax.set_axis_off()
        self.ax.set_facecolor(p["surface"])
        self.fig.canvas.draw()
        self.rend = self.fig.canvas.get_renderer()

    # -- medicao ------------------------------------------------------------
    def largura(self, s, **kw):
        t = self.ax.text(0, 0, s, alpha=0, **kw)
        bb = t.get_window_extent(self.rend)
        t.remove()
        return bb.width * 72.0 / self.fig.dpi

    def largura_linha(self, linha, size=PT, peso="normal"):
        return sum(self.largura(s, family=f, fontsize=size, fontweight=peso)
                   for s, f in linha)

    def caber(self, linha, limite, onde, size=PT, peso="normal"):
        """Trava de transbordo: nenhum texto pode estourar a sua caixa.

        Sem ela, um ajuste de vocabulario passaria despercebido no .pdf e so
        apareceria na prova impressa."""
        w = self.largura_linha(linha, size, peso)
        if w > limite:
            sys.stderr.write(
                "Texto estoura a caixa em %s: %.1f pt para %.1f pt disponiveis"
                "\n  %r\n" % (onde, w, limite, "".join(s for s, _f in linha)))
            raise SystemExit(1)
        return w

    # -- primitivas ---------------------------------------------------------
    def caixa(self, x0, y0, x1, y1, borda, fill, lw=0.6, dash=None, r=2.6,
              zorder=2):
        pat = FancyBboxPatch(
            (x0 + r, y0 + r), x1 - x0 - 2 * r, y1 - y0 - 2 * r,
            boxstyle="round,pad=%f,rounding_size=%f" % (r, r),
            linewidth=lw, edgecolor=borda, facecolor=fill, zorder=zorder)
        if dash is not None:
            pat.set_linestyle((0, dash))
        self.ax.add_patch(pat)

    def texto(self, x, y, s, ha="center", size=PT, cor=None, familia=None,
              peso="normal"):
        kw = {"family": familia} if familia else {}
        self.ax.text(x, y, s, ha=ha, va="baseline", fontsize=size,
                     color=cor or self.p["ink"], fontweight=peso, zorder=5,
                     **kw)

    def linha(self, x, y, partes, ha="center", size=PT, cor=None,
              peso="normal"):
        """Uma linha com trechos em Times e trechos em Courier."""
        larguras = [self.largura(s, family=f, fontsize=size, fontweight=peso)
                    for s, f in partes]
        total = sum(larguras)
        cur = {"left": x, "right": x - total,
               "center": x - total / 2.0}[ha]
        for (s, f), w in zip(partes, larguras):
            self.texto(cur, y, s, ha="left", size=size, cor=cor, familia=f,
                       peso=peso)
            cur += w
        return total

    def bloco(self, xc, y_topo, linhas, size=PT, cor=None, peso="normal"):
        """Linhas centradas, do topo para baixo. Devolve a base da ultima."""
        y = y_topo - LINHA + 2.6
        for ln in linhas:
            self.linha(xc, y, ln, ha="center", size=size, cor=cor, peso=peso)
            y -= LINHA
        return y + LINHA

    def seta(self, x0, y0, x1, y1, cor, lw=0.7, tracejada=False):
        pat = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-" if tracejada else "-|>",
            mutation_scale=5.0, linewidth=lw, color=cor,
            shrinkA=0, shrinkB=0, zorder=4)
        if tracejada:
            pat.set_linestyle((0, (2.2, 1.8)))
        self.ax.add_patch(pat)


def desenhar(out_path, p):
    # --- alturas, somadas antes de abrir a figura ---------------------------
    h_entrada = altura(len(ENTRADA["titulo"])
                       + sum(len(i) for i in ENTRADA["itens"]))
    h_decisao = altura(1 + len(DECISAO["linhas"]))
    h_sim = [altura(len(b)) for b in SIM]
    h_estagio = [altura(len(linhas)) for linhas, _n in ESTAGIOS]
    h_nota = {i: altura(len(n)) for i, (_l, n) in enumerate(ESTAGIOS)
              if n is not None}

    seta_v = 10.0                       # entre acoes empilhadas
    h_grupo = sum(h_estagio) + (len(ESTAGIOS) - 1) * seta_v + 2 * PADG
    h_tit_grupo = LINHA + 3.0

    # O ramo "Yes" desce a partir da decisao; o desenho tem de acomodar tambem
    # a nota do ultimo estagio, que e mais alta do que o estagio a que se liga.
    h_coluna_grupo = h_tit_grupo + h_grupo
    x_ent = MARGEM
    x_dec = x_ent + W_ENTRADA + GAP_ENT_DEC
    x_grp = x_dec + W_DECISAO + GAP_DEC_GRP
    x_nota = x_grp + W_GRUPO + GAP_GRP_NOTA
    assert abs((x_nota + W_NOTA + MARGEM) - W_PT) < 1.0, \
        "as colunas nao somam a largura do texto"

    # Centro do primeiro estagio, ao qual a decisao e a entrada se alinham.
    def montar(H):
        y_grp_topo = H - MARGEM - h_tit_grupo
        y_est_centro = []
        y = y_grp_topo - PADG
        for h in h_estagio:
            y_est_centro.append(y - h / 2.0)
            y -= h + seta_v
        return y_grp_topo, y_est_centro

    # Altura provisoria e correcao: nem a cadeia "Yes" nem a nota do ultimo
    # estagio podem furar a margem inferior.
    H = MARGEM * 2 + h_coluna_grupo
    for _ in range(4):
        _yt, centros = montar(H)
        y_dec_centro = centros[0]
        base_sim = (y_dec_centro - h_decisao / 2.0 - seta_v - h_sim[0]
                    - seta_v - h_sim[1])
        base_nota = min(centros[i] - h_nota[i] / 2.0 for i in h_nota)
        falta = MARGEM - min(base_sim, base_nota, MARGEM)
        if falta <= 0.01:
            break
        H += falta

    t = Tela(H, p)
    y_grp_topo, centros = montar(H)
    y_dec_centro = centros[0]

    # --- coluna 1: a informacao disponivel antes da busca -------------------
    y0 = y_dec_centro - h_entrada / 2.0
    t.caixa(x_ent, y0, x_ent + W_ENTRADA, y0 + h_entrada, p["borda"],
            p["surface"])
    interno = W_ENTRADA - 2 * PADB
    y = y0 + h_entrada - PADB - LINHA + 2.6
    for s in ENTRADA["titulo"]:
        t.caber([(s, SF)], interno, "entrada", peso="bold")
        t.texto(x_ent + W_ENTRADA / 2.0, y, s, size=PT, cor=p["acento"],
                peso="bold")
        y -= LINHA
    x_marca = x_ent + PADB + 1.0
    x_item = x_marca + 6.5
    for item in ENTRADA["itens"]:
        t.texto(x_marca, y, "•", ha="left", size=PT, cor=p["ink_2"])
        for s in item:
            t.caber([(s, SF)], x_ent + W_ENTRADA - PADB - x_item, "entrada")
            t.texto(x_item, y, s, ha="left", size=PT, cor=p["ink"])
            y -= LINHA

    # --- coluna 2: a decisao antecipada -------------------------------------
    y0 = y_dec_centro - h_decisao / 2.0
    t.caixa(x_dec, y0, x_dec + W_DECISAO, y0 + h_decisao, p["acento"],
            p["acento_fill"], lw=1.0, r=3.2)
    x_dec_c = x_dec + W_DECISAO / 2.0
    t.texto(x_dec_c, y0 + h_decisao - PADB - LINHA + 2.6, DECISAO["titulo"],
            size=PT, cor=p["acento"], peso="bold")
    yb = y0 + h_decisao - PADB - 2 * LINHA + 2.6
    for ln in DECISAO["linhas"]:
        t.caber(ln, W_DECISAO - 2 * PADB, "decisao")
        t.linha(x_dec_c, yb, ln, ha="center")
        yb -= LINHA
    t.seta(x_ent + W_ENTRADA, y_dec_centro, x_dec, y_dec_centro, p["ink_2"])

    # --- ramo "Yes": desce da decisao em duas acoes -------------------------
    x_sim_c = x_dec_c
    x_sim0 = x_sim_c - W_SIM / 2.0
    y = y0
    t.seta(x_sim_c, y, x_sim_c, y - seta_v, p["ink_2"])
    t.texto(x_sim_c + 3.0, y - seta_v + 3.2, RAMOS["sim"], ha="left", size=PT,
            cor=p["ink_2"], peso="bold")
    y -= seta_v
    for i, bloco in enumerate(SIM):
        t.caixa(x_sim0, y - h_sim[i], x_sim0 + W_SIM, y, p["borda"],
                p["surface"])
        yb = y - PADB - LINHA + 2.6
        for ln in bloco:
            t.caber(ln, W_SIM - 2 * PADB, "ramo Yes")
            t.linha(x_sim_c, yb, ln, ha="center")
            yb -= LINHA
        y -= h_sim[i]
        if i < len(SIM) - 1:
            t.seta(x_sim_c, y, x_sim_c, y - seta_v, p["ink_2"])
            y -= seta_v

    # --- ramo "No": o grupo tracejado com os estagios nativos ---------------
    t.caixa(x_grp, y_grp_topo - h_grupo, x_grp + W_GRUPO, y_grp_topo,
            p["acento_leve"], p["surface"], lw=0.8, dash=(3.0, 2.2), r=3.2,
            zorder=1)
    # O titulo do grupo corre numa linha so, e por isso pode ser mais largo do
    # que a caixa: ele vive fora dela, e as folgas dos dois lados o comportam.
    t.caber([(GRUPO_TITULO, SF)], W_GRUPO + GAP_DEC_GRP + GAP_GRP_NOTA,
            "titulo do grupo", peso="bold")
    t.texto(x_grp + W_GRUPO / 2.0, y_grp_topo + 3.0 - LINHA + 2.6 + LINHA,
            GRUPO_TITULO, size=PT, cor=p["acento"], peso="bold")

    x_est0 = x_grp + PADG
    x_est1 = x_grp + W_GRUPO - PADG
    x_est_c = (x_est0 + x_est1) / 2.0
    for i, (linhas, nota) in enumerate(ESTAGIOS):
        y1 = centros[i] + h_estagio[i] / 2.0
        t.caixa(x_est0, y1 - h_estagio[i], x_est1, y1, p["borda"],
                p["surface"], zorder=3)
        yb = y1 - PADB - LINHA + 2.6
        for ln in linhas:
            t.caber(ln, x_est1 - x_est0 - 2 * PADB, "estagio %d" % (i + 1))
            t.linha(x_est_c, yb, ln, ha="center")
            yb -= LINHA
        if i < len(ESTAGIOS) - 1:
            t.seta(x_est_c, y1 - h_estagio[i], x_est_c,
                   y1 - h_estagio[i] - seta_v, p["ink_2"])
        if nota is None:
            continue
        # A nota lateral, tracejada, e o instante em que a grandeza existe.
        yn = centros[i] + h_nota[i] / 2.0
        t.caixa(x_nota, yn - h_nota[i], x_nota + W_NOTA, yn, p["nota_borda"],
                p["surface"], lw=0.5, dash=(2.2, 1.8))
        t.seta(x_est1, centros[i], x_nota, centros[i], p["nota_borda"],
               lw=0.5, tracejada=True)
        yb = yn - PADB - LINHA + 2.6
        for ln in nota:
            t.caber(ln, W_NOTA - 2 * PADB, "nota %d" % (i + 1))
            t.linha(x_nota + W_NOTA / 2.0, yb, ln, ha="center")
            yb -= LINHA

    # A seta "No" atravessa a borda do grupo, como no rascunho: e ela que diz
    # que a busca inteira e a consequencia da decisao negativa.
    t.seta(x_dec + W_DECISAO, y_dec_centro, x_est0, y_dec_centro, p["ink_2"])
    t.texto((x_dec + W_DECISAO + x_est0) / 2.0, y_dec_centro + 3.2,
            RAMOS["nao"], size=PT, cor=p["ink_2"], peso="bold")

    t.fig.savefig(out_path, facecolor=p["surface"])
    t.fig.savefig(out_path[:-4] + ".png", facecolor=p["surface"], dpi=400)
    plt.close(t.fig)
    print("  gravado: %s  (%.3f x %.2f in)" % (out_path, FIG_W_IN, H / 72.0))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--tex", default=os.path.join(
        root, "results", "thesis", "IEEE_Conference_Template", "ICASSP",
        "paper.tex"))
    ap.add_argument("--out-dir", default=os.path.join(
        root, "results", "thesis", "figuras"))
    ap.add_argument("--variante", choices=sorted(PALETAS) + ["ambas"],
                    default="ambas")
    args = ap.parse_args()

    conferir(args.tex)
    os.makedirs(args.out_dir, exist_ok=True)
    sufixo = {"cor": "", "cinza": "_cinza"}
    alvos = sorted(PALETAS) if args.variante == "ambas" else [args.variante]
    for nome in alvos:
        desenhar(os.path.join(args.out_dir,
                              "figura1_fluxo_informacao%s.pdf" % sufixo[nome]),
                 PALETAS[nome])


if __name__ == "__main__":
    main()
