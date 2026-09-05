#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 - fluxo da busca de particionamento intra do
AV1 num no, com o ponto de decisao antecipada e a informacao disponivel em cada
instante.

Diagrama ESTRUTURAL, como a figura 2 do LASCAS: nao le artefato numerico, e sim
declara em blocos os fatos que desenha, cada um com a sua procedencia no .tex, de
modo que uma mudanca de terminologia no artigo obrigue a mexer aqui e nao passe
despercebida. A trava `conferir()` reconfere o vocabulario contra o proprio
`paper.tex` e para se divergirem.

Substitui a versao em TikZ (`ICASSP/fig1_information_flow.txt`), que compunha em
Computer Modern, usava um azul proprio e desenhava numa proporcao larga demais
para a coluna: reduzida a 86 mm, a tipografia caia para cerca de 5 pt, muito
abaixo do piso do template.

Composicao, em tres bandas sobre um unico eixo em pontos PostScript:

  1. ENTRADA - o que ja existe antes de comecar a busca RD do no corrente.
     Ocupa a largura inteira porque e a premissa do artigo: sao esses os
     conjuntos de atributos comparados na Secao III-B.

  2. DECISAO - a decisao antecipada, com a saida "Yes" saindo pela DIREITA, na
     mesma banda, e nao para baixo. E o que torna a figura estreita o bastante
     para caber na coluna com corpo de 9 pt: a banda de baixo fica livre para
     usar a largura inteira.

  3. BUSCA - os estagios nativos do no, empilhados, cada um numa unica linha.
     Nas duas linhas em que uma grandeza de custo passa a existir, a nota vai
     ALINHADA A DIREITA na mesma linha, de modo que o instante em que a
     informacao aparece se leia por alinhamento vertical, sem conector.

Duas paletas, MESMA GEOMETRIA, como nos demais geradores:
  cor   - o azul-aco #25599f dos acentos das figuras do grupo;
  cinza - o mesmo desenho em luminancia, para impressao monocromatica.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \
        src/scripts/benchmark/plot_information_flow_icassp_fig1.py \
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
# OTF, e o matplotlib embute CFF declarando TrueType — mismatch que o PDF
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
# 9 pt declarados aqui sejam 9 pt medidos na pagina.
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt PostScript
COL_W_IN = W_PT / 72.0

PT = 9.0        # piso tipografico do template: nada abaixo disso na pagina
PT_TT = 9.0     # o mesmo corpo do \texttt da pagina, e a mesma fonte
LINHA = 10.6    # entrelinha
PAD = 5.0       # respiro interno das caixas

# --- os fatos desenhados, com procedencia ------------------------------------
# Todo texto abaixo vem do proprio paper.tex, Secao II ("AV1 Intra Partition
# Search and Early-Termination Point"). A terminologia do artigo e normativa:
# esta figura se ajusta a ela, nunca o contrario.
ENTRADA = {
    "titulo": "Available before the current-node RD search",
    "linhas": ["Source and block information, coding parameters, and",
               "previously coded causal-neighbor partitions"],
}
DECISAO = {
    "titulo": "Early decision",
    "linha": "Terminate the node as",
    "token": "PARTITION_NONE?",
}
TERMINA = ["Skip the remaining", "partition alternatives and",
           "their recursive descendants"]
BUSCA = {
    "titulo": "Generated during the current-node search",
    # (prefixo em Times, token monoespacado, sufixo em Times, nota a direita)
    "estagios": [
        ("Evaluate ", "PARTITION_NONE", "",
         "$J_\\mathrm{none}$ becomes available"),
        ("Evaluate ", "PARTITION_SPLIT", " and recurse", None),
        ("Evaluate rectangular and extended partitions", None, "", None),
        ("Select best partition", None, "", "RD cost of the best partition"),
    ],
}
RAMOS = {"sim": "Yes", "nao": "No"}

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        borda="#b9b7ae", regua="#e3e1d9",
        acento="#25599f", acento_fill="#eef3fa", acento_leve="#c3d1e6",
        faixa="#e9eff8",
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        borda="#b5b3ac", regua="#e0ded7",
        acento="#141413", acento_fill="#eeece7", acento_leve="#bcbab3",
        faixa="#eae8e2",
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


class Tela(object):
    """Eixo unico em pontos PostScript, origem no canto inferior esquerdo."""

    def __init__(self, altura_pt, p):
        self.h = altura_pt
        self.p = p
        self.fig = plt.figure(figsize=(COL_W_IN, altura_pt / 72.0), dpi=400)
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

    # -- primitivas ---------------------------------------------------------
    def caixa(self, x0, y0, x1, y1, borda, fill, lw=0.5, dash=None, r=2.4):
        pat = FancyBboxPatch(
            (x0 + r, y0 + r), x1 - x0 - 2 * r, y1 - y0 - 2 * r,
            boxstyle="round,pad=%f,rounding_size=%f" % (r, r),
            linewidth=lw, edgecolor=borda, facecolor=fill, zorder=2)
        if dash is not None:
            pat.set_linestyle((0, dash))
        self.ax.add_patch(pat)
        return pat

    def texto(self, x, y, s, ha="center", size=PT, cor=None, familia=None,
              peso="normal", zorder=5):
        kw = {}
        if familia is not None:
            kw["family"] = familia
        return self.ax.text(x, y, s, ha=ha, va="baseline", fontsize=size,
                            color=cor or self.p["ink"], fontweight=peso,
                            zorder=zorder, **kw)

    def linha_mista(self, x, y, partes, ha="left"):
        """Uma linha com trechos em Times e trechos monoespacados.

        `partes` = [(texto, familia, tamanho, cor)]. Devolve a largura total."""
        larguras = [self.largura(s, family=f, fontsize=t)
                    for s, f, t, _c in partes]
        total = sum(larguras)
        cur = x if ha == "left" else (x - total if ha == "right"
                                      else x - total / 2.0)
        for (s, f, t, c), w in zip(partes, larguras):
            self.texto(cur, y, s, ha="left", size=t, cor=c, familia=f)
            cur += w
        return total

    def caber(self, s, limite, onde, **kw):
        """Trava de transbordo: nenhum texto pode estourar a sua caixa.

        Sem ela, um ajuste de vocabulario passaria despercebido no .pdf e so
        apareceria na prova impressa."""
        w = self.largura(s, **kw)
        if w > limite:
            sys.stderr.write(
                "Texto estoura a caixa em %s: %.1f pt para %.1f pt disponiveis\n"
                "  %r\n" % (onde, w, limite, s))
            raise SystemExit(1)
        return w

    def seta(self, x0, y0, x1, y1, cor, lw=0.8):
        self.ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>",
            mutation_scale=5.0, linewidth=lw, color=cor,
            shrinkA=0, shrinkB=0, zorder=4))


def desenhar(out_path, p):
    # --- altura, somada banda a banda antes de abrir a figura ---------------
    h_entrada = 3 * LINHA + 2 * PAD
    h_decisao = 3 * LINHA + 2 * PAD
    h_busca = LINHA + len(BUSCA["estagios"]) * (LINHA + 2.6) + 2 * PAD
    seta_v = 11.0
    H = 1.0 + h_entrada + seta_v + h_decisao + seta_v + h_busca + 1.0

    t = Tela(H, p)
    x0, x1 = 0.6, W_PT - 0.6
    xc = (x0 + x1) / 2.0
    # A caixa da decisao recebe so a largura de que precisa; a folga vai para a
    # saida "Yes", que carrega o texto mais longo da banda. O eixo do fluxo e o
    # centro DELA, e nao o centro da figura.
    w_decisao = 106.0
    seta_h = 16.0
    xd1 = x0 + w_decisao
    xdc = (x0 + xd1) / 2.0
    w_saida = x1 - xd1 - seta_h

    # --- banda 1: informacao disponivel antes da busca ----------------------
    ye1 = H - 1.0
    ye0 = ye1 - h_entrada
    t.caixa(x0, ye0, x1, ye1, p["borda"], p["surface"])
    base = ye1 - PAD - LINHA + 2.6
    t.texto(xc, base, ENTRADA["titulo"], size=PT, cor=p["acento"], peso="bold")
    for i, s in enumerate(ENTRADA["linhas"]):
        t.texto(xc, base - (i + 1) * LINHA, s, size=PT, cor=p["ink"])

    # --- seta para a decisao ------------------------------------------------
    t.seta(xdc, ye0, xdc, ye0 - seta_v, p["ink_2"])

    # --- banda 2: a decisao, com a saida "Yes" pela direita -----------------
    yd1 = ye0 - seta_v
    yd0 = yd1 - h_decisao
    t.caixa(x0, yd0, xd1, yd1, p["acento"], p["acento_fill"], lw=0.9)
    base = yd1 - PAD - LINHA + 2.6
    t.texto(xdc, base, DECISAO["titulo"], size=PT, cor=p["acento"], peso="bold")
    t.texto(xdc, base - LINHA, DECISAO["linha"], size=PT, cor=p["ink"])
    t.texto(xdc, base - 2 * LINHA, DECISAO["token"], size=PT_TT, cor=p["ink"],
            familia="monospace")

    # saida "Yes": o no termina aqui
    ys1, ys0 = yd1, yd0
    t.caixa(x1 - w_saida, ys0, x1, ys1, p["borda"], p["surface"])
    ymeio = (ys0 + ys1) / 2.0
    t.seta(xd1, ymeio, x1 - w_saida, ymeio, p["ink_2"])
    t.texto((xd1 + x1 - w_saida) / 2.0, ymeio + 3.4, RAMOS["sim"], size=PT,
            cor=p["ink_2"])
    base = ys1 - PAD - LINHA + 2.6
    for i, s in enumerate(TERMINA):
        t.caber(s, w_saida - 2 * PAD, "saida Yes", fontsize=PT)
        t.texto(x1 - w_saida / 2.0, base - i * LINHA, s, size=PT, cor=p["ink"])

    # --- seta para a busca --------------------------------------------------
    t.seta(xdc, yd0, xdc, yd0 - seta_v, p["ink_2"])
    t.texto(xdc + 3.0, yd0 - seta_v + 2.8, RAMOS["nao"], ha="left", size=PT,
            cor=p["ink_2"])

    # --- banda 3: os estagios nativos do no ---------------------------------
    yb1 = yd0 - seta_v
    yb0 = yb1 - h_busca
    t.caixa(x0, yb0, x1, yb1, p["acento_leve"], p["surface"], lw=0.7,
            dash=(2.4, 1.8))
    base = yb1 - PAD - LINHA + 2.6
    t.texto(xc, base, BUSCA["titulo"], size=PT, cor=p["acento"], peso="bold")

    # Regua de ordem: a lista e uma SEQUENCIA, e sem ela leria como conjunto.
    # Uma regua fina com ponta na ultima linha custa menos altura do que uma
    # seta entre cada par de estagios.
    xt0, xt1 = x0 + PAD + 8.0, x1 - PAD - 1.0
    y_top = base - LINHA + 3.0
    y_fim = base - LINHA - (len(BUSCA["estagios"]) - 1) * (LINHA + 2.6) - 3.4
    t.seta(xt0 - 5.0, y_top, xt0 - 5.0, y_fim, p["acento_leve"], lw=0.6)

    for i, (pre, token, pos, nota) in enumerate(BUSCA["estagios"]):
        yb = base - LINHA - i * (LINHA + 2.6)
        # A faixa marca APENAS os dois estagios em que uma grandeza de custo
        # passa a existir; alternar linha sim, linha nao nao significaria nada.
        if nota is not None:
            t.ax.add_patch(FancyBboxPatch(
                (xt0 - 1.5, yb - 2.9), xt1 - xt0 + 1.5, LINHA - 0.6,
                boxstyle="round,pad=0,rounding_size=1.2", linewidth=0,
                facecolor=p["faixa"], zorder=3))
        partes = [(pre, "serif", PT, p["ink"])]
        if token is not None:
            partes.append((token, "monospace", PT_TT, p["ink"]))
        if pos:
            partes.append((pos, "serif", PT, p["ink"]))
        w_estagio = t.linha_mista(xt0, yb, partes, ha="left")
        if nota is not None:
            w_nota = t.largura(nota, fontsize=PT)
            if w_estagio + w_nota + 6.0 > xt1 - xt0:
                sys.stderr.write(
                    "Estagio e nota colidem na linha %d: %.1f + %.1f pt para "
                    "%.1f pt\n" % (i + 1, w_estagio, w_nota, xt1 - xt0))
                raise SystemExit(1)
            t.texto(xt1, yb, nota, ha="right", size=PT, cor=p["acento"])

    t.fig.savefig(out_path, facecolor=p["surface"])
    t.fig.savefig(out_path[:-4] + ".png", facecolor=p["surface"], dpi=400)
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
