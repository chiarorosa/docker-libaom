#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 — fluxograma da busca de particionamento intra
do AV1 num no, com o ponto de terminacao antecipada PRE-BUSCA que este trabalho
ataca.

Fluxograma classico, na convencao de livro-texto:

  terminador (retangulo de cantos arredondados)  inicio e fim do percurso;
  paralelogramo                                  entrada de dados;
  losango                                        decisao, com os ramos
                                                 rotulados "Yes" e "No";
  retangulo                                      processo;
  conector ortogonal com ponta cheia             fluxo de controle.

Line art em preto sobre branco, sem matiz: a unica enfase e a decisao estudada,
marcada por preenchimento leve e traco mais pesado -- que e o que a legenda
promete ao dizer "highlighting the pre-search early-termination point". Assim a
figura sobrevive a impressao monocromatica sem variante separada.

Duas economias deliberadas de texto:

  O desfecho do ramo "Yes" diz apenas "Select PARTITION_NONE". Que os candidatos
  restantes e os seus descendentes sao pulados NAO precisa ser escrito: o proprio
  desvio, que salta todas as caixas de avaliacao, e a afirmacao. Um fluxograma
  que repete em texto o que a seta ja diz esta dizendo duas vezes.

  As anotacoes de custo sao "J_none available" e "J* available", sem dizer QUANDO:
  o quando e a altura em que a anotacao esta presa. A da direita so aparece na
  ultima caixa, que e o que significa "only after the full search".

Toda linha e MEDIDA no renderizador e conferida contra a largura util do seu
simbolo; um rotulo que nao caiba PARA a execucao em vez de gerar um PDF com
texto vazando. E o que impede que mexer no texto sem reconferir a geometria
passe despercebido.

Diagrama ESTRUTURAL: nao le artefato numerico, declara em FLUXO os fatos que
desenha, e a trava `conferir()` reconfere cada expressao contra o .tex do artigo.
A terminologia do artigo e normativa.

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
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

# --- tipografia: identica a da figura 2 do artigo ----------------------------
# O spconf.sty compoe em Times; STIXGeneral e a serif de metrica Times que
# acompanha o matplotlib. fonttype 42 embute como TrueType, pois o Type 3 padrao
# costuma ser recusado pelo PDF eXpress do IEEE. O monoespacado e a conversao
# TrueType do NimbusMonL feita por `otf2ttf_nimbus_mono.py`, que e o desenho que
# o \texttt da pagina compoe.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "font.monospace": ["Nimbus Mono PS TT", "DejaVu Sans Mono"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DPI = 400
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt — \columnwidth do spconf.sty
H_PT = 250.0
COL_W_IN, FIG_H_IN = W_PT / 72.0, H_PT / 72.0

PT = 6.8        # corpo dos rotulos de simbolo
PT_MONO = 6.0   # tokens em Courier, mais largos que a serif no mesmo corpo
PT_NOTA = 6.0   # ramos "Yes"/"No" e anotacoes de custo
LH = 7.8        # entrelinha dentro de um simbolo
PAD = 4.0       # respiro horizontal minimo dentro de um simbolo

TINTA = "#0b0b0b"
FUNDO = "#ffffff"
REALCE = "#e8e6df"      # preenchimento do losango da decisao estudada
NOTA = "#3f3e3b"
LW, LW_REALCE = 0.6, 1.1

# --- as tres raias verticais -------------------------------------------------
# Esquerda: o desfecho do ramo "Yes". Centro: o fluxo principal. Direita: as
# anotacoes de custo. Separa-las e o que evita cruzamento de conectores.
X_ESQ, W_ESQ = 3.0, 79.0
X_CEN, W_CEN = 90.0, 100.0
X_DIR = 194.0
CX_ESQ, CX_CEN = X_ESQ + W_ESQ / 2, X_CEN + W_CEN / 2

# --- o fluxo desenhado, com procedencia --------------------------------------
# Ordem de avaliacao dentro do no e instante de cada custo: Secao II. Contagem
# de candidatos e representacao de PARTITION_SPLIT pela avaliacao recursiva dos
# filhos: Secao IV-A. Uma linha e uma lista de trechos (texto, monoespacado?).
S, M = False, True          # serif, monoespacado
FLUXO = [
    dict(tipo="terminador", t=2.0, h=13.0,
         linhas=[[("Node of the partition tree", S)]]),
    dict(tipo="entrada", t=25.0, h=36.0,
         linhas=[[("Pre-search information:", S)],
                 [("source/block information,", S)],
                 [("coding parameters,", S)],
                 [("causal-neighbor partitions", S)]]),
    dict(tipo="decisao", t=73.0, h=38.0, realce=True,
         linhas=[[("Terminate as", S)], [("PARTITION_NONE", M), ("?", S)]]),
    dict(tipo="processo", t=123.0, h=14.0, chave="none",
         linhas=[[("Evaluate ", S), ("PARTITION_NONE", M)]]),
    dict(tipo="processo", t=147.0, h=21.0,
         linhas=[[("Evaluate ", S), ("PARTITION_SPLIT", M)],
                 [("and its recursive descendants", S)]]),
    dict(tipo="processo", t=178.0, h=21.0,
         linhas=[[("Evaluate the remaining", S)],
                 [("partition candidates", S)]]),
    dict(tipo="processo", t=209.0, h=14.0, chave="best",
         linhas=[[("Select the best partition", S)]]),
    dict(tipo="terminador", t=233.0, h=13.0,
         linhas=[[("Node decided", S)]]),
]

# Desfecho do ramo "Yes", na raia da esquerda, na altura da ultima etapa.
RAMO_YES = dict(t=209.0, h=14.0,
                linhas=[[("Select ", S), ("PARTITION_NONE", M)]])

# Anotacoes de custo, na raia da direita. `alvo` e a chave da caixa de onde
# parte a linha tracejada; a altura e que diz "quando".
NOTAS = [dict(alvo="none", texto=r"$J_{\mathrm{none}}$ available"),
         dict(alvo="best", texto=r"$J^{*}$ available")]


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
    exigidos = ["node of the partition tree", "pre-search",
                "source/block information", "coding parameters",
                "causal-neighbor", "terminat", "partition_none",
                "partition_split", "recursive descendants",
                "partition candidates", "best partition", "j_none", "j^*"]
    faltam = [e for e in exigidos if e not in plano]
    if faltam:
        sys.stderr.write("Figura 1 usa termo ausente do artigo:\n  "
                         + "\n  ".join(faltam)
                         + "\nAjuste a figura ao vocabulario do .tex.\n")
        raise SystemExit(1)
    print("  trava de vocabulario: %d expressoes conferidas" % len(exigidos))


def draw(out_path):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT)
    ax.set_ylim(0, H_PT)
    ax.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    estouros = []

    def medir(s, size, mono):
        """Largura de um trecho, em pontos, medida no renderizador."""
        t = ax.text(0, 0, s, fontsize=size,
                    family="monospace" if mono else "serif")
        w = t.get_window_extent(renderer=rend).width / DPI * 72.0
        t.remove()
        return w

    def Y(t):
        """'Distancia do topo' -> ordenada, para que a leitura do FLUXO acima
        seja de cima para baixo, como o desenho."""
        return H_PT - t

    def rotular(cx, cy, linhas, util, nome):
        # `util` pode ser um numero (simbolos de lado reto) ou uma funcao da
        # distancia ao centro vertical: num losango a largura disponivel encolhe
        # a medida que a linha se afasta da meia altura, e conferir pela largura
        # central reprovaria rotulos que cabem.
        """Compoe as linhas centradas no conjunto. Uma linha pode misturar
        serif e Courier; os trechos sao medidos e assentados em sequencia, de
        modo que a LINHA fique centrada -- e nao a emenda entre os trechos."""
        n = len(linhas)
        for i, trechos in enumerate(linhas):
            y = cy + (n - 1) * LH / 2 - i * LH
            larguras = [medir(s, PT_MONO if mono else PT, mono)
                        for s, mono in trechos]
            total = sum(larguras)
            disp = util(y - cy) if callable(util) else util
            if total > disp:
                estouros.append("  %-28s %.1f pt em %.1f pt uteis: %s"
                                % (nome, total, disp,
                                   "".join(s for s, _ in trechos)))
            x = cx - total / 2
            for (s, mono), w in zip(trechos, larguras):
                ax.text(x, y, s, fontsize=PT_MONO if mono else PT, color=TINTA,
                        ha="left", va="center",
                        family="monospace" if mono else "serif", zorder=5)
                x += w

    def simbolo(tipo, x, w, t, h, linhas, realce=False):
        y0, cy, cx = Y(t + h), Y(t + h / 2), x + w / 2
        lw = LW_REALCE if realce else LW
        face = REALCE if realce else FUNDO
        util = w - 2 * PAD
        if tipo == "processo":
            ax.add_patch(Rectangle((x, y0), w, h, facecolor=face,
                                   edgecolor=TINTA, linewidth=lw, zorder=3))
        elif tipo == "terminador":
            r = h / 2 - 0.8
            ax.add_patch(FancyBboxPatch((x + r, y0 + 0.8), w - 2 * r, h - 1.6,
                                        boxstyle="round,pad=%.2f" % r,
                                        facecolor=face, edgecolor=TINTA,
                                        linewidth=lw, zorder=3))
            util = w - 2 * r - PAD
        elif tipo == "entrada":     # paralelogramo
            d = 8.0
            ax.add_patch(Polygon([(x + d, y0), (x + w, y0),
                                  (x + w - d, y0 + h), (x, y0 + h)],
                                 closed=True, facecolor=face, edgecolor=TINTA,
                                 linewidth=lw, zorder=3))
            util = w - 2 * d - PAD
        elif tipo == "decisao":     # losango
            ax.add_patch(Polygon([(cx, y0), (x + w, cy), (cx, y0 + h), (x, cy)],
                                 closed=True, facecolor=face, edgecolor=TINTA,
                                 linewidth=lw, zorder=3))
            util = lambda dy, w=w, h=h: w * (1.0 - abs(dy) / (h / 2.0)) - 2 * PAD
        rotular(cx, cy, linhas, util, linhas[0][0][0][:26])
        return dict(x=x, w=w, cx=cx, cy=cy, topo=Y(t), base=Y(t + h))

    def seta(p0, p1, ponta=True):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>" if ponta else "-",
                                    color=TINTA, linewidth=LW, shrinkA=0,
                                    shrinkB=0, mutation_scale=5), zorder=4)

    def cotovelo(p0, meio, p1):
        seta(p0, meio, ponta=False)
        seta(meio, p1)

    # ---------------- fluxo principal ---------------------------------------
    por_chave, caixas = {}, []
    for item in FLUXO:
        c = simbolo(item["tipo"], X_CEN, W_CEN, item["t"], item["h"],
                    item["linhas"], item.get("realce", False))
        caixas.append(c)
        if "chave" in item:
            por_chave[item["chave"]] = c
    for a, b in zip(caixas, caixas[1:]):
        seta((CX_CEN, a["base"]), (CX_CEN, b["topo"]))

    dec, fim = caixas[2], caixas[-1]
    ax.text(CX_CEN + 2.5, (dec["base"] + caixas[3]["topo"]) / 2, "No",
            fontsize=PT_NOTA, color=NOTA, ha="left", va="center", zorder=5)

    # ---------------- ramo "Yes" --------------------------------------------
    yes = simbolo("processo", X_ESQ, W_ESQ, RAMO_YES["t"], RAMO_YES["h"],
                  RAMO_YES["linhas"])
    cotovelo((dec["x"], dec["cy"]), (CX_ESQ, dec["cy"]), (CX_ESQ, yes["topo"]))
    ax.text(dec["x"] - 2.5, dec["cy"] + 2.5, "Yes", fontsize=PT_NOTA,
            color=NOTA, ha="right", va="bottom", zorder=5)
    cotovelo((CX_ESQ, yes["base"]), (CX_ESQ, fim["cy"]),
             (X_CEN - 0.4, fim["cy"]))

    # ---------------- anotacoes de custo ------------------------------------
    for nota in NOTAS:
        alvo = por_chave[nota["alvo"]]
        ax.plot([alvo["x"] + alvo["w"], X_DIR - 2.0], [alvo["cy"]] * 2,
                color=NOTA, linewidth=0.45, linestyle=(0, (2.2, 1.6)), zorder=2)
        ax.text(X_DIR, alvo["cy"], nota["texto"], fontsize=PT_NOTA, color=NOTA,
                ha="left", va="center", zorder=5)

    if estouros:
        sys.stderr.write("Rotulo nao cabe no simbolo:\n"
                         + "\n".join(estouros)
                         + "\nEncurte o texto ou alargue a raia.\n")
        raise SystemExit(1)
    print("  ajuste de texto: todas as linhas cabem nos seus simbolos")

    fig.savefig(out_path, facecolor=FUNDO)
    fig.savefig(out_path.replace(".pdf", ".png"), facecolor=FUNDO, dpi=DPI)
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
    args = ap.parse_args(argv)
    conferir(args.tex)
    os.makedirs(args.out_dir, exist_ok=True)
    draw(os.path.join(args.out_dir, "figura1_fluxo_informacao.pdf"))


if __name__ == "__main__":
    main(sys.argv[1:])
