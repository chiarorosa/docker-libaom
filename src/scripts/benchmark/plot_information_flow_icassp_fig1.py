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
promete ao dizer "highlighting the pre-search early-termination point".

DUAS RAIAS, e nao uma coluna unica. O ramo "No" desce pela raia da direita com
as quatro etapas de avaliacao; o ramo "Yes" desce pela da esquerda com o seu
unico desfecho. A altura da figura passa a ser o MAIOR dos dois percursos, e nao
a soma: empilhar tudo numa coluna alongava o desenho sem acrescentar leitura.
As duas raias se reencontram no terminador final.

Tres economias de texto:

  O rotulo lateral flutuante "before the current-node RD search" diz QUANDO a
  entrada existe; dentro do paralelogramo ficam so os itens. Assim o simbolo
  carrega dados e a nota carrega tempo, sem que um repita o outro.

  O desfecho do ramo "Yes" diz apenas "Select NONE". Que os candidatos restantes
  e os seus descendentes sao pulados NAO precisa ser escrito: o proprio desvio,
  que salta as quatro caixas de avaliacao, e a afirmacao.

  As anotacoes de custo dizem so o simbolo e "available": o quando e a altura em
  que cada uma esta presa.

Nomes curtos de particao (NONE, SPLIT) seguem a convencao declarada na Secao II
do artigo, que dispensa o prefixo PARTITION_.

Toda linha e MEDIDA no renderizador e conferida contra a largura util do seu
simbolo -- no losango, contra a largura na altura daquela linha. Um rotulo que
nao caiba PARA a execucao em vez de gerar um PDF com texto vazando.

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
H_PT = 213.0
COL_W_IN, FIG_H_IN = W_PT / 72.0, H_PT / 72.0

PT = 6.8        # corpo dos rotulos de simbolo
PT_MONO = 6.0   # tokens em Courier, mais largos que a serif no mesmo corpo
PT_NOTA = 6.0   # ramos, rotulo lateral e anotacoes de custo
LH = 7.8        # entrelinha dentro de um simbolo
PAD = 4.0       # respiro horizontal minimo dentro de um simbolo

TINTA = "#0b0b0b"
FUNDO = "#ffffff"
REALCE = "#e8e6df"
NOTA = "#3f3e3b"
LW, LW_REALCE = 0.6, 1.1

# --- as duas raias -----------------------------------------------------------
X_ESQ, W_ESQ = 3.0, 112.0          # entrada, decisao e o desfecho do "Yes"
X_DIR, W_DIR = 127.0, 114.0        # o percurso "No": as quatro avaliacoes
CX_ESQ, CX_DIR = X_ESQ + W_ESQ / 2, X_DIR + W_DIR / 2

S, M, I = False, True, "it"        # serif, monoespacado, italico

# --- o fluxo desenhado, com procedencia --------------------------------------
# Ordem de avaliacao dentro do no e instante de cada custo: Secao II. Contagem
# de candidatos e representacao de SPLIT pela avaliacao recursiva dos filhos:
# Secao IV-A. Uma linha e uma lista de trechos (texto, estilo).
RAIA_ESQ = [
    dict(tipo="terminador", t=2.0, h=13.0, chave="inicio",
         linhas=[[("Node of the partition tree", S)]]),
    dict(tipo="entrada", t=26.0, h=30.0, chave="entrada",
         linhas=[[("source/block information,", S)],
                 [("coding parameters,", S)],
                 [("causal-neighbor partitions", S)]]),
    dict(tipo="decisao", t=69.0, h=38.0, realce=True, chave="decisao",
         linhas=[[("Early decision:", S)],
                 [("terminate as ", S), ("NONE", M), ("?", S)]]),
    dict(tipo="processo", t=124.0, h=14.0, chave="yes",
         linhas=[[("Select ", S), ("NONE", M)]]),
]
RAIA_DIR = [
    dict(tipo="processo", t=77.5, h=21.0,   # centro alinhado ao do losango
         linhas=[[("Evaluate ", S), ("NONE", M)],
                 [(r"$J_{\mathrm{none}}$ available", I)]]),
    dict(tipo="processo", t=107.5, h=21.0,
         linhas=[[("Evaluate ", S), ("SPLIT", M)],
                 [("and its recursive descendants", S)]]),
    dict(tipo="processo", t=137.5, h=21.0,
         linhas=[[("Evaluate the remaining", S)],
                 [("partition candidates", S)]]),
    dict(tipo="processo", t=167.5, h=21.0,
         linhas=[[("Select the best partition", S)],
                 [(r"$J^{*}$ available", I)]]),
]
FINAL = dict(tipo="terminador", t=196.0, h=13.0, x=45.0, w=155.0,
             linhas=[[("Node decided", S)]])

# Rotulo lateral flutuante: diz QUANDO a entrada existe, no espaco livre a
# direita do paralelogramo, acima da primeira etapa da raia da direita.
LATERAL = dict(t=41.0, linhas=["before the", "current-node RD search"])


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
    exigidos = ["node of the partition tree", "early decision",
                "before the current-node rd search",
                "source/block information", "coding parameters",
                "causal-neighbor", "terminated as none", "cost of split",
                "recursive descendants", "partition candidates",
                "best partition", "j_none", "j^*"]
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

    def kw(estilo):
        if estilo is M:
            return dict(fontsize=PT_MONO, family="monospace")
        if estilo == I:
            return dict(fontsize=PT_NOTA, family="serif", style="italic")
        return dict(fontsize=PT, family="serif")

    def medir(s, estilo):
        t = ax.text(0, 0, s, **kw(estilo))
        w = t.get_window_extent(renderer=rend).width / DPI * 72.0
        t.remove()
        return w

    def Y(t):
        """'Distancia do topo' -> ordenada, para que a leitura acima seja de
        cima para baixo, como o desenho."""
        return H_PT - t

    def rotular(cx, cy, linhas, util, nome):
        # `util` pode ser um numero (simbolos de lado reto) ou uma funcao da
        # distancia ao centro vertical: num losango a largura disponivel encolhe
        # conforme a linha se afasta da meia altura.
        n = len(linhas)
        for i, trechos in enumerate(linhas):
            y = cy + (n - 1) * LH / 2 - i * LH
            larguras = [medir(s, e) for s, e in trechos]
            total = sum(larguras)
            disp = util(y - cy) if callable(util) else util
            if total > disp:
                estouros.append("  %-26s %.1f pt em %.1f pt uteis: %s"
                                % (nome, total, disp,
                                   "".join(s for s, _ in trechos)))
            x = cx - total / 2
            for (s, e), w in zip(trechos, larguras):
                ax.text(x, y, s, color=TINTA, ha="left", va="center",
                        zorder=5, **kw(e))
                x += w

    def simbolo(item, x, w):
        t, h = item["t"], item["h"]
        y0, cy, cx = Y(t + h), Y(t + h / 2), x + w / 2
        realce = item.get("realce", False)
        lw = LW_REALCE if realce else LW
        face = REALCE if realce else FUNDO
        tipo, util = item["tipo"], w - 2 * PAD
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
        rotular(cx, cy, item["linhas"], util, item["linhas"][0][0][0][:24])
        return dict(x=x, w=w, cx=cx, cy=cy, topo=Y(t), base=Y(t + h))

    def seta(p0, p1, ponta=True):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>" if ponta else "-",
                                    color=TINTA, linewidth=LW, shrinkA=0,
                                    shrinkB=0, mutation_scale=5), zorder=4)

    def cotovelo(p0, meio, p1):
        seta(p0, meio, ponta=False)
        seta(meio, p1)

    # ---------------- raia da esquerda --------------------------------------
    esq = {}
    for item in RAIA_ESQ:
        esq[item["chave"]] = simbolo(item, X_ESQ, W_ESQ)
    seta((CX_ESQ, esq["inicio"]["base"]), (CX_ESQ, esq["entrada"]["topo"]))
    seta((CX_ESQ, esq["entrada"]["base"]), (CX_ESQ, esq["decisao"]["topo"]))
    seta((CX_ESQ, esq["decisao"]["base"]), (CX_ESQ, esq["yes"]["topo"]))
    ax.text(CX_ESQ + 2.5, (esq["decisao"]["base"] + esq["yes"]["topo"]) / 2,
            "Yes", fontsize=PT_NOTA, color=NOTA, ha="left", va="center",
            zorder=5)

    # ---------------- raia da direita ---------------------------------------
    dir_ = [simbolo(item, X_DIR, W_DIR) for item in RAIA_DIR]
    for a, b in zip(dir_, dir_[1:]):
        seta((CX_DIR, a["base"]), (CX_DIR, b["topo"]))
    # ramo "No": do vertice direito do losango para a primeira avaliacao
    dec = esq["decisao"]
    seta((dec["x"] + dec["w"], dec["cy"]), (X_DIR - 0.4, dir_[0]["cy"]))
    ax.text((dec["x"] + dec["w"] + X_DIR) / 2, dec["cy"] + 2.0, "No",
            fontsize=PT_NOTA, color=NOTA, ha="center", va="bottom", zorder=5)

    # ---------------- reencontro no terminador final ------------------------
    fim = simbolo(FINAL, FINAL["x"], FINAL["w"])
    seta((CX_ESQ, esq["yes"]["base"]), (CX_ESQ, fim["topo"]))
    seta((CX_DIR, dir_[-1]["base"]), (CX_DIR, fim["topo"]))

    # ---------------- rotulo lateral flutuante ------------------------------
    ent = esq["entrada"]
    ax.plot([ent["x"] + ent["w"] - 6.0, X_DIR + 4.0], [ent["cy"]] * 2,
            color=NOTA, linewidth=0.45, linestyle=(0, (2.2, 1.6)), zorder=2)
    for i, s in enumerate(LATERAL["linhas"]):
        ax.text(X_DIR + 6.0, ent["cy"] + (LH / 2 if i == 0 else -LH / 2), s,
                fontsize=PT_NOTA, color=NOTA, ha="left", va="center",
                style="italic", zorder=5)

    if estouros:
        sys.stderr.write("Rotulo nao cabe no simbolo:\n" + "\n".join(estouros)
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
