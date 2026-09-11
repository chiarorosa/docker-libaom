#!/usr/bin/env python3
"""Figura 2 do artigo ICASSP 2027 — penalidade de custo RD normalizada contra
reducao de busca de particionamento casada, uma curva por conjunto de atributos.

Diferente da figura 1, que e diagrama estrutural, esta PLOTA MEDICAO. Le a fonte
unica do artigo, `results/models/oracle_regret_rpp/frontier.csv`, e nao aceita
numero digitado a mao: os valores da Tabela I sao reconferidos contra a
interpolacao aqui mesmo, e o script para se divergirem. E o que impede a figura e
a tabela de contarem historias diferentes.

Composicao:

  eixo x  reducao de busca de particionamento casada, em por cento — o
          contador analitico de candidatos ponderado por area, e nao tempo
          de parede;
  eixo y  penalidade de custo RD normalizada, em 10^-4 % (a unidade da
          Tabela I), em escala logaritmica.
          Logaritmica porque a faixa util cobre tres ordens de grandeza, de 2
          nas representacoes tabulares a 6342 no controle aleatorio, e numa escala
          linear as quatro tabulares colapsariam sobre o eixo.

  Nos cinco pontos de leitura da Tabela I, as representacoes tabulares levam
  BARRA DE AMPLITUDE entre a pior e a melhor das tres sementes. E amplitude, nao
  intervalo de confianca, e a legenda do artigo diz isso.

Duas paletas, MESMA GEOMETRIA, como nos geradores do LASCAS:
  cor   — a referencia em cinza, a familia profunda agrupada em tres tons de
          laranja, e cada uma das quatro tabulares com matiz proprio, porque
          elas ocupam uma faixa estreita do eixo e um matiz unico as tornava
          ilegiveis umas sobre as outras;
  cinza — o matiz da variante de cor vira luminancia, e tracado e marcador
          continuam separando as nove curvas na impressao monocromatica.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \
        src/scripts/benchmark/plot_loss_frontier_icassp_fig2.py \
        --out-dir results/thesis/figuras

Saidas: figura2_fronteira_perda.pdf       (variante de cor)
        figura2_fronteira_perda_cinza.pdf (variante monocromatica)
        e um .png de cada, para conferencia visual.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Piso tipografico do template do ICASSP: nenhum corpo abaixo de 9 pt na pagina,
# e isso vale para eixo, marcacoes e legenda, nao so para o texto corrido.
# Desenhando na largura de coluna exata do spconf.sty (\textwidth 178 mm com
# \columnsep 6 mm, logo \columnwidth = 86 mm), o \includegraphics[width=
# \columnwidth] fica em escala 1:1 e os 9 pt declarados aqui sao 9 pt medidos.
W_PT = 86.0 / 25.4 * 72.0          # 243,78 pt PostScript
H_PT = 190.0                       # 1,55 in nao comporta rotulos de 9 pt; a
                                   # legenda de oito entradas sozinha pede 58 pt
COL_W_IN = W_PT / 72.0
FIG_H_IN = H_PT / 72.0
PT = 9.0

# Margens do eixo, em pontos, para o texto de 9 pt nao ser cortado.
MARG_ESQ_PT = 38.0         # rotulo do eixo y + marcacoes "1000" + folgas
MARG_INF_PT = 29.0         # marcacoes + rotulo do eixo x + folgas
MARG_DIR_PT = 3.0
MARG_SUP_PT = 4.0

# --- o que entra, na ordem da Tabela I ---------------------------------------
# `convnext_ce` NAO entra: e o braco legado, treinado noutro conjunto e com duas
# das seis sequencias reservadas contaminadas. Ver Secao III-D.
# Chaves: nome no CSV (ou prefixo das sementes), rotulo do artigo, familia.
# O ultimo campo diz se a curva e DESENHADA. O controle aleatorio fica de fora do
# desenho e permanece na Tabela I: sozinho ele ocupa uma decada inteira do eixo, e
# gasta-la com a ancora oposta comprime justamente as oito curvas em disputa. A
# trava de coerencia abaixo continua conferindo TODAS as linhas da tabela,
# desenhadas ou nao.
CURVAS = [
    ("random",              "Random control",           "ref",  False),
    ("variance",            "Variance",                 "ref",  True),
    ("convnext_ce_h9",      "ConvNeXt, plain CE",       "deep", True),
    ("convnext_ce_h9_f256", "ConvNeXt, width 256",      "deep", True),
    ("convnext_regret",     "ConvNeXt, cost-sensitive", "deep", True),
    ("RPP_A",               "BC",                       "tab",  True),
    # Controle de permutacao de BC+NC: mesmas 32 entradas, com as oito colunas
    # de NC reatribuidas entre amostras. Fora do desenho pela mesma razao que o
    # controle aleatorio: a sua curva cai SOBRE a de BC -- que e o resultado --
    # e duas curvas sobrepostas numa coluna de 86 mm tornam ambas ilegiveis sem
    # acrescentar leitura alguma. A trava de coerencia abaixo continua a conferir.
    ("RPP_A_Bshuf",         "BC+shuffled NC",           "tab",  False),
    ("RPP_A_C",             "BC+CSP",                   "tab",  True),
    ("RPP_A_B_C",           "BC+NC+CSP",                "tab",  True),
    ("RPP_A_B",             "BC+NC",                    "tab",  True),
]
SEEDS = (0, 1, 2)
# Ordem da legenda: a da Tabela I, sem o controle aleatorio, que nao e desenhado.
LEGENDA = ["Variance", "ConvNeXt, plain CE", "ConvNeXt, width 256",
           "ConvNeXt, cost-sensitive", "BC", "BC+NC", "BC+CSP", "BC+NC+CSP"]
LEITURA = [10, 15, 20, 25, 30]     # os pontos da Tabela I
X_MIN, X_MAX = 6.0, 31.0

# Valores da Tabela I, em 10^-4 % (equivalente a ppm), para a trava de
# coerencia. Os rotulos sao os do artigo: BC, NC e CSP nomeiam os blocos de
# atributos da Secao III-B; nos artefatos as chaves seguem RPP_A/_B/_C.
TABELA_I = {
    "Random control":           [1963, 2989, 4066, 5166, 6342],
    "Variance":                 [43, 127, 250, 374, 555],
    "ConvNeXt, plain CE":       [65, 105, 164, 272, 442],
    "ConvNeXt, width 256":      [75, 122, 194, 294, 444],
    "ConvNeXt, cost-sensitive": [54, 80, 140, 219, 379],
    "BC":                       [5, 13, 28, 53, 91],
    "BC+shuffled NC":           [5, 13, 27, 48, 86],
    "BC+CSP":                   [5, 12, 27, 64, 122],
    "BC+NC+CSP":                [2, 6, 18, 40, 78],
    "BC+NC":                    [2, 5, 15, 33, 69],
}

# Cor POR CURVA, e nao por familia. A familia tabular ocupa uma faixa estreita do
# eixo — a 25% as quatro cabem entre 3,3 e 6,4 unidades —, e um matiz unico para
# as quatro deixava o cruzamento de BC+NC com BC+NC+CSP ilegivel. Cada uma
# recebe agora um matiz proprio, com boa dispersao de luminancia, e mantem o
# tracado e o marcador como canal redundante: e isso que preserva a leitura em
# impressao monocromatica acidental e sob daltonismo.
# A familia profunda continua agrupada em laranja, em tres tons: ali o que
# importa e que as tres estejam juntas e acima, nao qual e qual.
CURVA_COR = {
    "Random control":           "#8a8880",
    "Variance":                 "#8a8880",   # referencia, fora da disputa
    "ConvNeXt, plain CE":       "#bf6a1a",
    "ConvNeXt, width 256":      "#d99441",
    "ConvNeXt, cost-sensitive": "#8f4a10",
    "BC":                       "#2b7bba",   # azul medio
    "BC+shuffled NC":           "#7fb2d8",   # o azul de BC, dessaturado
    "BC+CSP":                   "#46a08a",   # verde-azulado, o mais claro
    "BC+NC+CSP":                "#8a5fa8",   # violeta
    "BC+NC":                    "#10375c",   # azul profundo, a protagonista
}

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        grade="#e3e1d9", curva=CURVA_COR,
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        grade="#e0ded7",
        curva={"Random control": "#9b9992", "Variance": "#9b9992",
               "ConvNeXt, plain CE": "#6b6a65",
               "ConvNeXt, width 256": "#8d8c86",
               "ConvNeXt, cost-sensitive": "#55544f",
               "BC": "#4a4945", "BC+shuffled NC": "#6d6c66",
               "BC+CSP": "#7a7973",
               "BC+NC+CSP": "#2b2a27", "BC+NC": "#141413"},
    ),
}
# Tracado e marcador por curva. Na variante monocromatica sao eles, e nao o
# matiz, que separam as nove curvas.
ESTILO = {
    "Random control":           dict(ls=(0, (1.2, 1.2)), marker=None,  lw=0.6),
    "Variance":                 dict(ls=(0, (4, 1.6)),   marker="v",   lw=0.7),
    "ConvNeXt, plain CE":       dict(ls="-",             marker="s",   lw=0.7),
    "ConvNeXt, width 256":      dict(ls=(0, (2.4, 1.2)), marker="P",   lw=0.7),
    "ConvNeXt, cost-sensitive": dict(ls=(0, (5, 1.4, 1, 1.4)), marker="X", lw=0.7),
    "BC":                       dict(ls="-",             marker="o",   lw=0.85),
    "BC+shuffled NC":           dict(ls=(0, (2, 1.4)),   marker="o",   lw=0.7),
    "BC+CSP":                   dict(ls=(0, (3, 1.3)),   marker="^",   lw=0.85),
    "BC+NC+CSP":                dict(ls=(0, (1.4, 1.2)), marker="D",   lw=0.85),
    "BC+NC":                    dict(ls="-",             marker="*",   lw=1.05),
}


def ler(csv_path):
    """{pruner: [(cost_red, perda em ppm)]}, ordenado por cost_red.

    `reg_frac_pct` esta em por cento; ppm = por cento x 1e4."""
    bruto = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            bruto[r["pruner"]].append(
                (float(r["cost_red"]), float(r["reg_frac_pct"]) * 1e4))
    return {k: sorted(v) for k, v in bruto.items()}


def curva(dados, chave, xs):
    """Interpola uma perna em xs. Para as tabulares, devolve tambem a pior e a
    melhor das tres sementes, que e o que a barra de amplitude desenha."""
    def um(nome):
        pts = dados[nome]
        return np.interp(xs, [p[0] for p in pts], [p[1] for p in pts])

    if chave.startswith("RPP_"):
        m = np.vstack([um("%s_s%d" % (chave, s)) for s in SEEDS])
        return m.mean(axis=0), m.min(axis=0), m.max(axis=0)
    y = um(chave)
    return y, None, None


def conferir(dados):
    """Trava de coerencia contra a Tabela I. Uma divergencia aqui significa que
    a fronteira foi refeita e que a tabela do .tex ficou para tras."""
    xs = np.array(LEITURA, dtype=float)
    erros = []
    for chave, rotulo, _fam, _plot in CURVAS:
        y, _, _ = curva(dados, chave, xs)
        esperado = TABELA_I[rotulo]
        for g, obtido, alvo in zip(LEITURA, y, esperado):
            if abs(obtido - alvo) > 0.5:
                erros.append("  %-26s @%2d%%: figura %.1f, Tabela I %.1f"
                             % (rotulo, g, obtido, alvo))
    if erros:
        sys.stderr.write("Figura e Tabela I divergem:\n" + "\n".join(erros)
                         + "\nAtualize TABELA_I e a tabela do .tex.\n")
        raise SystemExit(1)
    print("  trava de coerencia: as {} celulas batem com a Tabela I".format(
        len(CURVAS) * len(LEITURA)))


def draw(out_path, p, dados):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([MARG_ESQ_PT / W_PT, MARG_INF_PT / H_PT,
                       1.0 - (MARG_ESQ_PT + MARG_DIR_PT) / W_PT,
                       1.0 - (MARG_INF_PT + MARG_SUP_PT) / H_PT])
    ax.set_facecolor(p["surface"])

    xs = np.linspace(X_MIN, X_MAX, 260)
    xr = np.array(LEITURA, dtype=float)

    for chave, rotulo, fam, plota in CURVAS:
        if not plota:
            continue
        e = ESTILO[rotulo]
        cor = p["curva"][rotulo]
        y, lo, hi = curva(dados, chave, xs)
        ax.plot(xs, y, color=cor, linewidth=e["lw"], linestyle=e["ls"],
                zorder=3, solid_capstyle="round")
        if e["marker"] is not None:
            yr, lor, hir = curva(dados, chave, xr)
            if lor is not None:
                ax.errorbar(xr, yr, yerr=[yr - lor, hir - yr], fmt="none",
                            ecolor=cor, elinewidth=0.55, capsize=1.2,
                            capthick=0.55, zorder=4)
            ax.plot(xr, yr, linestyle="none", marker=e["marker"],
                    markersize=3.0, color=cor, markeredgecolor="none",
                    zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(1.3, 9000)
    # Marcacoes de y como inteiros, e nao como 10^n. O expoente de uma potencia
    # le-se mais direto do que a potencia.
    # 9 pt do template; alem disso, com so tres decadas uteis "10 / 100 / 1000"
    # le-se mais direto em ppm.
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _pos: "%g" % v))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Sem escapar o sinal de porcentagem: o matplotlib nao e LaTeX, e "\%"
    # imprimiria a propria barra invertida.
    ax.set_xlabel("Partition-search reduction (%)", fontsize=PT,
                  color=p["ink"], labelpad=2.0)
    ax.set_ylabel("Normalized RD-cost penalty ($10^{-4}$%)", fontsize=PT,
                  color=p["ink"], labelpad=2.0)
    ax.tick_params(axis="both", which="major", labelsize=PT,
                   colors=p["ink_2"], length=2.2, width=0.4, pad=1.8)
    ax.tick_params(axis="both", which="minor", length=1.0, width=0.3)
    ax.grid(True, which="major", color=p["grade"], linewidth=0.35, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(p["muted"])

    # Legenda em duas colunas, dentro do eixo: numa figura de coluna unica uma
    # caixa externa custaria mais altura do que a que a paginacao tem.
    # Com ncol=2 e oito entradas, o matplotlib preenche coluna a coluna, e a
    # ordem de CURVAS ja separa as quatro longas (referencia e familia profunda)
    # das quatro curtas (tabulares) -- e o que faz a caixa caber a 9 pt.
    # A legenda segue a ORDEM DA TABELA I, e nao a ordem de desenho: quem le
    # chega aqui vindo da tabela, e uma ordem diferente obrigaria a procurar
    # linha por linha. A ordem de desenho continua sendo a de CURVAS, que deixa
    # BC+NC por ultimo para que a protagonista nao seja encoberta.
    handles = [plt.Line2D([], [], color=p["curva"][r], linewidth=ESTILO[r]["lw"],
                          linestyle=ESTILO[r]["ls"],
                          marker=ESTILO[r]["marker"], markersize=3.0,
                          markeredgecolor="none")
               for r in LEGENDA]
    leg = ax.legend(handles, LEGENDA,
                    loc="upper left",
                    ncol=2, fontsize=PT, frameon=True, framealpha=0.94,
                    borderpad=0.28, labelspacing=0.22, handlelength=1.8,
                    handletextpad=0.38, columnspacing=0.8, borderaxespad=0.30)
    leg.get_frame().set_linewidth(0.35)
    leg.get_frame().set_edgecolor(p["grade"])
    for t in leg.get_texts():
        t.set_color(p["ink_2"])

    fig.savefig(out_path, facecolor=p["surface"])
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=p["surface"], dpi=400)
    plt.close(fig)
    print("  gravado: %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--frontier",
                    default=os.path.join(root, "results", "models",
                                         "oracle_regret_rpp", "frontier.csv"))
    ap.add_argument("--out-dir",
                    default=os.path.join(root, "results", "thesis", "figuras"))
    ap.add_argument("--variante", choices=sorted(PALETAS) + ["ambas"],
                    default="ambas")
    args = ap.parse_args()

    dados = ler(args.frontier)
    conferir(dados)

    os.makedirs(args.out_dir, exist_ok=True)
    sufixo = {"cor": "", "cinza": "_cinza"}
    alvos = sorted(PALETAS) if args.variante == "ambas" else [args.variante]
    for nome in alvos:
        draw(os.path.join(args.out_dir,
                          "figura2_fronteira_perda%s.pdf" % sufixo[nome]),
             PALETAS[nome], dados)
    print("\nDimensoes finais: %.3f x %.2f in, para "
          "\\includegraphics[width=\\columnwidth]{figura2_fronteira_perda}"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
