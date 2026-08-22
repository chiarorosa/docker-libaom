#!/usr/bin/env python3
"""Figura 2 do artigo ICASSP 2027 — perda de taxa-distorcao contra reducao de
custo de busca casada, uma curva por representacao.

Diferente da figura 1, que e diagrama estrutural, esta PLOTA MEDICAO. Le a fonte
unica do artigo, `results/models/oracle_regret_rpp/frontier.csv`, e nao aceita
numero digitado a mao: os valores da Tabela I sao reconferidos contra a
interpolacao aqui mesmo, e o script para se divergirem. E o que impede a figura e
a tabela de contarem historias diferentes.

Composicao:

  eixo x  reducao de custo de busca casada, em por cento — o contador analitico
          de candidatos ponderado por area, e nao tempo de parede;
  eixo y  perda de taxa-distorcao, em 10^-3 por cento, em escala logaritmica.
          Logaritmica porque a faixa util cobre tres ordens de grandeza, de 0,2
          nas representacoes tabulares a 634 no controle aleatorio, e numa escala
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

COL_W_IN = 252.0 / 72.27
FIG_H_IN = 1.72            # reserva no .tex: 4,4 cm = 1,73 in

# --- o que entra, na ordem da Tabela I ---------------------------------------
# `convnext_ce` NAO entra: e o braco legado, treinado noutro conjunto e com duas
# das seis sequencias reservadas contaminadas. Ver Secao III-D.
# Chaves: nome no CSV (ou prefixo das sementes), rotulo do artigo, familia.
# O ultimo campo diz se a curva e DESENHADA. O controle aleatorio fica de fora do
# desenho e permanece na Tabela I: sozinho ele ocupa uma decada inteira do eixo, e
# gasta-la com a ancora oposta comprime justamente as oito curvas em disputa. A
# trava de coerencia abaixo continua conferindo as nove.
CURVAS = [
    ("random",              "random control",           "ref",  False),
    ("variance",            "variance",                 "ref",  True),
    ("convnext_ce_h9",      "ConvNeXt, plain CE",       "deep", True),
    ("convnext_ce_h9_f256", "ConvNeXt, width 256",      "deep", True),
    ("convnext_regret",     "ConvNeXt, cost-sensitive", "deep", True),
    ("RPP_A",               "A",                        "tab",  True),
    ("RPP_A_C",             "A+C",                      "tab",  True),
    ("RPP_A_B_C",           "A+B+C",                    "tab",  True),
    ("RPP_A_B",             "A+B",                      "tab",  True),
]
SEEDS = (0, 1, 2)
LEITURA = [10, 15, 20, 25, 30]     # os pontos da Tabela I
X_MIN, X_MAX = 6.0, 31.0

# Valores da Tabela I, em 10^-3 %, para a trava de coerencia.
TABELA_I = {
    "random control":           [196.3, 298.9, 406.6, 516.6, 634.2],
    "variance":                 [4.3, 12.7, 25.0, 37.4, 55.5],
    "ConvNeXt, plain CE":       [6.5, 10.5, 16.4, 27.2, 44.2],
    "ConvNeXt, width 256":      [7.5, 12.2, 19.4, 29.4, 44.4],
    "ConvNeXt, cost-sensitive": [5.4, 8.0, 14.0, 21.9, 37.9],
    "A":                        [0.5, 1.3, 2.8, 5.3, 9.1],
    "A+C":                      [0.5, 1.2, 2.7, 6.4, 12.2],
    "A+B+C":                    [0.2, 0.6, 1.8, 4.0, 7.8],
    "A+B":                      [0.2, 0.5, 1.5, 3.3, 6.9],
}

# Cor POR CURVA, e nao por familia. A familia tabular ocupa uma faixa estreita do
# eixo — a 25% as quatro cabem entre 3,3 e 6,4 unidades —, e um matiz unico para
# as quatro deixava o cruzamento de A+B com A+B+C ilegivel. Cada uma recebe agora
# um matiz proprio, com boa dispersao de luminancia, e mantem o tracado e o
# marcador como canal redundante: e isso que preserva a leitura em impressao
# monocromatica acidental e sob daltonismo.
# A familia profunda continua agrupada em laranja, em tres tons: ali o que
# importa e que as tres estejam juntas e acima, nao qual e qual.
CURVA_COR = {
    "random control":           "#8a8880",
    "variance":                 "#8a8880",   # referencia, fora da disputa
    "ConvNeXt, plain CE":       "#bf6a1a",
    "ConvNeXt, width 256":      "#d99441",
    "ConvNeXt, cost-sensitive": "#8f4a10",
    "A":                        "#2b7bba",   # azul medio
    "A+C":                      "#46a08a",   # verde-azulado, o mais claro
    "A+B+C":                    "#8a5fa8",   # violeta
    "A+B":                      "#10375c",   # azul profundo, a protagonista
}

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        grade="#e3e1d9", curva=CURVA_COR,
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        grade="#e0ded7",
        curva={"random control": "#9b9992", "variance": "#9b9992",
               "ConvNeXt, plain CE": "#6b6a65",
               "ConvNeXt, width 256": "#8d8c86",
               "ConvNeXt, cost-sensitive": "#55544f",
               "A": "#4a4945", "A+C": "#7a7973",
               "A+B+C": "#2b2a27", "A+B": "#141413"},
    ),
}
# Tracado e marcador por curva. Na variante monocromatica sao eles, e nao o
# matiz, que separam as nove curvas.
ESTILO = {
    "random control":           dict(ls=(0, (1.2, 1.2)), marker=None,  lw=0.6),
    "variance":                 dict(ls=(0, (4, 1.6)),   marker="v",   lw=0.7),
    "ConvNeXt, plain CE":       dict(ls="-",             marker="s",   lw=0.7),
    "ConvNeXt, width 256":      dict(ls=(0, (2.4, 1.2)), marker="P",   lw=0.7),
    "ConvNeXt, cost-sensitive": dict(ls=(0, (5, 1.4, 1, 1.4)), marker="X", lw=0.7),
    "A":                        dict(ls="-",             marker="o",   lw=0.85),
    "A+C":                      dict(ls=(0, (3, 1.3)),   marker="^",   lw=0.85),
    "A+B+C":                    dict(ls=(0, (1.4, 1.2)), marker="D",   lw=0.85),
    "A+B":                      dict(ls="-",             marker="*",   lw=1.05),
}


def ler(csv_path):
    """{pruner: [(cost_red, perda em 10^-3 %)]}, ordenado por cost_red."""
    bruto = collections.defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            bruto[r["pruner"]].append(
                (float(r["cost_red"]), float(r["reg_frac_pct"]) * 1000.0))
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
            if abs(obtido - alvo) > 0.05:
                erros.append("  %-26s @%2d%%: figura %.1f, Tabela I %.1f"
                             % (rotulo, g, obtido, alvo))
    if erros:
        sys.stderr.write("Figura e Tabela I divergem:\n" + "\n".join(erros)
                         + "\nAtualize TABELA_I e a tabela do .tex.\n")
        raise SystemExit(1)
    print("  trava de coerencia: as 45 celulas batem com a Tabela I")


def draw(out_path, p, dados):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0.148, 0.185, 0.836, 0.788])
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
                    markersize=2.4, color=cor, markeredgecolor="none",
                    zorder=5)

    ax.set_yscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0.13, 900)
    # Sem escapar o sinal de porcentagem: o matplotlib nao e LaTeX, e "\%"
    # imprimiria a propria barra invertida.
    ax.set_xlabel("Matched cost reduction (%)", fontsize=6.6,
                  color=p["ink"], labelpad=1.6)
    ax.set_ylabel("Rate-distortion loss ($10^{-3}$ %)", fontsize=6.6,
                  color=p["ink"], labelpad=1.6)
    ax.tick_params(axis="both", which="major", labelsize=6.0,
                   colors=p["ink_2"], length=1.9, width=0.4, pad=1.4)
    ax.tick_params(axis="both", which="minor", length=1.0, width=0.3)
    ax.grid(True, which="major", color=p["grade"], linewidth=0.35, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color(p["muted"])

    # Legenda em duas colunas, dentro do eixo: numa figura de coluna unica uma
    # caixa externa custaria mais altura do que a que a paginacao tem.
    handles = [plt.Line2D([], [], color=p["curva"][r], linewidth=ESTILO[r]["lw"],
                          linestyle=ESTILO[r]["ls"],
                          marker=ESTILO[r]["marker"], markersize=2.4,
                          markeredgecolor="none")
               for _c, r, f, plota in CURVAS if plota]
    leg = ax.legend(handles, [r for _c, r, _f, plota in CURVAS if plota],
                    loc="upper left",
                    ncol=2, fontsize=5.1, frameon=True, framealpha=0.94,
                    borderpad=0.28, labelspacing=0.20, handlelength=2.1,
                    handletextpad=0.38, columnspacing=0.9, borderaxespad=0.30)
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
