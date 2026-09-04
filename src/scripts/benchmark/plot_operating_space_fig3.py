#!/usr/bin/env python3
"""Figura 3 do artigo LASCAS 2027 — espaco de operacao do SNP-AV1 na CTC A1.

Le DIRETAMENTE o artefato canonico `results/benchmark/fase6/bdrate_average.csv`,
que e a mesma fonte de onde cada numero do texto do artigo foi transcrito, de
modo que figura e texto nao possam divergir. Os valores lidos sao conferidos
contra os do artigo antes de plotar; qualquer divergencia aborta a execucao.

Eixo x: reducao de tempo (TS, definicao canonica da M3, media por ponto de
quantizacao e depois por sequencia). Eixo y: taxa BD sobre PSNR-Y. Mais a
direita e mais abaixo e melhor. A origem e a ancora, o libaom v3.10.0 intocado
em `cpu-used=0`.

ESCOPO (decisao editorial de 2026-09-04, orientacao): a figura NAO apresenta a
escada de presets nativos. O SNP-AV1 age apenas dentro da busca de particao,
enquanto um preset reconfigura todas as etapas do codificador; po-los lado a
lado compara arranjos de escopo distinto. O eixo de comparacao desta figura e
interno: os seis pontos medidos do SNP-AV1 contra a ancora exaustiva, e as duas
taxas de cambio entre tempo e taxa BD que a geometria dos pontos expoe.

    - o segmento tracejado liga as duas BASES (apenas o primeiro estagio) e e,
      portanto, o botao de limiar do primeiro estagio: a unica alternativa que a
      solucao implantada oferece ao usuario para comprar mais tempo;
    - o primeiro salto dentro de cada familia e o segundo estagio na calibracao
      implantada, e a sua inclinacao e visivelmente mais suave.

As duas inclinacoes anotadas sao calculadas no proprio script, a partir dos
mesmos pontos plotados, e impressas no terminal para auditoria.

Uso (dentro do conteiner):
    build/venv-ml/bin/python src/scripts/benchmark/plot_operating_space_fig3.py \
        --out-dir results/thesis/figuras

Saidas: figura3_espaco_operacao.pdf (vetorial, 1:1 na largura de coluna do
                                     IEEEtran, para \\includegraphics)
        figura3_espaco_operacao.png (para conferencia visual)
        figura3_dados.csv           (os valores plotados, para auditoria)
"""
import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- tipografia: a mesma familia do corpo do artigo -------------------------
# O IEEEtran compoe em Times. STIXGeneral e a serif desenhada justamente para
# casar com as metricas de Times em publicacao cientifica, e acompanha o
# matplotlib, o que dispensa instalar fonte no conteiner. O resultado le como
# continuacao do texto, e nao como um grafico colado de outra ferramenta.
#   fonttype 42 embute a fonte como TrueType. O padrao do matplotlib e Type 3,
#   que o PDF eXpress do IEEE costuma recusar.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- geometria: a figura e desenhada no TAMANHO FINAL ------------------------
# \columnwidth do IEEEtran [conference] em papel carta mede 252,0 pt de TeX,
# isto e, 252/72,27 = 3,487 in. Desenhando neste tamanho e incluindo com
# width=\columnwidth, a escala e 1:1 e os corpos de texto abaixo sao os corpos
# que o leitor ve. Por isso NAO se usa bbox_inches="tight" aqui: o recorte
# automatico mudaria a altura final e quebraria o orcamento de pagina.
COL_W_IN = 252.0 / 72.27
FIG_H_IN = 1.62          # teto acordado com o enquadramento de 4+1 paginas

# --- paleta: os mesmos slots validados da figura 1 ---------------------------
# Reaproveitados por coerencia entre as figuras do artigo. A validacao (faixa de
# luminosidade, piso de croma, separacao sob deficiencia de visao de cor e
# contraste contra a superficie) esta registrada em plot_partstats_fig1.py e nao
# se repete aqui. Cada entidade tem slot fixo; a cor segue a entidade.
C_BAL = "#25599f"   # azul-aco   — SNP-AV1, calibracao equilibrada
C_AGG = "#4a3aa7"   # violeta    — SNP-AV1, calibracao agressiva

SURFACE = "#ffffff"   # o papel do artigo e branco: a figura deve fundir-se nele
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
KNOB = "#8a8880"   # botao de limiar do primeiro estagio: elemento de apoio
DETAIL = "#f6f5f0"  # fundo do quadro de detalhe

# Marcadores distintos por serie, e nao apenas cores: em impressao em cinza a
# forma continua separando as duas series sem exigir uma variante texturizada.
M_BAL, M_AGG = "o", "^"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CSV = os.path.join(ROOT, "results", "benchmark", "fase6",
                           "bdrate_average.csv")

# (chave no CSV, rotulo curto para o CSV de auditoria)
BALANCED = [("ml_balanced", "balanced, first stage only"),
            ("ml_bal_h9d", "balanced, deployed"),
            ("ml_bal_h9d_pl20", "balanced, strong stage 2")]
AGGRESSIVE = [("ml_aggr", "aggressive, first stage only"),
              ("ml_aggr_h9d", "aggressive, deployed"),
              ("ml_aggr_h9d_pl20", "aggressive, strong stage 2")]

# Valores publicados no texto do artigo, na precisao em que la aparecem. Servem
# de crivo: a figura so e gerada se o CSV ainda os reproduzir.
EXPECTED = {
    "ml_balanced": (0.568, 17.72),
    "ml_bal_h9d": (0.586, 18.74),
    "ml_bal_h9d_pl20": (0.651, 19.81),
    "ml_aggr": (1.403, 31.51),
    "ml_aggr_h9d": (1.409, 31.68),
    "ml_aggr_h9d_pl20": (1.420, 32.16),
}


def load(path):
    """Devolve {config: (bd_rate, ts_pct)} e confere contra o texto do artigo."""
    if not os.path.exists(path):
        sys.exit(f"ERRO: artefato canonico ausente: {path}")
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["config"]] = (float(row["bd_rate"]), float(row["ts_pct"]))

    faltando = [k for k in EXPECTED if k not in data]
    if faltando:
        sys.exit(f"ERRO: configuracoes ausentes no CSV: {', '.join(faltando)}")

    for k, (bd_txt, ts_txt) in EXPECTED.items():
        bd, ts = data[k]
        if round(bd, 3) != bd_txt or round(ts, 2) != ts_txt:
            sys.exit(f"ERRO: {k} divergiu do texto do artigo. "
                     f"CSV = {bd:.4f}% / {ts:.4f}%, "
                     f"artigo = {bd_txt}% / {ts_txt}%. "
                     f"Reconcilie antes de regerar a figura.")
    print("  conferido: as 6 configuracoes reproduzem os valores do artigo")
    return data


def preco(p_ini, p_fim):
    """Taxa de cambio de um segmento: pp de taxa BD por pp de tempo poupado."""
    return (p_fim[0] - p_ini[0]) / (p_fim[1] - p_ini[1])


def angulo_aparente(slope, ax_w_in, ax_h_in, x_span, y_span):
    """Angulo, em graus, com que uma inclinacao em unidades de dado aparece na
    tela. Sem esta conversao, um rotulo rotacionado pela inclinacao nominal
    descola da reta que deveria acompanhar."""
    return math.degrees(math.atan(slope * (ax_h_in / y_span) /
                                  (ax_w_in / x_span)))


def inset(fig, ax, bal, precos):
    """Detalhe da familia equilibrada, onde a comparacao de precos acontece.

    Os dois saltos do segundo estagio medem de 1,0 a 2,1 pontos percentuais de
    tempo, contra os 13,8 que separam as duas calibracoes do primeiro estagio.
    Na escala do grafico principal eles sao um borrao, e e justamente a
    inclinacao deles que carrega o argumento do artigo. O detalhe resolve isso
    sem exigir uma segunda figura: mesma familia, mesma cor, mesmos marcadores.
    """
    ix0, ix1 = 17.32, 20.42
    iy0, iy1 = 0.532, 0.738
    axi = ax.inset_axes([0.035, 0.505, 0.375, 0.440])
    # Fundo levemente distinto do papel: sinaliza que o quadro e um destaque, e
    # nao um segundo grafico independente, sem precisar de moldura fechada.
    axi.set_facecolor(DETAIL)
    axi.grid(True, color=GRID, linewidth=0.4, zorder=1)
    axi.set_axisbelow(True)

    # A reta do botao de limiar, redesenhada a partir da MESMA base do primeiro
    # estagio: as duas alternativas partem do mesmo ponto de operacao, que e a
    # unica leitura em que a comparacao de precos e legitima.
    dx = ix1 - bal[0][1]
    axi.plot([bal[0][1], ix1], [bal[0][0], bal[0][0] + precos["knob"] * dx],
             color=KNOB, linewidth=0.9, linestyle=(0, (4, 2.2)), zorder=2)
    axi.plot([p[1] for p in bal], [p[0] for p in bal], color=C_BAL,
             linewidth=1.0, zorder=4)
    axi.plot([p[1] for p in bal], [p[0] for p in bal], linestyle="none",
             marker=M_BAL, markersize=2.8, color=C_BAL,
             markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=5)

    w_in, h_in = 0.375 * 0.875 * COL_W_IN, 0.440 * 0.765 * FIG_H_IN
    a_knob = angulo_aparente(precos["knob"], w_in, h_in, ix1 - ix0, iy1 - iy0)
    a_st2 = angulo_aparente(precos["stage2"], w_in, h_in, ix1 - ix0, iy1 - iy0)

    axi.text(18.85, bal[0][0] + precos["knob"] * (18.85 - bal[0][1]) + 0.007,
             f"threshold knob, {precos['knob']:.3f}", fontsize=4.8,
             color=INK_2, ha="center", va="bottom", rotation=a_knob,
             rotation_mode="anchor", zorder=6)
    axi.text(18.23, 0.5767 - 0.005,
             f"second stage, {precos['stage2']:.3f}", fontsize=4.8,
             color=C_BAL, ha="center", va="top", rotation=a_st2,
             rotation_mode="anchor", zorder=6)
    # Sem titulo dentro do detalhe: a legenda da figura ja o identifica, e cada
    # linha de texto a mais nesta area disputa espaco com os proprios dados.

    axi.set_xlim(ix0, ix1)
    axi.set_ylim(iy0, iy1)
    axi.set_xticks([18, 19, 20])
    axi.set_yticks([0.6, 0.7])
    axi.tick_params(colors=MUTED, labelsize=4.8, length=0, pad=1.0)
    for lbl in axi.get_xticklabels() + axi.get_yticklabels():
        lbl.set_color(INK)
    for side in ("top", "right"):
        axi.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axi.spines[side].set_color(MUTED)
        axi.spines[side].set_linewidth(0.5)


def draw(data, out_path, precos):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    # Margens em fracao da figura, fixadas a mao porque o tamanho final e o
    # tamanho desenhado. Sobra o necessario para rotulo de eixo e marcas.
    ax = fig.add_axes([0.115, 0.205, 0.875, 0.765])
    ax.set_facecolor(SURFACE)

    bal = [data[k] for k, _ in BALANCED]
    agg = [data[k] for k, _ in AGGRESSIVE]

    ax.grid(True, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Botao de limiar do primeiro estagio: o segmento entre as duas bases. E a
    # unica alternativa disponivel a quem quiser comprar mais tempo sem o
    # segundo estagio, e a sua inclinacao e a referencia contra a qual a do
    # segundo estagio deve ser lida.
    ax.plot([bal[0][1], agg[0][1]], [bal[0][0], agg[0][0]], color=KNOB,
            linewidth=0.9, linestyle=(0, (4, 2.2)), zorder=2)

    # Ancora: e um ponto de operacao real, o denominador de todo o resto.
    ax.plot([0], [0], marker="P", markersize=3.8, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=6)
    ax.text(0.9, 0.035, "anchor, cpu-used=0", fontsize=5.8, color=INK_2,
            va="bottom", ha="left", zorder=6)

    for pts, cor, marca in ((bal, C_BAL, M_BAL), (agg, C_AGG, M_AGG)):
        ax.plot([p[1] for p in pts], [p[0] for p in pts], color=cor,
                linewidth=1.0, zorder=4)
        ax.plot([p[1] for p in pts], [p[0] for p in pts], linestyle="none",
                marker=marca, markersize=3.2, color=cor,
                markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=5)

    # Rotulo direto de cada familia, no lugar de uma legenda: em pouco mais de
    # 1,6 in de altura a legenda consome espaco que os proprios dados precisam,
    # e o rotulo junto ao dado dispensa a correspondencia por cor. Em duas
    # linhas para nao invadir a moldura nem a reta do botao de limiar.
    ax.text(bal[1][1] + 0.3, bal[1][0] - 0.135, "balanced\ncalibration",
            fontsize=6.0, color=C_BAL, ha="center", va="top",
            linespacing=1.15, zorder=6)
    ax.text(32.2, agg[1][0] - 0.115, "aggressive\ncalibration",
            fontsize=6.0, color=C_AGG, ha="center", va="top",
            linespacing=1.15, zorder=6)

    # Rotulo da reta do botao de limiar, alinhado a ela. O angulo e o angulo
    # aparente na tela, e nao a inclinacao em unidades de dado: depende da
    # razao entre as escalas dos dois eixos, que estao fixadas logo abaixo.
    ang = angulo_aparente(precos["knob"], ax_w_in=0.875 * COL_W_IN,
                          ax_h_in=0.765 * FIG_H_IN,
                          x_span=36.8, y_span=1.795)
    x_rot = 25.0
    y_rot = bal[0][0] + precos["knob"] * (x_rot - bal[0][1])
    ax.text(x_rot, y_rot + 0.035, "first-stage threshold knob",
            fontsize=5.8, color=INK_2, ha="center", va="bottom",
            rotation=ang, rotation_mode="anchor", zorder=6)

    inset(fig, ax, bal, precos)

    ax.set_xlim(-1.2, 35.6)
    ax.set_ylim(-0.075, 1.72)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    ax.set_yticks([0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Time savings (%)", fontsize=7.0, color=INK_2, labelpad=1.5)
    ax.set_ylabel("BD-BR (%)", fontsize=7.0, color=INK_2, labelpad=1.5)
    ax.tick_params(colors=MUTED, labelsize=6.5, length=0, pad=1.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)

    fig.savefig(out_path, facecolor=SURFACE)
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=SURFACE, dpi=400)
    plt.close(fig)
    print(f"  gravado: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir",
                    default=os.path.join(ROOT, "results", "thesis", "figuras"))
    args = ap.parse_args()

    print(f"Lendo {args.csv}")
    data = load(args.csv)

    # As duas taxas de cambio do artigo, recalculadas aqui a partir dos mesmos
    # pontos que a figura plota. O estimador e unico e declarado: media das
    # medias sobre as oito sequencias da CTC, que e o que o CSV canonico traz.
    # NAO se mistura com o estimador de interpolacao por sequencia (0,063),
    # usado apenas no teste de nao dominancia sequencia a sequencia.
    precos = {
        "knob": preco(data["ml_balanced"], data["ml_aggr"]),
        "stage2": preco(data["ml_balanced"], data["ml_bal_h9d"]),
        "stage2_pl20": preco(data["ml_balanced"], data["ml_bal_h9d_pl20"]),
    }
    print(f"  botao de limiar do 1o estagio : {precos['knob']:.4f} pp/pp")
    print(f"  2o estagio, calibracao PL10   : {precos['stage2']:.4f} pp/pp"
          f"  ({precos['knob'] / precos['stage2']:.2f}x mais barato)")
    print(f"  2o estagio, calibracao PL20   : {precos['stage2_pl20']:.4f} pp/pp"
          f"  ({precos['knob'] / precos['stage2_pl20']:.2f}x mais barato)")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "figura3_espaco_operacao")
    draw(data, base + ".pdf", precos)

    csv_path = os.path.join(args.out_dir, "figura3_dados.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "point", "config", "bd_rate_pct", "ts_pct"])
        for serie, itens in (("SNP-AV1 balanced", BALANCED),
                             ("SNP-AV1 aggressive", AGGRESSIVE)):
            for chave, rotulo in itens:
                bd, ts = data[chave]
                w.writerow([serie, rotulo, chave, f"{bd:.4f}", f"{ts:.4f}"])
        w.writerow(["anchor", "libaom v3.10.0 cpu-used=0", "anchor",
                    "0.0000", "0.0000"])
    print(f"  gravado: {csv_path}")

    print(f"\nDimensoes finais: {COL_W_IN:.3f} x {FIG_H_IN:.2f} in, "
          f"para \\includegraphics[width=\\columnwidth]{{figura3_espaco_operacao}}")


if __name__ == "__main__":
    main()
