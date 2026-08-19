#!/usr/bin/env python3
"""Figura 3 do artigo LASCAS 2027 — espaco de operacao medido na CTC Classe A1.

Le DIRETAMENTE o artefato canonico `results/benchmark/fase6/bdrate_average.csv`,
que e a mesma fonte de onde cada numero do texto do artigo foi transcrito, de
modo que figura e texto nao possam divergir. Os valores lidos sao conferidos
contra os do artigo antes de plotar; qualquer divergencia aborta a execucao.

Eixo x: reducao de tempo (TS, definicao canonica da M3, media por ponto de
quantizacao e depois por sequencia). Eixo y: taxa BD sobre PSNR-Y. Mais a
direita e mais abaixo e melhor. A origem e a ancora, o libaom v3.10.0 intocado
em `cpu-used=0`.

A faixa sombreada marca o intervalo de TS que a escada de presets nativa nao
cobre: entre a ancora e o primeiro degrau (`cpu-used=1`) nao existe ponto de
operacao nativo, e e ai que os seis pontos do SNP-AV1 caem.

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
FIG_H_IN = 1.40          # teto acordado com o enquadramento de 4+1 paginas

# --- paleta: os mesmos slots validados da figura 1 ---------------------------
# Reaproveitados por coerencia entre as figuras do artigo. A validacao (faixa de
# luminosidade, piso de croma, separacao sob deficiencia de visao de cor e
# contraste contra a superficie) esta registrada em plot_partstats_fig1.py e nao
# se repete aqui. Cada entidade tem slot fixo; a cor segue a entidade.
C_BAL   = "#25599f"   # azul-aco   — SNP-AV1, calibracao equilibrada
C_AGG   = "#4a3aa7"   # violeta    — SNP-AV1, calibracao agressiva
C_NAT   = "#9b2d4f"   # vinho      — escada de presets nativa

SURFACE = "#ffffff"   # o papel do artigo e branco: a figura deve fundir-se nele
INK     = "#0b0b0b"
INK_2   = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
BAND    = "#f2f0e9"   # faixa nao coberta pela escada de presets

# Marcadores distintos por serie, e nao apenas cores: em impressao em cinza a
# forma continua separando as tres series sem exigir uma variante texturizada.
M_BAL, M_AGG, M_NAT = "o", "^", "s"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CSV = os.path.join(ROOT, "results", "benchmark", "fase6",
                           "bdrate_average.csv")

# (chave no CSV, rotulo curto para o CSV de auditoria)
BALANCED = [("ml_balanced",       "balanced, first stage only"),
            ("ml_bal_h9d",        "balanced, deployed"),
            ("ml_bal_h9d_pl20",   "balanced, strong stage 2")]
AGGRESSIVE = [("ml_aggr",          "aggressive, first stage only"),
              ("ml_aggr_h9d",      "aggressive, deployed"),
              ("ml_aggr_h9d_pl20", "aggressive, strong stage 2")]
NATIVE = [("native_cpu1", "cpu-used=1"),
          ("native_cpu2", "cpu-used=2"),
          ("native_cpu3", "cpu-used=3")]

# Valores publicados no texto do artigo, na precisao em que la aparecem. Servem
# de crivo: a figura so e gerada se o CSV ainda os reproduzir.
EXPECTED = {
    "ml_balanced":       (0.568, 17.72),
    "ml_bal_h9d":        (0.586, 18.74),
    "ml_bal_h9d_pl20":   (0.651, 19.81),
    "ml_aggr":           (1.403, 31.51),
    "ml_aggr_h9d":       (1.409, 31.68),
    "ml_aggr_h9d_pl20":  (1.420, 32.16),
    "native_cpu1":       (0.449, 32.59),
    "native_cpu2":       (0.536, 42.72),
    "native_cpu3":       (2.722, 67.94),
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
    print(f"  conferido: as 9 configuracoes reproduzem os valores do artigo")
    return data


def draw(data, out_path):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    # Margens em fracao da figura, fixadas a mao porque o tamanho final e o
    # tamanho desenhado. Sobra o necessario para rotulo de eixo e marcas.
    ax = fig.add_axes([0.115, 0.235, 0.875, 0.735])
    ax.set_facecolor(SURFACE)

    bal = [data[k] for k, _ in BALANCED]
    agg = [data[k] for k, _ in AGGRESSIVE]
    nat = [data[k] for k, _ in NATIVE]

    x_max, y_max = 74.0, 3.05
    primeiro_degrau = data["native_cpu1"][1]

    # Faixa que a escada nativa nao cobre: da ancora ao primeiro degrau.
    ax.axvspan(0, primeiro_degrau, color=BAND, zorder=0, linewidth=0)
    ax.text(primeiro_degrau / 2.0, 2.92, "no native preset in this range",
            ha="center", va="top", fontsize=5.8, color=MUTED, style="italic",
            zorder=2)

    ax.grid(True, color=GRID, linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Escada nativa. A ligacao vai apenas de um degrau ao seguinte, e NAO da
    # ancora ate o primeiro degrau: tracar aquele segmento sugeriria pontos de
    # operacao intermediarios que a escada nao oferece, que e justamente o
    # contrario do que a figura mostra. O salto fica por conta da faixa.
    ax.plot([p[1] for p in nat], [p[0] for p in nat], color=C_NAT,
            linewidth=0.9, linestyle="--", dash_capstyle="round", zorder=3)
    ax.plot([p[1] for p in nat], [p[0] for p in nat], linestyle="none",
            marker=M_NAT, markersize=3.2, color=C_NAT,
            markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=5)

    # Ancora: e um ponto de operacao real, o denominador de todo o resto.
    ax.plot([0], [0], marker="P", markersize=3.8, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=6)
    ax.text(1.8, 0.02, "anchor, cpu-used=0", fontsize=5.8, color=INK_2,
            va="bottom", ha="left", zorder=6)

    for pts, cor, marca in ((bal, C_BAL, M_BAL), (agg, C_AGG, M_AGG)):
        ax.plot([p[1] for p in pts], [p[0] for p in pts], color=cor,
                linewidth=1.0, zorder=4)
        ax.plot([p[1] for p in pts], [p[0] for p in pts], linestyle="none",
                marker=marca, markersize=3.2, color=cor,
                markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=5)

    # Rotulo direto de cada serie, no lugar de uma legenda: em 1,4 in de altura
    # a legenda consome espaco que os proprios dados precisam, e o rotulo junto
    # ao dado dispensa o leitor de fazer a correspondencia por cor.
    ax.text(18.8, 0.92, "SNP-AV1 balanced", fontsize=6.0, color=C_BAL,
            ha="center", va="bottom", zorder=6)
    ax.text(31.6, 1.55, "SNP-AV1 aggressive", fontsize=6.0, color=C_AGG,
            ha="center", va="bottom", zorder=6)
    ax.text(47.0, 1.95, "libaom presets", fontsize=6.0, color=C_NAT,
            ha="left", va="bottom", zorder=6)

    # Rotulo direto dos degraus: o valor de cpu-used ao lado do marcador.
    for chave, desloc in zip([k for k, _ in NATIVE],
                             ((1.0, 0.14), (1.0, 0.14), (-1.2, 0.12))):
        bd, ts = data[chave]
        ha = "right" if desloc[0] < 0 else "left"
        ax.text(ts + desloc[0], bd + desloc[1], chave[-1],
                fontsize=6.0, color=C_NAT, ha=ha, va="bottom", zorder=6)

    ax.set_xlim(-1.5, x_max)
    ax.set_ylim(-0.12, y_max)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_yticks([0, 1, 2, 3])
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

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "figura3_espaco_operacao")
    draw(data, base + ".pdf")

    csv_path = os.path.join(args.out_dir, "figura3_dados.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "point", "config", "bd_rate_pct", "ts_pct"])
        for serie, itens in (("SNP-AV1 balanced", BALANCED),
                             ("SNP-AV1 aggressive", AGGRESSIVE),
                             ("libaom presets", NATIVE)):
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
