#!/usr/bin/env python3
"""Figura 3 do artigo LASCAS 2027 — espaco de operacao do SNP-AV1 na CTC A1.

Le DIRETAMENTE o artefato canonico `results/benchmark/fase6/bdrate_average.csv`,
que e a mesma fonte de onde cada numero do texto do artigo foi transcrito, de
modo que figura e texto nao possam divergir. Os valores lidos sao conferidos
contra os do artigo antes de plotar; qualquer divergencia aborta a execucao.

Eixo x: reducao de tempo (TS, definicao canonica da M3, media por ponto de
quantizacao e depois por sequencia). Eixo y: taxa BD sobre PSNR-Y. Mais a
direita e mais abaixo e melhor. A ancora — o libaom v3.10.0 intocado em
`cpu-used=0` — e a origem dos dois eixos e fica FORA da faixa desenhada: nao ha
ponto medido abaixo de 17,7% de reducao de tempo, e comecar o eixo em 10% devolve
a area do desenho aos dados. A legenda do artigo declara a ancora.

ESCOPO (decisao editorial de 2026-09-04, orientacao): a figura NAO apresenta a
escada de presets nativos. O SNP-AV1 age apenas dentro da busca de particao,
enquanto um preset reconfigura todas as etapas do codificador; po-los lado a lado
compara arranjos de escopo distinto. O eixo de comparacao desta figura e interno:
os seis pontos medidos e as duas taxas de cambio entre tempo e taxa BD que a
geometria deles expoe.

    - o segmento tracejado liga as duas BASES (apenas o primeiro estagio) e e,
      portanto, o botao de limiar do primeiro estagio: a unica alternativa que a
      solucao implantada oferece ao usuario para comprar mais tempo;
    - o primeiro salto dentro de cada familia e o segundo estagio na calibracao
      implantada, e a sua inclinacao e visivelmente mais suave.

Os dois saltos do segundo estagio medem de 1,0 a 2,1 pontos percentuais de tempo,
contra os 13,8 que separam as duas calibracoes do primeiro estagio. Na escala do
grafico principal eles sao um borrao, e e justamente a inclinacao deles que
carrega o argumento do artigo; o quadro de detalhe resolve isso sem exigir uma
segunda figura. As duas inclinacoes anotadas sao calculadas no proprio script, a
partir dos mesmos pontos plotados, e impressas no terminal para auditoria.

TIPOGRAFIA: piso de 9 pt para TODO texto da figura — eixo, marcacoes, rotulos de
serie e anotacoes. Como a figura e desenhada na largura de coluna exata do
IEEEtran e incluida com `width=\\columnwidth`, a escala e 1:1 e os 9 pt
declarados aqui sao 9 pt medidos na pagina. O piso e o que fixa a altura: em
1,6 in nao cabem rotulos deste corpo sem colisao.

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
# width=\columnwidth, a escala e 1:1 e os corpos declarados abaixo sao os corpos
# que o leitor ve. Por isso NAO se usa bbox_inches="tight" aqui: o recorte
# automatico mudaria a altura final e quebraria o orcamento de pagina.
COL_W_IN = 252.0 / 72.27
FIG_H_IN = 170.0 / 72.0    # altura pedida pelo piso de 9 pt, ver docstring
PT = 9.0                   # piso tipografico: nada abaixo disto na figura

# Margens do eixo, em pontos PostScript, dimensionadas para o texto de 9 pt.
MARG_ESQ_PT = 33.0         # rotulo do eixo y + marcacoes "0.5" + folga
MARG_INF_PT = 30.0         # marcacoes + rotulo do eixo x + folga
MARG_DIR_PT = 4.0
MARG_SUP_PT = 15.0        # titulo de cada janela

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
KNOB = "#8a8880"    # botao de limiar do primeiro estagio: elemento de apoio
DETAIL = "#f6f5f0"  # fundo do quadro de detalhe

# Marcadores distintos por serie, e nao apenas cores: em impressao em cinza a
# forma continua separando as duas series sem exigir uma variante texturizada.
M_BAL, M_AGG = "o", "^"
MS = 5.0            # corpo do marcador, proporcional ao texto de 9 pt
LW = 1.5

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


# --- faixas dos eixos --------------------------------------------------------
# O eixo do tempo e QUEBRADO em duas janelas. Os seis pontos medidos vivem em
# dois aglomerados separados por 12 pontos percentuais de tempo, e cada
# aglomerado tem largura interna de 2,1 (equilibrado) e 0,65 (agressivo) pontos
# percentuais. Num eixo continuo de 10% a 35%, os tres pontos de cada familia
# caem sobre o mesmo corpo de marcador e leem-se como um ponto so, que e
# justamente o que a figura nao pode esconder: a estrutura interna de cada
# familia E o argumento do artigo. Duas janelas ampliadas, com a quebra marcada,
# resolvem os seis pontos e mantem a relacao vertical entre as familias, uma vez
# que o eixo da taxa BD e comum e continuo.
#   A janela da esquerda e a que carrega a comparacao de precos: a reta do botao
# de limiar e o salto do segundo estagio partem do mesmo ponto de operacao e sao
# lidos na mesma escala. Por isso ela dispensa o quadro de detalhe da versao
# anterior.
XA_MIN, XA_MAX = 17.25, 20.45      # janela da calibracao equilibrada
XB_MIN, XB_MAX = 31.15, 32.60      # janela da calibracao agressiva
Y_MIN, Y_MAX = 0.42, 1.58

# Reparticao da area de desenho entre as duas janelas, em fracao da largura
# util. A janela da esquerda leva mais porque cobre 3,2 pontos percentuais
# contra 1,45, e porque e nela que ficam as duas retas anotadas.
# A folga entre as janelas, em fracao da largura util; a reparticao entre elas
# nao e escolhida, e derivada das faixas dentro de draw(), para que a escala do
# eixo do tempo seja a mesma nos dois lados.
FOLGA = 0.055
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


def angulo(slope, w_in, h_in, x_span, y_span):
    """Angulo, em graus, com que uma inclinacao em unidades de dado aparece na
    tela. Sem esta conversao, um rotulo rotacionado pela inclinacao nominal
    descola da reta que deveria acompanhar."""
    return math.degrees(math.atan(slope * (h_in / y_span) / (w_in / x_span)))


def estilo_eixo(ax):
    ax.tick_params(colors=MUTED, labelsize=PT, length=0, pad=3.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.7)




def marca_quebra(ax_a, ax_b):
    """Marcas de quebra nas bordas que se encaram, no estilo usual: dois tracos
    inclinados sobre a linha de base de cada janela. Sem elas o leitor pode tomar
    as duas janelas por um eixo continuo e ler a distancia entre as familias
    errado."""
    d, y = 0.018, 0.0
    for ax, xs in ((ax_a, (1.0,)), (ax_b, (0.0,))):
        for x in xs:
            for dy in (0,):
                ax.plot([x - d, x + d], [y - 2.2 * d + dy, y + 2.2 * d + dy],
                        transform=ax.transAxes, color=MUTED, linewidth=0.8,
                        clip_on=False, zorder=8)


def serie(ax, pts, cor, marca):
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color=cor,
            linewidth=LW, zorder=4)
    ax.plot([p[1] for p in pts], [p[0] for p in pts], linestyle="none",
            marker=marca, markersize=MS, color=cor, markeredgecolor=SURFACE,
            markeredgewidth=0.7, zorder=5)


def draw(data, out_path, precos):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(SURFACE)

    w_pt, h_pt = COL_W_IN * 72.0, FIG_H_IN * 72.0
    x0 = MARG_ESQ_PT / w_pt
    y0 = MARG_INF_PT / h_pt
    w = 1.0 - (MARG_ESQ_PT + MARG_DIR_PT) / w_pt
    h = 1.0 - (MARG_INF_PT + MARG_SUP_PT) / h_pt
    # As duas janelas partilham a MESMA escala do eixo do tempo: a largura de
    # cada uma sai da sua propria faixa. Sem isso, um ponto percentual de tempo
    # mediria coisas diferentes nos dois lados e a dispersao interna das duas
    # familias deixaria de ser comparavel a olho, que e metade do que a figura
    # precisa mostrar.
    span_a, span_b = XA_MAX - XA_MIN, XB_MAX - XB_MIN
    util = w * (1.0 - FOLGA)
    w_a = util * span_a / (span_a + span_b)
    w_b = util - w_a
    ax_a = fig.add_axes([x0, y0, w_a, h])
    ax_b = fig.add_axes([x0 + w_a + w * FOLGA, y0, w_b, h])

    bal = [data[k] for k, _ in BALANCED]
    agg = [data[k] for k, _ in AGGRESSIVE]

    for ax in (ax_a, ax_b):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=1)
        ax.set_axisbelow(True)
        ax.set_ylim(Y_MIN, Y_MAX)
        # Botao de limiar do primeiro estagio: o segmento entre as duas BASES,
        # desenhado inteiro nas duas janelas e recortado por elas. E a unica
        # alternativa disponivel a quem quiser comprar mais tempo sem o segundo
        # estagio, e e a referencia contra a qual o preco do segundo estagio
        # deve ser lido. Atravessar a quebra e o que amarra as duas janelas.
        ax.plot([bal[0][1], agg[0][1]], [bal[0][0], agg[0][0]], color=KNOB,
                linewidth=1.3, linestyle=(0, (4, 2.2)), zorder=2)

    ax_a.set_xlim(XA_MIN, XA_MAX)
    ax_b.set_xlim(XB_MIN, XB_MAX)
    serie(ax_a, bal, C_BAL, M_BAL)
    serie(ax_b, agg, C_AGG, M_AGG)

    ax_a.set_xticks([18, 19, 20])
    ax_b.set_xticks([31.5, 32.5])
    ax_a.set_yticks([0.5, 1.0, 1.5])
    ax_b.set_yticks([0.5, 1.0, 1.5])
    estilo_eixo(ax_a)
    estilo_eixo(ax_b)
    # A janela da direita nao repete os rotulos da taxa BD: o eixo e o mesmo, e
    # repeti-los sugeriria duas escalas verticais distintas.
    ax_b.set_yticklabels([])
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    marca_quebra(ax_a, ax_b)

    ax_a.set_ylabel("BD-BR (%)", fontsize=PT, color=INK_2, labelpad=2.0)
    # Um unico rotulo de eixo x, centrado sobre as duas janelas: a grandeza e a
    # mesma, so a faixa muda.
    fig.text(x0 + w / 2.0, 1.5 / h_pt, "Time savings (%)", fontsize=PT,
             color=INK_2, ha="center", va="bottom")

    # Cada janela E uma calibracao, e por isso o nome da serie vai no topo da
    # janela, e nao flutuando ao lado dos pontos: nao ha o que confundir dentro
    # de cada uma, e o rotulo deixa de disputar espaco com os dados e com as
    # duas retas anotadas. A legenda da figura completa o nome.
    ax_a.set_title("balanced", fontsize=PT, color=C_BAL, pad=3.0)
    ax_b.set_title("aggressive", fontsize=PT, color=C_AGG, pad=3.0)

    # As duas taxas de cambio, anotadas na janela em que partem do mesmo ponto
    # de operacao e sao lidas na mesma escala. O angulo e o angulo aparente na
    # tela, e nao a inclinacao em unidades de dado.
    ax_w_in, ax_h_in = w_a * COL_W_IN, h * FIG_H_IN
    a_knob = angulo(precos["knob"], ax_w_in, ax_h_in, XA_MAX - XA_MIN,
                    Y_MAX - Y_MIN)
    a_st2 = angulo(precos["stage2"], ax_w_in, ax_h_in, XA_MAX - XA_MIN,
                   Y_MAX - Y_MIN)
    x_k = XA_MAX - 0.08
    ax_a.text(x_k, bal[0][0] + precos["knob"] * (x_k - bal[0][1]) + 0.020,
              f"threshold knob, {precos['knob']:.3f}", fontsize=PT,
              color=INK_2, ha="right", va="bottom", rotation=a_knob,
              rotation_mode="anchor", zorder=6)
    x_s = 18.52
    ax_a.text(x_s, bal[0][0] + precos["stage2"] * (x_s - bal[0][1]) - 0.022,
              f"second stage, {precos['stage2']:.3f}", fontsize=PT,
              color=C_BAL, ha="center", va="top", rotation=a_st2,
              rotation_mode="anchor", zorder=6)

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

    print(f"\nDimensoes finais: {COL_W_IN:.3f} x {FIG_H_IN:.2f} in, piso de "
          f"{PT:.0f} pt, para "
          f"\\includegraphics[width=\\columnwidth]{{figura3_espaco_operacao}}")


if __name__ == "__main__":
    main()
