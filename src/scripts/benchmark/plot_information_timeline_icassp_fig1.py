#!/usr/bin/env python3
"""Figura 1 do artigo ICASSP 2027 — linha do tempo de disponibilidade de
informacao dentro da busca de um no da arvore de particionamento do AV1.

Como a figura 2 do LASCAS, esta NAO plota medicao: e um diagrama estrutural.
Por isso nao le artefato numerico, e sim declara em SPEC os fatos que desenha,
cada um com a sua procedencia, para que uma mudanca de instrumentacao ou de
composicao do vetor obrigue a mexer aqui e nao passe despercebida.

COMPOSICAO NOVA, e nao a da figura 2 do LASCAS. Aquela e uma matriz de cobertura
de acoes de poda; esta e uma linha do tempo de DISPONIBILIDADE. O eixo horizontal
e o percurso da busca dentro de um unico no, com tres instantes marcados, e cada
faixa e um item de informacao que comeca exatamente no instante em que passa a
existir:

  t0  antes da busca do no        — luminancia, quantizacao, posicao e a
                                    vizinhanca causal ja decidida;
  t1  depois de PARTITION_NONE    — taxa, distorcao e custo RD do candidato
                                    indiviso;
  t2  depois da recursao          — custo RD da subarvore otima.

E o alinhamento vertical que faz o argumento do artigo sem precisar de seta: as
tres faixas dos blocos A, B e C comecam todas em t0, ou seja, o podador pre-busca
as recebe de graca; a faixa do custo de NONE so comeca em t1, o que a torna
estruturalmente indisponivel a ele; e a faixa da subarvore otima, que e o alvo
que a perda de otimalidade mede, so comeca em t2.

Duas paletas, MESMA GEOMETRIA, como no gerador da figura 2 do LASCAS:
  cor   — acento azul-aco para o instante atacado;
  cinza — preto e escalas de cinza, para impressao monocromatica; o instante
          atacado deixa de ser separado por matiz e passa a se-lo pelo tracado
          cheio contra o pontilhado dos demais.

Uso (dentro do conteiner):
    build/venv-ml/bin/python \
        src/scripts/benchmark/plot_information_timeline_icassp_fig1.py \
        --out-dir results/thesis/figuras

Saidas: figura1_linha_do_tempo.pdf       (variante de cor)
        figura1_linha_do_tempo_cinza.pdf (variante monocromatica)
        e um .png de cada, para conferencia visual.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- tipografia: identica a das figuras do LASCAS ----------------------------
# O IEEEtran compoe em Times; STIXGeneral e a serif de metrica Times que
# acompanha o matplotlib. fonttype 42 embute como TrueType, pois o Type 3
# padrao costuma ser recusado pelo PDF eXpress do IEEE.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_W_IN = 252.0 / 72.27   # \columnwidth do IEEEtran [conference], em polegadas
FIG_H_IN = 1.34            # reserva no .tex era 2,8 cm; conferir paginacao

PALETAS = {
    "cor": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#c9c7be", regua_leve="#e3e1d9",
        alvo="#25599f",                       # o instante que o artigo ataca
        barra_livre="#c2d0e8", borda_livre="#25599f",
        barra_tarde="#dcdad2", borda_tarde="#a8a7a1",
    ),
    "cinza": dict(
        surface="#ffffff", ink="#0b0b0b", ink_2="#3f3e3b", muted="#6e6d68",
        regua="#bcbab3", regua_leve="#e0ded7",
        alvo="#141413",
        barra_livre="#b5b3ac", borda_livre="#141413",
        barra_tarde="#e3e1db", borda_tarde="#a8a7a1",
    ),
}

# --- os fatos desenhados, com procedencia ------------------------------------
# Ordem de avaliacao e instantes: Secao II do artigo, conferida contra
# av1_rd_pick_partition() em src/aom/av1/encoder/partition_strategy.c.
# Composicao e contagem de colunas dos blocos: Secao III-A, espelhando
# src/scripts/partition_model/features.py (FEATURE_NAMES, RPP_SUBSETS).
# O bloco de taxa/distorcao/custo de NONE existe no vetor completo da tese
# (indices 38..40) e NAO e consumido por nenhuma representacao deste artigo:
# aparece aqui para mostrar POR QUE nao poderia ser.
SPEC = {
    # O rotulo vai como texto simples: o matplotlib nao e LaTeX, e escapar o
    # sublinhado imprimiria a propria barra invertida.
    "instantes": [
        ("before the search", 0.055),
        ("after NONE",        0.520),
        ("after recursion",   0.855),
    ],
    # (rotulo, indice do instante em que a informacao passa a existir)
    "faixas": [
        ("block A, 24 columns", 0),
        ("block B, 8 columns",  0),
        ("block C, 4 columns",  0),
        ("RD cost of NONE",     1),
        ("RD cost of the best partition", 2),
    ],
}

X0, X1 = 0.372, 0.988      # extensao horizontal da regiao de instantes
Y_EIXO = 0.845             # linha do tempo
Y_TOPO_FAIXA = 0.700       # centro da primeira faixa
PASSO = 0.140              # de centro a centro de faixas consecutivas
FAIXA_H = 0.072


def draw(out_path, p):
    fig = plt.figure(figsize=(COL_W_IN, FIG_H_IN), dpi=400)
    fig.patch.set_facecolor(p["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(p["surface"])

    xs = [X0 + t * (X1 - X0) for _, t in SPEC["instantes"]]

    # ---------------- eixo do tempo dentro do no ----------------------------
    ax.text(0.010, Y_EIXO + 0.030, "search of", ha="left", va="center",
            fontsize=6.2, color=p["ink_2"], zorder=4)
    ax.text(0.010, Y_EIXO - 0.032, "one node", ha="left", va="center",
            fontsize=6.2, color=p["ink_2"], zorder=4)
    ax.annotate("", xy=(X1 + 0.010, Y_EIXO), xytext=(0.140, Y_EIXO),
                arrowprops=dict(arrowstyle="-|>", color=p["muted"],
                                linewidth=0.55, shrinkA=0, shrinkB=0))

    # Os tres instantes. O primeiro e o que o artigo ataca: tracado cheio e
    # acento; os outros dois, pontilhados e neutros. Na variante monocromatica
    # e essa diferenca de tracado, e nao o matiz, que carrega a distincao.
    for i, ((rotulo, _), x) in enumerate(zip(SPEC["instantes"], xs)):
        alvo = (i == 0)
        cor = p["alvo"] if alvo else p["muted"]
        y_base = Y_TOPO_FAIXA - (len(SPEC["faixas"]) - 1) * PASSO - 0.062
        ax.plot([x, x], [y_base, Y_EIXO], color=cor,
                linewidth=0.85 if alvo else 0.45, zorder=3,
                linestyle="-" if alvo else (0, (1.4, 1.4)))
        ax.plot([x], [Y_EIXO], marker="o", markersize=3.0 if alvo else 2.0,
                color=cor, markeredgecolor="none", zorder=5, clip_on=False)
        ax.text(x, Y_EIXO + 0.052, rotulo, ha="center", va="bottom",
                fontsize=6.2, color=p["ink"] if alvo else p["ink_2"],
                weight="bold" if alvo else "normal", zorder=6)

    # ---------------- faixas de disponibilidade -----------------------------
    for k, (rotulo, inst) in enumerate(SPEC["faixas"]):
        y = Y_TOPO_FAIXA - k * PASSO
        cedo = (inst == 0)
        ax.text(0.010, y, rotulo, ha="left", va="center", fontsize=6.2,
                color=p["ink"] if cedo else p["ink_2"], zorder=4)
        # trilho apagado antes de existir, barra cheia a partir do instante
        ax.plot([X0, X1], [y, y], color=p["regua_leve"], linewidth=0.4,
                zorder=1, solid_capstyle="butt")
        ax.add_patch(plt.Rectangle(
            (xs[inst], y - FAIXA_H / 2), X1 - xs[inst], FAIXA_H,
            facecolor=p["barra_livre"] if cedo else p["barra_tarde"],
            edgecolor=p["borda_livre"] if cedo else p["borda_tarde"],
            linewidth=0.4, zorder=2))

    # ---------------- nota de leitura ---------------------------------------
    ax.text(0.010, 0.040,
            "each band is shaded from the instant the encoder holds it",
            ha="left", va="center", fontsize=5.8, color=p["ink_2"], zorder=4)

    fig.savefig(out_path, facecolor=p["surface"])
    if out_path.endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", facecolor=p["surface"], dpi=400)
    plt.close(fig)
    print("  gravado: %s" % out_path)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    ap.add_argument("--out-dir",
                    default=os.path.join(root, "results", "thesis", "figuras"))
    ap.add_argument("--variante", choices=sorted(PALETAS) + ["ambas"],
                    default="ambas")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sufixo = {"cor": "", "cinza": "_cinza"}
    alvos = sorted(PALETAS) if args.variante == "ambas" else [args.variante]
    for nome in alvos:
        draw(os.path.join(args.out_dir,
                          "figura1_linha_do_tempo%s.pdf" % sufixo[nome]),
             PALETAS[nome])
    print("\nDimensoes finais: %.3f x %.2f in, para "
          "\\includegraphics[width=\\columnwidth]{figura1_linha_do_tempo}"
          % (COL_W_IN, FIG_H_IN))


if __name__ == "__main__":
    main()
