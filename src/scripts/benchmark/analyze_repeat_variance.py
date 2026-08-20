#!/usr/bin/env python3
"""Dispersao de medicao entre repeticoes da campanha de tempo de codificacao.

Le o CSV bruto de uma campanha em que a mesma configuracao foi codificada
varias vezes (sufixo _r<N> no nome da configuracao) e reporta duas coisas:

  1. a dispersao do tempo de parede por configuracao e ponto de qualidade,
     como coeficiente de variacao;
  2. a dispersao do TS derivado, que e a grandeza efetivamente reportada no
     artigo, calculada pela Eq. (1) -- razao por ponto de qualidade antes de
     qualquer media.

O TS e calculado de duas formas. Na forma pareada, a repeticao i da
configuracao e comparada com a repeticao i da ancora, que e o pareamento
natural quando as repeticoes correm intercaladas e partilham a deriva de
carga da maquina. Na forma cruzada, todas as combinacoes de repeticoes sao
percorridas, o que remove a hipotese de pareamento e da o limite superior
pessimista da dispersao.

Uso (dentro do conteiner):
    python3 src/scripts/benchmark/analyze_repeat_variance.py \
        results/benchmark/fase6_repeat/raw_results.csv --anchor anchor
"""

import argparse
import csv
import itertools
import re
import statistics
import sys
from collections import defaultdict

SUFIXO_REP = re.compile(r"_r(\d+)$")


def carrega(caminho):
    """Devolve tempos[(base, cq, rep)] = segundos, a partir do CSV bruto."""
    tempos = {}
    with open(caminho, newline="") as fp:
        for linha in csv.DictReader(fp):
            m = SUFIXO_REP.search(linha["config"])
            if not m:
                continue  # linha sem marca de repeticao: fora deste estudo
            base = SUFIXO_REP.sub("", linha["config"])
            chave = (base, linha["cq"], m.group(1))
            tempos[chave] = float(linha["time_s"])
    return tempos


def ts(tempos, cfg, ancora, rep_cfg, rep_anc, cqs):
    """TS da Eq. (1) para uma sequencia: media das razoes por ponto de CQ."""
    parcelas = [
        1.0 - tempos[(cfg, q, rep_cfg)] / tempos[(ancora, q, rep_anc)] for q in cqs
    ]
    return 100.0 * statistics.fmean(parcelas)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="raw_results.csv da campanha de repeticao")
    ap.add_argument("--anchor", default="anchor", help="nome base da ancora")
    args = ap.parse_args()

    tempos = carrega(args.csv)
    if not tempos:
        sys.exit("nenhuma configuracao com sufixo _r<N> no CSV")

    bases = sorted({b for b, _, _ in tempos})
    cqs = sorted({q for _, q, _ in tempos}, key=int)
    reps = sorted({r for _, _, r in tempos}, key=int)
    if args.anchor not in bases:
        sys.exit(f"ancora '{args.anchor}' ausente; bases: {bases}")

    print(f"repeticoes: {len(reps)}   pontos de qualidade: {', '.join(cqs)}\n")

    print("dispersao do tempo de parede")
    print(f"{'config':<16}{'cq':>5}{'media_s':>11}{'dp_s':>9}{'cv%':>8}")
    for base in bases:
        for q in cqs:
            v = [tempos[(base, q, r)] for r in reps]
            m, dp = statistics.fmean(v), statistics.stdev(v)
            print(f"{base:<16}{q:>5}{m:>11.2f}{dp:>9.3f}{100 * dp / m:>8.3f}")

    print("\ndispersao do TS derivado (Eq. 1)")
    print(f"{'config':<16}{'modo':>9}{'media%':>10}{'dp_pp':>9}{'min%':>9}{'max%':>9}")
    for base in bases:
        if base == args.anchor:
            continue
        pareado = [ts(tempos, base, args.anchor, r, r, cqs) for r in reps]
        cruzado = [
            ts(tempos, base, args.anchor, a, b, cqs)
            for a, b in itertools.product(reps, reps)
        ]
        for nome, v in (("pareado", pareado), ("cruzado", cruzado)):
            print(
                f"{base:<16}{nome:>9}{statistics.fmean(v):>10.3f}"
                f"{statistics.stdev(v):>9.3f}{min(v):>9.3f}{max(v):>9.3f}"
            )


if __name__ == "__main__":
    main()
