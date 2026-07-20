#!/usr/bin/env python3
"""C1/C3 — decompoe o custo de busca por CANDIDATO de particao a partir de um ou
mais part_timing.csv (build com CONFIG_COLLECT_PARTITION_STATS=1).

Uso:
    analyze_partstats.py file1.csv [file2.csv ...] [--label seq1,seq2,...]

Com varios arquivos: imprime um resumo por-sequencia (share AB+4-way por pool) e
o AGREGADO global detalhado sobre o conjunto congelado de teste.

Pergunta (falsificacao da alavanca C3): AB (HORZ_A/B, VERT_A/B) + 4-way
(HORZ_4, VERT_4) somam >=10% do tempo de busca LOCAL do no? Se <10%, podar so
AB/4-way nao vale o esforco de engenharia (C3 morre sem experimento).

CUIDADO METODOLOGICO: o timer de PARTITION_SPLIT engloba a recursao
(partition_search.c:4552-4610), logo times[SPLIT] inclui todos os descendentes.
Somar a coluna SPLIT contaria o mesmo trabalho varias vezes. O tempo de trabalho
LOCAL de um no e a soma dos candidatos nao-recursivos: NONE+HORZ+VERT+AB+4way.
A coluna SPLIT e deliberadamente EXCLUIDA (o custo dela ja aparece nas linhas
dos nos-filhos).

Layout do CSV (sem cabecalho), por no de av1_rd_pick_partition:
  0 bsize, 1 frame, 2 frame_update_type, 3 mi_row, 4 mi_col,
  5 rate, 6 dist, 7 rdcost,
  8..17  decisions[0..9]
  18..27 attempts[0..9]
  28..37 times[0..9]   (microssegundos)
  38..47 rdcost[0..9]
Ordem EXT_PARTITION_TYPES: 0 NONE 1 HORZ 2 VERT 3 SPLIT 4 HORZ_A 5 HORZ_B
6 VERT_A 7 VERT_B 8 HORZ_4 9 VERT_4.
"""
import sys
from collections import defaultdict

T0 = 28  # primeira coluna de times[]
NAMES = ["NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B",
         "VERT_A", "VERT_B", "HORZ_4", "VERT_4"]
AB = [4, 5, 6, 7]
FOURWAY = [8, 9]
RECT = [1, 2]
LOCAL = [0, 1, 2, 4, 5, 6, 7, 8, 9]  # tudo menos SPLIT (3)

BSIZE = {0: "4x4", 1: "4x8", 2: "8x4", 3: "8x8", 4: "8x16", 5: "16x8",
         6: "16x16", 7: "16x32", 8: "32x16", 9: "32x32", 10: "32x64",
         11: "64x32", 12: "64x64", 13: "64x128", 14: "128x64", 15: "128x128"}


def scan(path):
    """Retorna (n_nodes, times[type], attempts[type], skipped, local_by_bsize)."""
    times = defaultdict(int)
    attempts = defaultdict(int)
    local_by_bsize = defaultdict(lambda: defaultdict(int))
    n = skipped = 0
    with open(path) as f:
        for line in f:
            p = line.rstrip(",\n").split(",")
            if len(p) < 48:
                skipped += 1
                continue
            try:
                bsize = int(p[0])
                t = [int(p[T0 + i]) for i in range(10)]
                a = [int(p[18 + i]) for i in range(10)]
            except ValueError:
                skipped += 1
                continue
            n += 1
            for i in range(10):
                times[i] += t[i]
                attempts[i] += a[i]
                local_by_bsize[bsize][i] += t[i]
    return n, times, attempts, skipped, local_by_bsize


def pools(times):
    local = sum(times[i] for i in LOCAL)
    return {
        "local": local,
        "NONE": times[0],
        "RECT": sum(times[i] for i in RECT),
        "AB": sum(times[i] for i in AB),
        "4WAY": sum(times[i] for i in FOURWAY),
        "ABFW": sum(times[i] for i in AB + FOURWAY),
    }


def pct(x, tot):
    return 100.0 * x / tot if tot else 0.0


def main(argv):
    paths = [a for a in argv if not a.startswith("--")]
    labels = None
    for a in argv:
        if a.startswith("--label"):
            labels = a.split("=", 1)[1].split(",") if "=" in a else None
    if labels is None:
        labels = [p.split("/")[-2] if "/" in p else p for p in paths]

    grand_t = defaultdict(int)
    grand_a = defaultdict(int)
    grand_bs = defaultdict(lambda: defaultdict(int))
    grand_n = 0

    print("=== por sequencia (share do tempo de busca LOCAL) ===")
    hdr = f"{'seq':16s} {'nos':>10s} {'NONE':>7s} {'RECT':>7s} {'AB':>7s} {'4WAY':>7s} {'AB+4W':>8s}"
    print(hdr)
    for path, lab in zip(paths, labels):
        n, times, att, skip, lbs = scan(path)
        P = pools(times)
        print(f"{lab:16s} {n:>10,} "
              f"{pct(P['NONE'],P['local']):6.1f}% {pct(P['RECT'],P['local']):6.1f}% "
              f"{pct(P['AB'],P['local']):6.1f}% {pct(P['4WAY'],P['local']):6.1f}% "
              f"{pct(P['ABFW'],P['local']):7.2f}%")
        grand_n += n
        for i in range(10):
            grand_t[i] += times[i]
            grand_a[i] += att[i]
        for bs in lbs:
            for i in range(10):
                grand_bs[bs][i] += lbs[bs][i]

    G = pools(grand_t)
    lt = G["local"]
    print(f"\n=== AGREGADO ({len(paths)} seqs, {grand_n:,} nos) ===")
    print(f"tempo LOCAL total (exclui SPLIT recursivo): {lt/1e6:.1f} s\n")
    print("candidato   tempo(s)   share    attempts")
    for i in LOCAL:
        print(f"  {NAMES[i]:8s} {grand_t[i]/1e6:8.1f}  {pct(grand_t[i],lt):6.2f}%  {grand_a[i]:>10,}")
    print(f"\n  NONE          : {pct(G['NONE'],lt):6.2f}%")
    print(f"  RECT (H+V)    : {pct(G['RECT'],lt):6.2f}%")
    print(f"  AB            : {pct(G['AB'],lt):6.2f}%")
    print(f"  4-way         : {pct(G['4WAY'],lt):6.2f}%")
    print(f"  AB + 4-way    : {pct(G['ABFW'],lt):6.2f}%   <-- alavanca C3")
    print(f"  RECT+AB+4way  : {pct(G['RECT']+G['ABFW'],lt):6.2f}%   (disable_rect derruba os 3)")
    verdict = "MORTA (<10%)" if pct(G['ABFW'], lt) < 10 else "VIVA (>=10%)"
    print(f"\n>>> C3 (podar so AB/4-way): {verdict}\n")

    print("=== share AB+4way por bsize (agregado) ===")
    for bs in sorted(grand_bs):
        d = grand_bs[bs]
        blt = sum(d[i] for i in LOCAL)
        if blt == 0:
            continue
        abfw = sum(d[i] for i in AB + FOURWAY)
        print(f"  {BSIZE.get(bs,'bs'+str(bs)):9s} local={blt/1e6:7.1f}s  AB+4way={pct(abfw,blt):6.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: analyze_partstats.py file1.csv [file2.csv ...]")
    main(sys.argv[1:])
