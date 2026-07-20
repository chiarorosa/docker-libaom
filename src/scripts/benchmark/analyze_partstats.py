#!/usr/bin/env python3
"""C1 — decompoe o custo de busca por CANDIDATO de particao a partir do
part_timing.csv emitido por um build com CONFIG_COLLECT_PARTITION_STATS=1.

Pergunta (falsificacao da alavanca C3): AB (HORZ_A/B, VERT_A/B) + 4-way
(HORZ_4, VERT_4) somam >=10% do tempo de busca LOCAL do no? Se <10%, podar
so AB/4-way nao vale o esforco de engenharia (C3 morre sem experimento).

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

# libaom BLOCK_SIZE -> label (subset relevante a All-Intra)
BSIZE = {0: "4x4", 1: "4x8", 2: "8x4", 3: "8x8", 4: "8x16", 5: "16x8",
         6: "16x16", 7: "16x32", 8: "32x16", 9: "32x32", 10: "32x64",
         11: "64x32", 12: "64x64", 13: "64x128", 14: "128x64", 15: "128x128"}


def main(path):
    times_by_type = defaultdict(int)      # type -> soma times (todos os nos)
    attempts_by_type = defaultdict(int)   # type -> soma attempts
    local_by_bsize = defaultdict(lambda: defaultdict(int))  # bsize -> type -> times
    n_nodes = 0

    with open(path) as f:
        for line in f:
            p = line.rstrip(",\n").split(",")
            if len(p) < 48:
                continue
            try:
                bsize = int(p[0])
                times = [int(p[T0 + i]) for i in range(10)]
                attempts = [int(p[18 + i]) for i in range(10)]
            except ValueError:
                continue
            n_nodes += 1
            for i in range(10):
                times_by_type[i] += times[i]
                attempts_by_type[i] += attempts[i]
                local_by_bsize[bsize][i] += times[i]

    local_total = sum(times_by_type[i] for i in LOCAL)
    ab = sum(times_by_type[i] for i in AB)
    fw = sum(times_by_type[i] for i in FOURWAY)
    rect = sum(times_by_type[i] for i in RECT)
    none = times_by_type[0]

    print(f"nos processados: {n_nodes:,}")
    print(f"tempo LOCAL total (exclui SPLIT recursivo): {local_total/1e6:.3f} s\n")

    print("=== share do tempo de busca LOCAL por candidato (global) ===")
    for i in LOCAL:
        pct = 100.0 * times_by_type[i] / local_total if local_total else 0
        print(f"  {NAMES[i]:8s} {times_by_type[i]/1e6:8.3f} s  {pct:6.2f}%   "
              f"(attempts {attempts_by_type[i]:,})")
    print()
    print("=== agregados-chave (share do tempo LOCAL) ===")
    print(f"  NONE            : {100.0*none/local_total:6.2f}%")
    print(f"  RECT (H+V)      : {100.0*rect/local_total:6.2f}%")
    print(f"  AB (H/V_A/B)    : {100.0*ab/local_total:6.2f}%")
    print(f"  4-way (H/V_4)   : {100.0*fw/local_total:6.2f}%")
    print(f"  AB + 4-way      : {100.0*(ab+fw)/local_total:6.2f}%   "
          f"<-- alavanca C3")
    print(f"  RECT+AB+4way    : {100.0*(rect+ab+fw)/local_total:6.2f}%   "
          f"(cota superior: disable_rect derruba os 3)")
    print()
    verdict = "MORTA (<10%)" if 100.0*(ab+fw)/local_total < 10 else "VIVA (>=10%)"
    print(f">>> C3 (podar so AB/4-way): {verdict}")
    print()

    print("=== attempts por tipo (AB/4-way foram buscados?) ===")
    for i in AB + FOURWAY:
        print(f"  {NAMES[i]:8s} attempts={attempts_by_type[i]:,}")
    print()

    print("=== share AB+4way por bsize (onde se concentra) ===")
    for bs in sorted(local_by_bsize):
        d = local_by_bsize[bs]
        lt = sum(d[i] for i in LOCAL)
        if lt == 0:
            continue
        abfw = sum(d[i] for i in AB + FOURWAY)
        label = BSIZE.get(bs, f"bs{bs}")
        print(f"  {label:9s} local={lt/1e6:7.3f}s  AB+4way={100.0*abfw/lt:6.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "part_timing.csv")
