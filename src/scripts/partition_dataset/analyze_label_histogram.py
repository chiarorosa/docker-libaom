#!/usr/bin/env python3
"""Report the label distribution of the partition dataset from
label_histogram.csv (produced by rebuild_manifest_stats.py).

This is the measurement that item A8 of the action plan was blocked on: every
prior statement about label balance in the project came from reading code or
from gate CSVs, never from the dataset itself. It feeds the modeling items:

  * class imbalance per block size (B4: train_student_h9.py:112 disables class
    weighting; split_recall at dim=16 is ~0.026),
  * the qindex dependence of the label mix (B2: adaptive tau per qindex),
  * the HORZ/VERT balance inside REST (B3: the collapse in student.py:23-29
    discards orientation).

Usage:
  python analyze_label_histogram.py --hist results/dataset_h9/label_histogram.csv
"""

import argparse
import csv
import sys
from collections import defaultdict

PART_NAMES = ["NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B",
              "VERT_A", "VERT_B", "HORZ_4", "VERT_4"]
HORZ_SET = {"HORZ", "HORZ_A", "HORZ_B", "HORZ_4"}
VERT_SET = {"VERT", "VERT_A", "VERT_B", "VERT_4"}
DECISION_DIMS = [64, 32, 16]


def student_class(name):
    """student.py:23-29 collapse: NONE / SPLIT / REST."""
    if name == "NONE":
        return "NONE"
    if name == "SPLIT":
        return "SPLIT"
    return "REST"


def pct(x, tot):
    return 100.0 * x / tot if tot else 0.0


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hist", default="/workspace/results/dataset_h9/label_histogram.csv")
    p.add_argument("--out", default=None, help="optional markdown report path")
    args = p.parse_args(argv)

    rows = list(csv.DictReader(open(args.hist, newline="")))
    for r in rows:
        r["count"] = int(r["count"])
        r["block_dim"] = int(r["block_dim"])
        r["cq_level"] = int(r["cq_level"])
        r["base_qindex"] = int(r["base_qindex"])

    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    # --- 1. 3-class balance per block size (all sequences, all CQ) ---------
    by_dim = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_dim[r["block_dim"]][student_class(r["partition_name"])] += r["count"]

    emit("## 1. Student 3-class balance per block size (all 16 seqs x 4 CQ)")
    emit()
    emit("| block_dim | N | NONE | SPLIT | REST |")
    emit("|---|--:|--:|--:|--:|")
    for d in DECISION_DIMS:
        c = by_dim[d]
        t = sum(c.values())
        emit("| {}x{} | {:,} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            d, d, t, pct(c["NONE"], t), pct(c["SPLIT"], t), pct(c["REST"], t)))
    c8 = by_dim[8]
    t8 = sum(c8.values())
    emit("| 8x8 (excluded) | {:,} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
        t8, pct(c8["NONE"], t8), pct(c8["SPLIT"], t8), pct(c8["REST"], t8)))
    emit()

    # --- 2. SPLIT rate per (block size, CQ) -- the B2 signal ---------------
    cqs = sorted({r["cq_level"] for r in rows})
    q_of = {r["cq_level"]: r["base_qindex"] for r in rows}
    by_dim_cq = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_dim_cq[(r["block_dim"], r["cq_level"])][
            student_class(r["partition_name"])] += r["count"]

    emit("## 2. SPLIT rate by (block size, CQ) -- evidence for tau per qindex")
    emit()
    emit("| block_dim | " + " | ".join(
        "cq{} (q={})".format(q, q_of[q]) for q in cqs) + " |")
    emit("|---|" + "--:|" * len(cqs))
    for d in DECISION_DIMS:
        cells = []
        for q in cqs:
            c = by_dim_cq[(d, q)]
            cells.append("{:.2f}%".format(pct(c["SPLIT"], sum(c.values()))))
        emit("| {}x{} | ".format(d, d) + " | ".join(cells) + " |")
    emit()
    emit("NONE rate by (block size, CQ):")
    emit()
    emit("| block_dim | " + " | ".join("cq{}".format(q) for q in cqs) + " |")
    emit("|---|" + "--:|" * len(cqs))
    for d in DECISION_DIMS:
        cells = []
        for q in cqs:
            c = by_dim_cq[(d, q)]
            cells.append("{:.2f}%".format(pct(c["NONE"], sum(c.values()))))
        emit("| {}x{} | ".format(d, d) + " | ".join(cells) + " |")
    emit()

    # --- 3. HORZ vs VERT inside REST -- the B3 signal ----------------------
    emit("## 3. Orientation inside REST (collapsed away by student.py:23-29)")
    emit()
    emit("| block_dim | REST N | HORZ-family | VERT-family | H/(H+V) |")
    emit("|---|--:|--:|--:|--:|")
    for d in DECISION_DIMS:
        h = sum(r["count"] for r in rows
                if r["block_dim"] == d and r["partition_name"] in HORZ_SET)
        v = sum(r["count"] for r in rows
                if r["block_dim"] == d and r["partition_name"] in VERT_SET)
        emit("| {}x{} | {:,} | {:.2f}% | {:.2f}% | {:.3f} |".format(
            d, d, h + v, pct(h, h + v), pct(v, h + v),
            h / (h + v) if (h + v) else 0.0))
    emit()

    # --- 4. Full 10-class breakdown per block size -------------------------
    emit("## 4. Full 10-class breakdown per block size")
    emit()
    emit("| block_dim | " + " | ".join(PART_NAMES) + " |")
    emit("|---|" + "--:|" * len(PART_NAMES))
    for d in DECISION_DIMS:
        tot = sum(r["count"] for r in rows if r["block_dim"] == d)
        cells = []
        for name in PART_NAMES:
            c = sum(r["count"] for r in rows
                    if r["block_dim"] == d and r["partition_name"] == name)
            cells.append("{:.2f}".format(pct(c, tot)))
        emit("| {}x{} | ".format(d, d) + " | ".join(cells) + " |")
    emit()

    # --- 5. Per-sequence spread of the SPLIT rate at dim=16 ----------------
    emit("## 5. Per-sequence SPLIT rate at 16x16 (spread across content)")
    emit()
    per_seq = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["block_dim"] == 16:
            per_seq[r["sequence"]][student_class(r["partition_name"])] += r["count"]
    ranked = sorted(per_seq.items(),
                    key=lambda kv: pct(kv[1]["SPLIT"], sum(kv[1].values())))
    emit("| sequence | N | SPLIT | NONE |")
    emit("|---|--:|--:|--:|")
    for seq, c in ranked:
        t = sum(c.values())
        emit("| {} | {:,} | {:.2f}% | {:.2f}% |".format(
            seq.split("_")[0], t, pct(c["SPLIT"], t), pct(c["NONE"], t)))
    emit()

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(out) + "\n")
        print("report -> " + args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
