#!/usr/bin/env python3
"""Rebuild the dataset manifest statistics from the .pkl files, and emit the
joint label histogram by (sequence, cq_level, block_dim, partition).

Why this exists: build_dataset.py wrote the manifest statistics with a stale
4116-byte record layout (SAMPLE at build_dataset.py:46) while the C
instrumentation and the converter both use the canonical 4144-byte layout
(partition_search.c:80-100, convert_partition_data.py:43-45). Reading 4116-byte
frames over 4144-byte records desynchronizes after the first sample, so every
statistical column in manifest.csv is invalid:

  * base_qindex     -- read from the wrong offset (recorded 0 everywhere)
  * dim8..dim64     -- garbage; they do not sum to num_samples
  * part_*          -- garbage; mass collapses onto part_NONE
  * num_samples     -- inflated by exactly 4144/4116 = 1.0068

Only these columns are affected. The .pkl payloads are intact (the converter was
always correct) and the training loader reads only pkl_path/sequence/cq_level
from the manifest (data.py:49-62), so no model, BD-rate or Gate result depends
on the corrupted columns.

The .bin files were deleted at build time (--no-keep-bin), so the counts are
recovered from the .pkl, which is the authoritative payload.

Usage:
  python rebuild_manifest_stats.py --dataset-dir /workspace/results/dataset_h9
"""

import argparse
import csv
import os
import pickle
import shutil
import sys
from collections import Counter

import numpy as np

PART_NAMES = ["NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B",
              "VERT_A", "VERT_B", "HORZ_4", "VERT_4"]
DIMS = [8, 16, 32, 64]
# 8x8 is a terminal leaf in 4K All-Intra (always NONE) and is excluded from the
# model -- see partition_defs.py:41-54 (MODEL_LEVELS).
DECISION_DIMS = [16, 32, 64]

STAT_FIELDS = (["base_qindex", "num_samples"]
               + ["dim{}".format(d) for d in DIMS]
               + ["part_{}".format(p) for p in PART_NAMES])


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--manifest", default=None,
                   help="default: <dataset-dir>/manifest.csv")
    p.add_argument("--hist-out", default=None,
                   help="default: <dataset-dir>/label_histogram.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and report, but do not touch manifest.csv")
    return p.parse_args(argv)


def scan_pkl(path):
    """Return (n, base_qindex, dim_counts, part_counts, joint) for one .pkl.

    `joint` maps (block_dim, partition) -> count. qindex is verified constant
    across the file; a per-(seq,QP) dataset has exactly one base_qindex.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    bdim = np.asarray(d["block_dim"])
    part = np.asarray(d["partition"])
    q = np.asarray(d["qindex"])
    n = int(bdim.size)

    uq = np.unique(q)
    if uq.size != 1:
        raise SystemExit("{}: expected one qindex, found {}".format(
            os.path.basename(path), uq.tolist()))
    base_q = int(uq[0])

    # Joint (block_dim, partition) crosstab in one vectorized pass.
    pairs, counts = np.unique(
        np.stack([bdim.astype(np.int64), part.astype(np.int64)]),
        axis=1, return_counts=True)
    joint = {(int(pairs[0, i]), int(pairs[1, i])): int(counts[i])
             for i in range(pairs.shape[1])}

    dim_c, part_c = Counter(), Counter()
    for (bd, pt), c in joint.items():
        dim_c[bd] += c
        part_c[pt] += c
    return n, base_q, dim_c, part_c, joint


def main(argv):
    args = parse_args(argv)
    manifest = args.manifest or os.path.join(args.dataset_dir, "manifest.csv")
    hist_out = args.hist_out or os.path.join(args.dataset_dir,
                                             "label_histogram.csv")
    if not os.path.exists(manifest):
        raise SystemExit("manifest not found: " + manifest)

    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    print("manifest: {} rows".format(len(rows)))

    hist_rows = []
    old_total = new_total = 0
    repaired = 0

    for i, row in enumerate(rows, 1):
        pkl = row["pkl_path"]
        if not os.path.exists(pkl):  # tolerate a relocated dataset dir
            pkl = os.path.join(args.dataset_dir, os.path.basename(pkl))
        if not os.path.exists(pkl):
            raise SystemExit("missing .pkl for row {}: {}".format(
                i, row["pkl_path"]))

        n, base_q, dim_c, part_c, joint = scan_pkl(pkl)
        old_n = int(row["num_samples"])
        old_total += old_n
        new_total += n

        row["num_samples"] = n
        row["base_qindex"] = base_q
        for d in DIMS:
            row["dim{}".format(d)] = dim_c.get(d, 0)
        for k, name in enumerate(PART_NAMES):
            row["part_{}".format(name)] = part_c.get(k, 0)
        repaired += 1

        for (bd, pt), c in sorted(joint.items()):
            hist_rows.append({
                "sequence": row["sequence"],
                "cq_level": row["cq_level"],
                "base_qindex": base_q,
                "block_dim": bd,
                "partition": pt,
                "partition_name": PART_NAMES[pt],
                "count": c,
                "is_decision_node": int(bd in DECISION_DIMS),
            })

        print("  [{:2d}/{}] {} cq{} -> n={} qindex={} dims={} (was n={})".format(
            i, len(rows), row["sequence"][:28], row["cq_level"], n, base_q,
            {d: dim_c.get(d, 0) for d in DIMS}, old_n), flush=True)

    dec = sum(r["count"] for r in hist_rows if r["is_decision_node"])
    print("\ntotal samples: manifest said {:,} -> real {:,} (delta {:+,}, "
          "{:+.3f}%)".format(old_total, new_total, new_total - old_total,
                             100.0 * (new_total - old_total) / old_total))
    print("decision nodes (dim in {}): {:,} ({:.1f}% of all)".format(
        DECISION_DIMS, dec, 100.0 * dec / new_total))

    with open(hist_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence", "cq_level", "base_qindex", "block_dim", "partition",
            "partition_name", "count", "is_decision_node"])
        w.writeheader()
        w.writerows(hist_rows)
    print("histogram -> {} ({} rows)".format(hist_out, len(hist_rows)))

    if args.dry_run:
        print("(dry-run: manifest.csv untouched)")
        return

    backup = manifest + ".bak-4116"
    if not os.path.exists(backup):
        shutil.copy2(manifest, backup)
        print("backup -> " + backup)
    tmp = manifest + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, manifest)
    print("manifest repaired: {} rows, columns {}".format(repaired, STAT_FIELDS))


if __name__ == "__main__":
    main(sys.argv[1:])
