#!/usr/bin/env python3
"""Approach B deployable pixel-only GNN: REAL BD-rate x encode-time speedup.

Thin driver that reuses the H8 replay harness (h7h8_bench.eval_point) instead
of reimplementing the encode/decode/BD-rate plumbing. It first scores the
deployable GNN (pixel+qindex features only, feat_mode "pixelquant") over the
held-out sequence via gnn_replay.py, dumping per-cq per-superblock probs in
the same on-disk format H8's surrogate_replay.py produces. Those probs are
then replayed through the encoder's AV1_STUDENT_PROBS_FILE hook -- so the
in-loop decisions are exactly what the GNN would produce, but the encoder
never runs GNN inference; it is purely a measurement device for the GNN's
real BD-rate x speedup at two operating points (balanced / aggressive tau).
"""

import argparse
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from h7h8_bench import eval_point  # noqa: E402

GNN_REPLAY = os.path.join(os.path.dirname(HERE), "partition_model",
                          "gnn_replay.py")
VENV_PY = "/workspace/build/venv-ml/bin/python3"


def gen_probs(seq, seqbase, cq, args, probs_dir):
    """Run gnn_replay.py for one cq; skip if the output already exists."""
    out = os.path.join(probs_dir, "{}_gnn_cq{}.bin".format(seqbase, cq))
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("[skip] probs exist: {}".format(out), flush=True)
        return out
    cmd = [VENV_PY, GNN_REPLAY,
           "--gnn-bundle", args.gnn_bundle,
           "--yuv", seq,
           "--cq", str(cq),
           "--frames", str(args.frames),
           "--skip", str(args.skip),
           "--out", out]
    print("[gen] {}".format(" ".join(cmd)), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("gnn_replay.py failed for cq={}".format(cq))
    return out


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gnn-bundle",
                   default="/workspace/results/models/gnn_L2_pixel/gnn.pt")
    p.add_argument("--seq", required=True, help="single .yuv (held-out)")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--test-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--decoder", default="/workspace/build/libaom_ml_check/aomdec")
    p.add_argument("--probs-dir",
                   default="/workspace/results/benchmark/gnn_probs")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/gnn_replay")
    p.add_argument("--work", default=None,
                   help="temp dir for ivf/yuv; default {out-dir}/_work")
    args = p.parse_args(argv)

    os.makedirs(args.probs_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    work = args.work or os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    seqbase = os.path.basename(args.seq).split("_")[0]

    print("=== GNN PROB GENERATION ({}) ===".format(seqbase), flush=True)
    for cq in args.cqs:
        gen_probs(args.seq, seqbase, cq, args, args.probs_dir)

    replay_tmpl = os.path.join(args.probs_dir, seqbase + "_gnn_cq{cq}.bin")

    print("=== ANCHOR (once) ===", flush=True)
    _, _, baseline, all_rows = eval_point(
        "anchor", args.anchor_enc, args.decoder, args.seq, args, {}, work,
        anchor=None)

    # Two operating points replaying the deployable GNN's precomputed probs:
    # a balanced point (mirrors H7's P_rect taus) and an aggressive point
    # (mirrors H7's aggressive-preset A3) to bracket the GNN's speedup/BD-rate
    # trade-off.
    points = [
        ("gnn_balanced", args.test_enc,
         {"AV1_STUDENT_TAU_NONE": "0.95", "AV1_STUDENT_TAU_SPLIT": "0.90",
          "AV1_STUDENT_TAU_REST": "0.20", "replay_tmpl": replay_tmpl}),
        ("gnn_aggr", args.test_enc,
         {"AV1_STUDENT_TAU_NONE": "0.60", "AV1_STUDENT_TAU_SPLIT": "0.85",
          "AV1_STUDENT_TAU_REST": "0.40", "replay_tmpl": replay_tmpl}),
    ]

    summary = []
    for name, enc, env in points:
        print("=== {} ===".format(name), flush=True)
        bd, speed, _, rows = eval_point(name, enc, args.decoder, args.seq,
                                        args, env, work, anchor=baseline)
        all_rows += rows
        summary.append((name, bd, speed))

    csv_path = os.path.join(args.out_dir, "gnn_replay_{}.csv".format(seqbase))
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["point", "bd_rate_pct", "time_speedup_x", "ts_pct"])
        wtr.writerows([(n, round(b, 3), round(s, 3), round((1 - 1 / s) * 100, 2))
                       for n, b, s in summary])
        wtr.writerow([])
        wtr.writerow(["encoder", "cq", "kbps", "y_psnr", "encode_s"])
        wtr.writerows(all_rows)

    print("\n=== GNN REPLAY SUMMARY ({}, {} frames) ===".format(
        seqbase, args.frames))
    print("{:<16} {:>10} {:>8} {:>12}".format("point", "BD-rate%", "TS%",
                                              "speedup x"))
    for n, b, s in summary:
        print("{:<16} {:>10.3f} {:>8.1f} {:>12.3f}".format(
            n, b, (1 - 1 / s) * 100, s))
    print("csv ->", csv_path)
    print("GNN_REPLAY_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
