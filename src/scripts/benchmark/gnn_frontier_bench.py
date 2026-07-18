#!/usr/bin/env python3
"""REAL BD-rate x speedup FRONTIER: deployable pixel GNN vs the H9a student.

Single fixed-tau points hide how a policy trades BD-rate against speedup as
the NONE-prune threshold moves. This driver sweeps TAU_NONE and, for each
value, evaluates TWO points against the SAME anchor / SAME frames:

  * gnn_t{tau}: the deployable pixel-only GNN's decisions, replayed through
    AV1_STUDENT_PROBS_FILE (encoder never runs the GNN; gnn_replay.py already
    scored it offline via gnn_replay.py -- same mechanism as gnn_replay_bench.py).
  * h9a_t{tau}: the deployed H9a distilled student, running live in-loop via
    av1_nn_predict (no replay_tmpl in env).

That gives two real frontiers (not single points) on identical held-out
frames, so the GNN-vs-H9a comparison is fair at every operating point, not
just wherever their default taus happen to land.

Reuses h7h8_bench.eval_point for the encode/decode/BD-rate plumbing and
gnn_replay.py for offline GNN scoring, exactly as gnn_replay_bench.py does.
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
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--test-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--decoder", default="/workspace/build/libaom_ml_check/aomdec")
    p.add_argument("--probs-dir",
                   default="/workspace/results/benchmark/gnn_probs")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/gnn_frontier")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.80, 0.90, 0.95, 0.97, 0.99],
                   help="TAU_NONE sweep; SPLIT/REST held fixed at 0.90/0.20")
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

    # Two frontiers over the same TAU_NONE sweep, same anchor, same frames:
    # gnn_t{tau} replays the deployable GNN's precomputed probs; h9a_t{tau}
    # runs the deployed H9a student live (no replay_tmpl).
    points = []
    for tau in args.taus:
        tau_s = str(tau)
        points.append(("gnn_t{}".format(tau), args.test_enc,
                        {"AV1_STUDENT_TAU_NONE": tau_s,
                         "AV1_STUDENT_TAU_SPLIT": "0.90",
                         "AV1_STUDENT_TAU_REST": "0.20",
                         "replay_tmpl": replay_tmpl}))
        points.append(("h9a_t{}".format(tau), args.test_enc,
                        {"AV1_STUDENT_TAU_NONE": tau_s,
                         "AV1_STUDENT_TAU_SPLIT": "0.90",
                         "AV1_STUDENT_TAU_REST": "0.20"}))

    summary = []
    for name, enc, env in points:
        print("=== {} ===".format(name), flush=True)
        bd, speed, _, rows = eval_point(name, enc, args.decoder, args.seq,
                                        args, env, work, anchor=baseline)
        all_rows += rows
        summary.append((name, bd, speed))

    # Rebuild per-model rows keyed by name (safe regardless of eval order).
    by_name = {n: (b, s) for n, b, s in summary}
    gnn_rows = [(tau, by_name["gnn_t{}".format(tau)][0],
                 by_name["gnn_t{}".format(tau)][1]) for tau in args.taus]
    h9a_rows = [(tau, by_name["h9a_t{}".format(tau)][0],
                 by_name["h9a_t{}".format(tau)][1]) for tau in args.taus]

    def ts(speed):
        return (1 - 1 / speed) * 100.0

    gnn_sorted = sorted(gnn_rows, key=lambda r: ts(r[2]))
    h9a_sorted = sorted(h9a_rows, key=lambda r: ts(r[2]))

    csv_path = os.path.join(args.out_dir, "frontier_{}.csv".format(seqbase))
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["model", "tau", "bd_rate_pct", "ts_pct", "speedup"])
        for tau, bd, speed in gnn_sorted:
            wtr.writerow(["gnn", tau, round(bd, 3), round(ts(speed), 2),
                          round(speed, 3)])
        for tau, bd, speed in h9a_sorted:
            wtr.writerow(["h9a", tau, round(bd, 3), round(ts(speed), 2),
                          round(speed, 3)])
        wtr.writerow([])
        wtr.writerow(["encoder", "cq", "kbps", "y_psnr", "encode_s"])
        wtr.writerows(all_rows)

    print("\n=== GNN FRONTIER ({}, {} frames) ===".format(seqbase, args.frames))
    # Reference (human eyeball only, NOT encoded here): native in-loop CNN
    # (Fase 6, cpu-used=1) reaches ~0.45% BD-rate @ ~32.6% TS.
    print("{:<8} {:>10} {:>8} {:>12}".format("tau", "BD-rate%", "TS%", "speedup x"))
    for tau, bd, speed in gnn_sorted:
        print("{:<8} {:>10.3f} {:>8.1f} {:>12.3f}".format(tau, bd, ts(speed), speed))

    print("\n=== H9a FRONTIER ({}, {} frames) ===".format(seqbase, args.frames))
    print("{:<8} {:>10} {:>8} {:>12}".format("tau", "BD-rate%", "TS%", "speedup x"))
    for tau, bd, speed in h9a_sorted:
        print("{:<8} {:>10.3f} {:>8.1f} {:>12.3f}".format(tau, bd, ts(speed), speed))

    print("csv ->", csv_path)
    print("GNN_FRONTIER_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
