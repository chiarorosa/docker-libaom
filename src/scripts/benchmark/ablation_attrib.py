#!/usr/bin/env python3
"""Attribution ablation: is the speedup from the ML model or just the policy?

Runs the SAME pruning policy (NONE-commit only: tau_split=inf, tau_rest=off) in
the SAME encoder, sweeping tau_none, with three score sources selected by
AV1_STUDENT_BASELINE:
  * ml       -- the distilled ConvNeXt->student MLP (the proposal);
  * variance -- P(NONE)=exp(-var/V0), the obvious flat-block heuristic;
  * random   -- P(NONE)=uniform hash, prunes the same fraction at random.

Each (method, tau_none) yields a (speedup, BD-rate) point against one shared
anchor. If the ML curve Pareto-dominates both baselines (lower BD-rate at equal
speedup), the gain is attributable to the model's node selection, not the
wrapper. That is the figure the thesis needs.
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from run_benchmark import probe, read_y, y_psnr, encode, decode  # noqa: E402
from bd_rate import bd_rate  # noqa: E402
from h7h8_bench import eval_point  # noqa: E402


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--decoder", default="/workspace/build/libaom_ml_check/aomdec")
    p.add_argument("--seq", required=True)
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--methods", nargs="+", default=["ml", "variance", "random"])
    p.add_argument("--tau-none", type=float, nargs="+",
                   default=[0.6, 0.75, 0.9],
                   help="default tau grid, used by any method without an override")
    p.add_argument("--tau-none-for", nargs="+", default=[], metavar="METHOD=T,T,..",
                   help="per-method tau grid, e.g. variance=0.995,0.99,0.97. The "
                        "arms need different grids: at a given tau the variance "
                        "score prunes far more than the model's, so one shared "
                        "grid leaves their speedup ranges disjoint and no "
                        "matched-speedup comparison exists.")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/ablation_attrib")
    args = p.parse_args(argv)

    taus = {m: list(args.tau_none) for m in args.methods}
    for spec in args.tau_none_for:
        method, _, vals = spec.partition("=")
        if method not in taus:
            p.error("--tau-none-for names {!r}, not in --methods {}".format(
                method, args.methods))
        taus[method] = [float(v) for v in vals.split(",") if v]

    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== ANCHOR (once) ===", flush=True)
    _, _, anchor, rows = eval_point("anchor", args.anchor_enc, args.decoder,
                                    args.seq, args, {}, work, None)

    def flush(summary):
        """Rewrite both CSVs after every point: a 4K cpu0 sweep runs for hours,
        and a crash at the last point must not cost the whole campaign."""
        with open(os.path.join(args.out_dir, "runs.csv"), "w", newline="") as f:
            csv.writer(f).writerows(
                [("encoder", "cq", "kbps", "y_psnr", "encode_s")] + rows)
        with open(os.path.join(args.out_dir, "curve.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "tau_none", "bd_rate_pct", "speedup_x", "ts_pct"])
            w.writerows([(m, tn, round(bd, 3), round(s, 3),
                          round((1 - 1 / s) * 100, 2)) for m, tn, bd, s in summary])

    summary = []
    flush(summary)
    for method in args.methods:
        for tn in taus[method]:
            name = "{}_tn{:.3f}".format(method, tn)
            env = {"AV1_STUDENT_BASELINE": method,
                   "AV1_STUDENT_TAU_NONE": str(tn),
                   "AV1_STUDENT_TAU_SPLIT": "2",     # never force-split
                   "AV1_STUDENT_TAU_REST": "-1"}     # rect-off disabled
            print("=== {} ===".format(name), flush=True)
            bd, speed, _, r = eval_point(name, args.test_enc, args.decoder,
                                         args.seq, args, env, work, anchor)
            rows += r
            summary.append((method, tn, bd, speed))
            flush(summary)

    print("\n=== ATTRIBUTION ABLATION ({}, {} frames) ===".format(
        os.path.basename(args.seq).split("_")[0], args.frames))
    print("{:<10} {:>8} {:>10} {:>8} {:>10}".format("method", "tau_none",
                                                    "BD-rate%", "TS%", "speedup"))
    for m, tn, bd, s in summary:
        print("{:<10} {:>8.3f} {:>10.3f} {:>8.1f} {:>10.3f}".format(
            m, tn, bd, (1 - 1 / s) * 100, s))
    print("ABLATION_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
