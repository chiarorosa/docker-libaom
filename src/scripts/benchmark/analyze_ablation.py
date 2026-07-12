#!/usr/bin/env python3
"""Combine attribution-ablation curves and compare methods at matched speedup.

Reads every curve.csv (method, tau_none, bd_rate_pct, speedup_x) under the given
dirs, then for each method builds its BD-rate(speedup) curve and interpolates the
BD-rate at a grid of common speedup levels. The method with the lowest BD-rate at
every speedup is Pareto-dominant. That is the attribution verdict: if 'ml' wins
everywhere, the speedup is due to the model's node selection, not the policy.
"""

import argparse
import csv
import glob
import os
import sys


def load_curves(dirs):
    pts = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "curve.csv")):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    m = row["method"]
                    pts.setdefault(m, []).append(
                        (float(row["speedup_x"]), float(row["bd_rate_pct"])))
    for m in pts:
        pts[m] = sorted(set(pts[m]))  # by speedup
    return pts


def ts_pct(speedup):
    """Time saving %% from a speedup ratio: TS = (1 - 1/speedup)*100."""
    return (1.0 - 1.0 / speedup) * 100.0


def interp_bd(curve, s):
    """Linear-interpolate BD-rate at speedup s; None if out of range."""
    xs = [p[0] for p in curve]
    if s < xs[0] or s > xs[-1]:
        return None
    for i in range(1, len(curve)):
        x0, y0 = curve[i - 1]
        x1, y1 = curve[i]
        if x0 <= s <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return None


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--speedups", type=float, nargs="+",
                   default=[1.15, 1.30, 1.50, 1.75, 2.00])
    p.add_argument("--out-csv", default=None)
    args = p.parse_args(argv)

    pts = load_curves(args.dirs)
    print("raw points per method (speedup, TS%, BD%):")
    for m in sorted(pts):
        print("  {:<9}".format(m),
              " ".join("({:.2f}x,{:.1f}%,{:.2f})".format(s, ts_pct(s), b)
                       for s, b in pts[m]))

    methods = sorted(pts)
    print("\nBD-rate% at matched speedup (lower is better; '.' = out of range):")
    print("speedup   TS%   " + "".join("{:>12}".format(m) for m in methods) +
          "   winner")
    rows = []
    for s in args.speedups:
        vals = {m: interp_bd(pts[m], s) for m in methods}
        avail = {m: v for m, v in vals.items() if v is not None}
        winner = min(avail, key=avail.get) if avail else "-"
        cells = "".join(
            ("{:>12.3f}".format(vals[m]) if vals[m] is not None else
             "{:>12}".format(".")) for m in methods)
        print("{:>6.2f}x {:>5.1f}  {}   {}".format(s, ts_pct(s), cells, winner))
        rows.append([s, round(ts_pct(s), 2)] + [vals[m] for m in methods] + [winner])

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["speedup_x", "ts_pct"] + methods + ["winner"])
            w.writerows(rows)
        print("\nwrote", args.out_csv)

    # Verdict
    ml_wins = [r for r in rows if r[-1] == "ml"]
    comparable = [r for r in rows
                  if sum(1 for v in r[2:-1] if v is not None) >= 2]
    print("\nVERDICT: 'ml' is lowest-BD at {}/{} comparable speedup levels.".format(
        len(ml_wins), len(comparable)))


if __name__ == "__main__":
    main(sys.argv[1:])
