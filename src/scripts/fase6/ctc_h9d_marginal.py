#!/usr/bin/env python3
"""Fase 6 — marginal contribution of H9d over the deployed H9a point (CTC).

H9d is a complement to the deployed H9a pruner, so the question the results
chapter must answer is not "H9d vs native" but: starting from the deployed
balanced point (ml_balanced = P_rect), does stacking H9d buy time more cheaply
than simply turning the H9a threshold up?

The H9a threshold knob is represented on CTC by the segment between the two
frozen operating points, ml_balanced (P_rect) and ml_aggr (A3). For each
sequence we linearly interpolate that segment at the TS% actually reached by
ml_bal_h9d, giving the BD-rate the tau knob would have cost for the same time
saving. delta = BD(H9d) - BD(tau at matched TS); negative => H9d is better.

Reproduction:
    /workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d_marginal.py
"""

import argparse
import csv
import os
import statistics

BASE, H9D, AGGR = "ml_balanced", "ml_bal_h9d", "ml_aggr"


def load(path):
    """Return d[seq][config] = {"bd": bd_rate, "ts": ts_pct, "spd": speedup}."""
    d = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d.setdefault(row["seq"], {})[row["config"]] = {
                "bd": float(row["bd_rate"]),
                "ts": float(row["ts_pct"]),
                "spd": float(row["speedup"]),
            }
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-seq",
                   default="/workspace/results/benchmark/fase6/bdrate_per_seq.csv")
    args = p.parse_args()

    d = load(args.per_seq)
    print("=== H9d marginal over the deployed H9a point (CTC A1, 15 frames) ===")
    print("{:<14}{:>18}{:>18}{:>10}{:>10}{:>12}".format(
        "Sequence", "H9a P_rect", "H9a+H9d", "d BD", "d TS", "vs tau knob"))

    dbds, dtss, deltas, wins = [], [], [], 0
    for seq in sorted(d):
        c = d[seq]
        if not all(k in c for k in (BASE, H9D, AGGR)):
            continue
        b, h, a = c[BASE], c[H9D], c[AGGR]
        dbd, dts = h["bd"] - b["bd"], h["ts"] - b["ts"]
        # tau knob interpolated at the TS% H9d actually reached
        span = a["ts"] - b["ts"]
        tau_bd = (b["bd"] + (a["bd"] - b["bd"]) * (h["ts"] - b["ts"]) / span
                  if span > 1e-9 else float("nan"))
        delta = h["bd"] - tau_bd
        dbds.append(dbd); dtss.append(dts); deltas.append(delta)
        wins += delta < 0
        print("{:<14}{:>10.2f}%{:>7.1f}%{:>10.2f}%{:>7.1f}%{:>+10.2f}{:>+10.1f}"
              "{:>12}".format(
                  seq, b["bd"], b["ts"], h["bd"], h["ts"], dbd, dts,
                  "{:+.3f} pp".format(delta)))

    print("-" * 82)
    print("mean   d BD-rate = {:+.3f} pp   d TS = {:+.2f} pp   "
          "cost = {:.4f} pp BD per pp TS".format(
              statistics.mean(dbds), statistics.mean(dtss),
              statistics.mean(dbds) / statistics.mean(dtss)))
    print("mean   delta vs tau knob at matched TS = {:+.3f} pp   "
          "(H9d better in {}/{} seqs)".format(
              statistics.mean(deltas), wins, len(deltas)))

    # Slope of the tau knob itself, for the same "price of time" comparison.
    slopes = [(d[s][AGGR]["bd"] - d[s][BASE]["bd"])
              / (d[s][AGGR]["ts"] - d[s][BASE]["ts"]) for s in sorted(d)
              if all(k in d[s] for k in (BASE, AGGR))]
    print("tau knob slope (P_rect->A3)          = {:.4f} pp BD per pp TS"
          .format(statistics.mean(slopes)))


if __name__ == "__main__":
    main()
