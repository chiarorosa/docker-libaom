#!/usr/bin/env python3
"""Report of the D2-lean campaign (encode_swap_h9d.py): marginal of H9d stacked on
the H9a native-CNN substitute at cpu-used=1/2/3, read against the predictions
pre-registered in results/thesis/IEEE_Conference_Template/ISCAS/CLAUDE.md §6
(commit 28f9145, before the first encode).

Both arms come from the SAME campaign and binary; BD-BR and TS are referenced to
the Fase 6 cpu-used=0 anchor, with the vetted helpers of report_swap.py (canonical
TS: per-QP saving, mean over QPs, then mean over sequences).

    dTS(N) = TS(h9a_bal_h9d_cpuN) - TS(h9a_bal_cpuN)      [pp]
    dBD(N) = BD(h9a_bal_h9d_cpuN) - BD(h9a_bal_cpuN)      [pp]

Pre-registered reading rules: dTS <= 0.46 pp is "not resolved"; no recalibration.

Reproduction (container av1_bench):
    /workspace/build/venv-ml/bin/python src/scripts/fase6/report_swap_h9d.py
"""
import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "benchmark"))
from bd_rate import bd_rate  # noqa: E402
from report_swap import load_csv, curve  # noqa: E402

RESOLUTION_PP = 0.46
LASCAS_P0_DTS = 1.02
P2_MAX_DBD = 0.05


def eval_cfg(pts, ra, qa, atime):
    rt, qt = curve(pts)
    ts = statistics.mean((atime[c] - pts[c]["time"]) / atime[c] * 100.0
                         for c in pts)
    return bd_rate(ra, qa, rt, qt), ts


def paired_t(d):
    """Two-sided paired t-test on differences d; returns (t, p)."""
    from scipy import stats
    res = stats.ttest_1samp(d, 0.0)
    return float(res.statistic), float(res.pvalue)


def main():
    root = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", default=os.path.join(
        root, "results/benchmark/fase6_swap_h9d/raw_results.csv"))
    ap.add_argument("--anchor", default=os.path.join(
        root, "results/benchmark/fase6/raw_results.csv"))
    ap.add_argument("--out", default=os.path.join(
        root, "results/benchmark/fase6_swap_h9d"))
    args = ap.parse_args()

    d2 = load_csv(args.d2)
    f6 = load_csv(args.anchor)
    rows = []
    for seq in sorted(d2):
        ra, qa = curve(f6[seq]["anchor"])
        atime = {c: f6[seq]["anchor"][c]["time"] for c in f6[seq]["anchor"]}
        for n in (1, 2, 3):
            b_bd, b_ts = eval_cfg(d2[seq]["h9a_bal_cpu%d" % n], ra, qa, atime)
            h_bd, h_ts = eval_cfg(d2[seq]["h9a_bal_h9d_cpu%d" % n], ra, qa, atime)
            rows.append(dict(seq=seq, cpu=n, base_bd=b_bd, base_ts=b_ts,
                             h9d_bd=h_bd, h9d_ts=h_ts,
                             dbd=h_bd - b_bd, dts=h_ts - b_ts))

    with open(os.path.join(args.out, "marginal_per_seq.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("%-14s %3s %8s %8s %8s %8s %7s %7s" % (
        "seq", "cpu", "baseBD", "baseTS", "h9dBD", "h9dTS", "dBD", "dTS"))
    for r in rows:
        print("%-14s %3d %8.3f %8.2f %8.3f %8.2f %+7.3f %+7.2f" % (
            r["seq"], r["cpu"], r["base_bd"], r["base_ts"], r["h9d_bd"],
            r["h9d_ts"], r["dbd"], r["dts"]))

    avg = {}
    print("\nAverages over %d sequences:" % len(d2))
    with open(os.path.join(args.out, "marginal_average.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpu", "base_bd", "base_ts", "h9d_bd", "h9d_ts", "dbd",
                    "dts", "dts_t", "dts_p", "dbd_t", "dbd_p", "dts_pos_seqs"])
        for n in (1, 2, 3):
            sub = [r for r in rows if r["cpu"] == n]
            m = {k: statistics.mean(r[k] for r in sub)
                 for k in ("base_bd", "base_ts", "h9d_bd", "h9d_ts", "dbd", "dts")}
            m["dts_t"], m["dts_p"] = paired_t([r["dts"] for r in sub])
            m["dbd_t"], m["dbd_p"] = paired_t([r["dbd"] for r in sub])
            m["pos"] = sum(r["dts"] > 0 for r in sub)
            avg[n] = m
            w.writerow([n] + [round(m[k], 4) for k in (
                "base_bd", "base_ts", "h9d_bd", "h9d_ts", "dbd", "dts",
                "dts_t", "dts_p", "dbd_t", "dbd_p")] + [m["pos"]])
            print("  cpu%d  base %.3f%% / %.2f%%  +H9d %.3f%% / %.2f%%  "
                  "dBD %+.3f pp (p=%.3f)  dTS %+.2f pp (p=%.3f, %d/8 > 0)" % (
                      n, m["base_bd"], m["base_ts"], m["h9d_bd"], m["h9d_ts"],
                      m["dbd"], m["dbd_p"], m["dts"], m["dts_p"], m["pos"]))

    print("\nPre-registered predictions (ISCAS/CLAUDE.md §6):")
    for n in (1, 2):
        d = avg[n]["dts"]
        if d <= RESOLUTION_PP:
            v = "FAILED (dTS not resolved, <= %.2f pp)" % RESOLUTION_PP
        elif d >= LASCAS_P0_DTS:
            v = "FAILED (dTS >= %.2f pp, upper bound)" % LASCAS_P0_DTS
        else:
            v = "MET"
        print("  P1 cpu%d: %.2f < dTS=%.2f < %.2f  -> %s" % (
            n, RESOLUTION_PP, d, LASCAS_P0_DTS, v))
    for n in (1, 2, 3):
        d = avg[n]["dbd"]
        print("  P2 cpu%d: dBD=%+.3f <= +%.2f -> %s" % (
            n, d, P2_MAX_DBD, "MET" if d <= P2_MAX_DBD else "FAILED"))
    d3 = avg[3]["dts"]
    lim = min(avg[1]["dts"], avg[2]["dts"])
    print("  P3: dTS(cpu3)=%.2f < min(dTS cpu1, cpu2)=%.2f -> %s" % (
        d3, lim, "MET" if d3 < lim else "FAILED"))


if __name__ == "__main__":
    main()
