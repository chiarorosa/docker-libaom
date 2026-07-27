#!/usr/bin/env python3
"""Fase 6 — CTC final-results report (thesis tables, IEEE style).

Consumes raw_results.csv (from encode_ctc.py) and, per sequence, builds the
rate-distortion curve of each configuration over the QP grid, then reports each
non-anchor configuration against the libaom-original anchor:

    BD-Rate (%)   Bjontegaard delta-rate on PSNR-Y (negative => rate savings;
                  the ML pruner trades a small positive BD-rate for time).
    BD-PSNR (dB)  the dual view, same PSNR-Y curve (positive => quality gain).
    TS (%)        time saving = (T_anchor - T) / T_anchor, averaged over QPs.
    Speedup x     T_anchor / T, averaged over QPs.

Bitrate follows the AOM-CTC definition kbps = bytes*8*fps/frames/1000; the fps
factor cancels inside a per-sequence BD-Rate, so it only affects the printed
kbps, not the deltas.

Outputs (under --out-dir):
    bdrate_per_seq.csv     one row per (seq, config)
    bdrate_average.csv     one row per config, averaged over sequences
    tables.tex             LaTeX booktabs: per-sequence + Average, and the
                           BD-Rate x TS frontier positioning the two ML points
                           against the native cpu-used knob.

The average BD-Rate/BD-PSNR are arithmetic means over sequences (the standard
codec-paper convention); TS%/Speedup are averaged over QPs then over sequences.
"""

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "benchmark"))
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from bd_rate import bd_rate  # noqa: E402  (reused Bjontegaard math, Fase 5)

# Order configs are presented in tables (anchor is the reference, not a row).
CONFIG_ORDER = ["ml_balanced", "ml_bal_h9d", "ml_bal_h9d_pl20",
                "ml_aggr", "ml_aggr_h9d", "ml_aggr_h9d_pl20",
                "native_cpu1", "native_cpu2", "native_cpu3"]
CONFIG_LABEL = {
    "ml_balanced": "ML balanced (P_rect)",
    "ml_bal_h9d": "ML balanced + H9d PL10",
    "ml_bal_h9d_pl20": "ML balanced + H9d PL20",
    "ml_aggr": "ML aggressive (A3)",
    "ml_aggr_h9d": "ML aggressive + H9d PL10",
    "ml_aggr_h9d_pl20": "ML aggressive + H9d PL20",
    "native_cpu1": "libaom cpu-used=1",
    "native_cpu2": "libaom cpu-used=2",
    "native_cpu3": "libaom cpu-used=3",
}


def bd_psnr(rate_a, psnr_a, rate_t, psnr_t, samples=100):
    """Bjontegaard delta-PSNR (dB): mean PSNR gap over the overlapping rate
    range, integrated in log10(rate). Positive => test is higher quality."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    def prep(rate, psnr):
        r = np.log10(np.asarray(rate, float))
        q = np.asarray(psnr, float)
        order = np.argsort(r)
        r, q = r[order], q[order]
        keep = np.concatenate(([True], np.diff(r) > 1e-9))
        return r[keep], q[keep]

    ra, qa = prep(rate_a, psnr_a)
    rt, qt = prep(rate_t, psnr_t)
    if len(ra) < 2 or len(rt) < 2:
        return float("nan")
    lo, hi = max(ra.min(), rt.min()), min(ra.max(), rt.max())
    if hi <= lo:
        return float("nan")
    xs = np.linspace(lo, hi, samples)
    fa = PchipInterpolator(ra, qa)
    ft = PchipInterpolator(rt, qt)
    return float((ft(xs) - fa(xs)).mean())


def load(csv_path):
    """Return data[seq][config] = {cq: {rate, psnr, time}}."""
    data = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            seq, cfg, cq = row["seq"], row["config"], int(row["cq"])
            fps = int(row["fps_num"]) / int(row["fps_den"])
            frames = int(row["frames"])
            kbps = int(row["bytes"]) * 8 * fps / frames / 1000.0
            data.setdefault(seq, {}).setdefault(cfg, {})[cq] = {
                "rate": kbps, "psnr": float(row["psnr_y"]),
                "time": float(row["time_s"]),
            }
    return data


def curve(points):
    """(rates, psnrs) sorted by cq for a config's {cq: {...}} dict."""
    cqs = sorted(points)
    return ([points[c]["rate"] for c in cqs],
            [points[c]["psnr"] for c in cqs], cqs)


def per_seq_rows(data):
    """One dict per (seq, config) with bd_rate, bd_psnr, ts_pct, speedup."""
    rows = []
    for seq in sorted(data):
        cfgs = data[seq]
        if "anchor" not in cfgs:
            print("[warn] {}: no anchor, skipped".format(seq), file=sys.stderr)
            continue
        ra, qa, cqs_a = curve(cfgs["anchor"])
        for cfg in CONFIG_ORDER:
            if cfg not in cfgs:
                continue
            rt, qt, _ = curve(cfgs[cfg])
            # matched-QP time ratios (anchor and cfg share the QP grid)
            shared = [c for c in cqs_a if c in cfgs[cfg]]
            ts = statistics.mean(
                (cfgs["anchor"][c]["time"] - cfgs[cfg][c]["time"])
                / cfgs["anchor"][c]["time"] * 100.0 for c in shared)
            spd = statistics.mean(
                cfgs["anchor"][c]["time"] / cfgs[cfg][c]["time"]
                for c in shared)
            rows.append({
                "seq": seq, "config": cfg,
                "bd_rate": bd_rate(ra, qa, rt, qt),
                "bd_psnr": bd_psnr(ra, qa, rt, qt),
                "ts_pct": ts, "speedup": spd,
            })
    return rows


def averages(rows):
    """Arithmetic mean per config over sequences."""
    out = []
    for cfg in CONFIG_ORDER:
        sub = [r for r in rows if r["config"] == cfg]
        if not sub:
            continue
        out.append({
            "config": cfg,
            "bd_rate": statistics.mean(r["bd_rate"] for r in sub),
            "bd_psnr": statistics.mean(r["bd_psnr"] for r in sub),
            "ts_pct": statistics.mean(r["ts_pct"] for r in sub),
            "speedup": statistics.mean(r["speedup"] for r in sub),
        })
    return out


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in keys})


def _fmt(x):
    return "{:+.2f}".format(x) if x == x else "--"  # x==x is False for nan


def _tex(label):
    """Escape LaTeX-special chars in a config label (underscore in P_rect)."""
    return label.replace("_", r"\_")


def write_tex(path, per_seq, avg):
    """LaTeX booktabs: per-sequence + Average block, then frontier table."""
    seqs = sorted({r["seq"] for r in per_seq})
    lines = []
    a = lines.append

    a("% Fase 6 -- CTC All-Intra, Class A1 4K 10-bit. Anchor = libaom original")
    a("% (cpu-used=0). BD-Rate/BD-PSNR on PSNR-Y; TS%/Speedup vs anchor.")
    a(r"\begin{table}[t]")
    a(r"\centering")
    a(r"\caption{Fase 6 -- final CTC results (All Intra, Class~A1, 4K, 10-bit)."
      r" Anchor: unmodified libaom, \texttt{cpu-used=0}. Negative BD-Rate means"
      r" bitrate savings; TS and speedup are versus the anchor.}")
    a(r"\label{tab:fase6-ctc}")
    a(r"\begin{tabular}{llrrrr}")
    a(r"\toprule")
    a(r"Sequence & Configuration & BD-Rate (\%) & BD-PSNR (dB) & TS (\%)"
      r" & Speedup$\times$ \\")
    a(r"\midrule")
    for seq in seqs:
        first = True
        for cfg in CONFIG_ORDER:
            r = next((x for x in per_seq
                      if x["seq"] == seq and x["config"] == cfg), None)
            if r is None:
                continue
            label = seq if first else ""
            first = False
            a(r"{} & {} & {} & {} & {:.1f} & {:.2f} \\".format(
                label, _tex(CONFIG_LABEL[cfg]), _fmt(r["bd_rate"]),
                _fmt(r["bd_psnr"]), r["ts_pct"], r["speedup"]))
        a(r"\midrule")
    a(r"\multicolumn{6}{l}{\textbf{Average over sequences}} \\")
    a(r"\midrule")
    for r in avg:
        a(r" & {} & {} & {} & {:.1f} & {:.2f} \\".format(
            _tex(CONFIG_LABEL[r["config"]]), _fmt(r["bd_rate"]),
            _fmt(r["bd_psnr"]), r["ts_pct"], r["speedup"]))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")
    a("")

    # Frontier table: BD-Rate vs TS%, the "where does the proposal pay off"
    # positioning of the two ML points against the native cpu-used knob.
    a(r"\begin{table}[t]")
    a(r"\centering")
    a(r"\caption{BD-Rate$\times$time frontier (average over CTC A1). The ML"
      r" pruner (\texttt{cpu-used=0}) versus the encoder's native speed knob.}")
    a(r"\label{tab:fase6-frontier}")
    a(r"\begin{tabular}{lrr}")
    a(r"\toprule")
    a(r"Configuration & BD-Rate (\%) & TS (\%) \\")
    a(r"\midrule")
    for r in avg:
        a(r"{} & {} & {:.1f} \\".format(
            _tex(CONFIG_LABEL[r["config"]]), _fmt(r["bd_rate"]), r["ts_pct"]))
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def print_console(per_seq, avg):
    """Human-readable echo of the same numbers (the smoke-test table style)."""
    print("\n=== Fase 6 CTC -- per sequence (anchor = libaom original cpu0) ===")
    hdr = "{:<14}{:<22}{:>12}{:>12}{:>9}{:>10}".format(
        "Sequence", "Config", "BD-Rate(%)", "BD-PSNR(dB)", "TS(%)", "Speedup")
    seqs = sorted({r["seq"] for r in per_seq})
    for seq in seqs:
        print("-" * len(hdr))
        print(hdr if seq == seqs[0] else "")
        first = True
        for cfg in CONFIG_ORDER:
            r = next((x for x in per_seq
                      if x["seq"] == seq and x["config"] == cfg), None)
            if r is None:
                continue
            print("{:<14}{:<22}{:>12}{:>12}{:>9.1f}{:>9.2f}x".format(
                seq if first else "", CONFIG_LABEL[cfg], _fmt(r["bd_rate"]),
                _fmt(r["bd_psnr"]), r["ts_pct"], r["speedup"]))
            first = False
    print("\n=== Average over sequences ===")
    print(hdr)
    for r in avg:
        print("{:<14}{:<22}{:>12}{:>12}{:>9.1f}{:>9.2f}x".format(
            "", CONFIG_LABEL[r["config"]], _fmt(r["bd_rate"]),
            _fmt(r["bd_psnr"]), r["ts_pct"], r["speedup"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--csv", default=None,
                   help="raw_results.csv (default <out-dir>/raw_results.csv)")
    args = p.parse_args()

    csv_path = args.csv or os.path.join(args.out_dir, "raw_results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit("no raw results at " + csv_path)
    data = load(csv_path)
    per_seq = per_seq_rows(data)
    avg = averages(per_seq)

    write_csv(os.path.join(args.out_dir, "bdrate_per_seq.csv"), per_seq,
              ["seq", "config", "bd_rate", "bd_psnr", "ts_pct", "speedup"])
    write_csv(os.path.join(args.out_dir, "bdrate_average.csv"), avg,
              ["config", "bd_rate", "bd_psnr", "ts_pct", "speedup"])
    write_tex(os.path.join(args.out_dir, "tables.tex"), per_seq, avg)
    print_console(per_seq, avg)
    print("\nWrote bdrate_per_seq.csv, bdrate_average.csv, tables.tex to "
          + args.out_dir)


if __name__ == "__main__":
    main()
