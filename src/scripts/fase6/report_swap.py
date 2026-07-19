#!/usr/bin/env python3
"""Fase 6 extension report -- native CNN vs a swapped-in pruner at matched cpu-used.

Handles any swap batch (H9a from encode_swap.py, H9c from encode_swap_h9c.py):
the configurations are discovered from the CSV rather than hard-coded, because
the previous fixed list silently discarded every H9c row -- 192 encodes sat
unaggregated for that reason. See docs/RESULTADOS_fase6_swap_h9c.md.

Reads the swap batch (native_cpuN + <kind>_cpuN) and
references BD-Rate/BD-PSNR/TS/speedup to the SAME cpu-used=0 anchor as Fase 6
(results/benchmark/fase6/raw_results.csv). Groups the output by cpu-used level so
each block shows, on one common axis, the native CNN pruner against the two H9a
operating points -- answering: at a given speed preset, is the H9a student a
better partition pruner than libaom's shipped CNN?

Because native_cpuN and the H9a swaps share every speed feature except the
pruner, the within-level differences are attributable to the pruner alone. Also
prints the direct native->H9a swap delta (ΔBD-Rate, ΔTS) per level.

Outputs: swap_per_seq.csv, swap_average.csv, swap_tables.tex, console echo.
"""

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "benchmark"))
for pth in (HERE, BENCH):
    if pth not in sys.path:
        sys.path.insert(0, pth)
from bd_rate import bd_rate  # noqa: E402
from report_ctc import bd_psnr  # noqa: E402  reuse the BD-PSNR helper

LABEL = {
    "native": "libaom CNN (native SOTA)",
    "h9a_bal": "H9a balanced (swap)",
    "h9a_aggr": "H9a aggressive (swap)",
    "h9c_tau90": "H9c tau=0.90 (swap)",
    "h9c_tau95": "H9c tau=0.95 (swap)",
}
# Presentation order within each cpu-used level. Kinds present in the swap CSV
# but absent here are appended (sorted) rather than dropped: the previous
# hard-coded list silently discarded every H9c row, which is why 192 encodes
# sat unaggregated. `native` always leads -- it is the reference of each block.
KIND_ORDER = ["native", "h9a_bal", "h9a_aggr", "h9c_tau90", "h9c_tau95"]


def label_of(kind):
    return LABEL.get(kind, kind)


def order_kinds(kinds):
    """Canonical order first, then any unknown kind, alphabetically."""
    known = [k for k in KIND_ORDER if k in kinds]
    return known + sorted(k for k in kinds if k not in KIND_ORDER)


def _kbps(row):
    fps = int(row["fps_num"]) / int(row["fps_den"])
    return int(row["bytes"]) * 8 * fps / int(row["frames"]) / 1000.0


def load_csv(path):
    """data[seq][config] = {cq: {rate, psnr, time}}."""
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data.setdefault(row["seq"], {}).setdefault(
                row["config"], {})[int(row["cq"])] = {
                    "rate": _kbps(row), "psnr": float(row["psnr_y"]),
                    "time": float(row["time_s"])}
    return data


def curve(pts):
    cqs = sorted(pts)
    return ([pts[c]["rate"] for c in cqs], [pts[c]["psnr"] for c in cqs])


def parse_cfg(name):
    """'h9a_bal_cpu2' -> ('h9a_bal', 2); 'native_cpu1' -> ('native', 1)."""
    kind, _, lvl = name.rpartition("_cpu")
    return kind, int(lvl)


def _eval(seq, lvl, kind, pts, ra, qa, atime):
    """One row: pruner config vs the cpu0 anchor (rate/psnr curve + times)."""
    rt, qt = curve(pts)
    shared = [c for c in pts if c in atime]
    ts = statistics.mean((atime[c] - pts[c]["time"]) / atime[c] * 100.0
                         for c in shared)
    spd = statistics.mean(atime[c] / pts[c]["time"] for c in shared)
    return {"seq": seq, "level": lvl, "kind": kind,
            "bd_rate": bd_rate(ra, qa, rt, qt),
            "bd_psnr": bd_psnr(ra, qa, rt, qt), "ts_pct": ts, "speedup": spd}


def per_seq_rows(h9a, fase6, levels):
    """native_cpuN comes from the Fase 6 CSV (reused); h9a from the swap CSV.
    Both are referenced to the same Fase 6 cpu-used=0 anchor."""
    rows = []
    for seq in sorted(h9a):
        if seq not in fase6 or "anchor" not in fase6[seq]:
            print("[warn] {}: no Fase 6 cpu0 anchor, skipped".format(seq),
                  file=sys.stderr)
            continue
        ra, qa = curve(fase6[seq]["anchor"])
        atime = {c: fase6[seq]["anchor"][c]["time"] for c in fase6[seq]["anchor"]}
        # native reference (reused from Fase 6)
        for lvl in levels:
            ncfg = "native_cpu{}".format(lvl)
            if ncfg in fase6[seq]:
                rows.append(_eval(seq, lvl, "native", fase6[seq][ncfg],
                                  ra, qa, atime))
            else:
                print("[warn] {}: {} missing in Fase 6 CSV".format(seq, ncfg),
                      file=sys.stderr)
        # swapped pruners (this experiment) -- every kind found in the CSV
        for cfg, pts in h9a[seq].items():
            kind, lvl = parse_cfg(cfg)
            if lvl in levels and kind != "native":
                rows.append(_eval(seq, lvl, kind, pts, ra, qa, atime))
    return rows


def averages(rows, levels):
    out = []
    kinds = order_kinds({r["kind"] for r in rows})
    for lvl in levels:
        for kind in kinds:
            sub = [r for r in rows if r["level"] == lvl and r["kind"] == kind]
            if not sub:
                continue
            out.append({"level": lvl, "kind": kind,
                        "bd_rate": statistics.mean(r["bd_rate"] for r in sub),
                        "bd_psnr": statistics.mean(r["bd_psnr"] for r in sub),
                        "ts_pct": statistics.mean(r["ts_pct"] for r in sub),
                        "speedup": statistics.mean(r["speedup"] for r in sub)})
    return out


def _get(avg, lvl, kind):
    return next((a for a in avg if a["level"] == lvl and a["kind"] == kind),
                None)


def write_csv(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in keys})


def _f(x):
    return "{:+.2f}".format(x) if x == x else "--"


def console(avg, levels):
    print("\n=== Fase 6 SWAP -- native CNN vs H9a, by cpu-used "
          "(anchor = libaom original cpu0) ===")
    hdr = "{:<26}{:>12}{:>12}{:>9}{:>10}".format(
        "Config", "BD-Rate(%)", "BD-PSNR(dB)", "TS(%)", "Speedup")
    kinds = order_kinds({a["kind"] for a in avg})
    for lvl in levels:
        print("\n-- cpu-used={} --".format(lvl))
        print(hdr)
        for kind in kinds:
            a = _get(avg, lvl, kind)
            if not a:
                continue
            print("{:<26}{:>12}{:>12}{:>9.1f}{:>9.2f}x".format(
                label_of(kind), _f(a["bd_rate"]), _f(a["bd_psnr"]),
                a["ts_pct"], a["speedup"]))
        nat = _get(avg, lvl, "native")
        if nat:
            for kind in kinds:
                if kind == "native":
                    continue
                s = _get(avg, lvl, kind)
                if s:
                    print("   -> {} vs native: dBD-Rate={:+.2f}%  dTS={:+.1f}pp"
                          .format(kind, s["bd_rate"] - nat["bd_rate"],
                                  s["ts_pct"] - nat["ts_pct"]))


def write_tex(path, avg, levels):
    L = []
    a = L.append
    a("% Fase 6 extension -- native CNN vs H9a swap at matched cpu-used.")
    a("% Anchor = libaom original cpu-used=0. Within a level only the partition")
    a("% pruner differs, so gaps are attributable to the pruner.")
    a(r"\begin{table}[t]\centering")
    a(r"\caption{H9a student as a drop-in replacement for libaom's native intra"
      r" CNN partition pruner (AOM-CTC All Intra, Class~A1, 4K, 10-bit; anchor:"
      r" libaom \texttt{cpu-used=0}). Within each \texttt{cpu-used} level every"
      r" speed feature is identical except the pruner.}")
    a(r"\label{tab:fase6-swap}")
    a(r"\begin{tabular}{llrrrr}")
    a(r"\toprule")
    a(r"cpu-used & Pruner & BD-Rate (\%) & BD-PSNR (dB) & TS (\%) & Speedup$\times$ \\")
    a(r"\midrule")
    kinds = order_kinds({x["kind"] for x in avg})
    for lvl in levels:
        first = True
        for kind in kinds:
            av = _get(avg, lvl, kind)
            if not av:
                continue
            a(r"{} & {} & {} & {} & {:.1f} & {:.2f} \\".format(
                lvl if first else "", label_of(kind), _f(av["bd_rate"]),
                _f(av["bd_psnr"]), av["ts_pct"], av["speedup"]))
            first = False
        a(r"\midrule")
    a(r"\bottomrule")
    a(r"\end{tabular}\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_swap")
    p.add_argument("--swap-csv", default=None)
    p.add_argument("--anchor-csv",
                   default="/workspace/results/benchmark/fase6/raw_results.csv",
                   help="Fase 6 CSV holding the cpu0 anchor + native_cpuN")
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    swap_csv = args.swap_csv or os.path.join(args.out_dir, "raw_results.csv")
    for f in (swap_csv, args.anchor_csv):
        if not os.path.exists(f):
            raise SystemExit("missing CSV: " + f)
    h9a = load_csv(swap_csv)
    fase6 = load_csv(args.anchor_csv)  # holds cpu0 anchor + native_cpuN

    rows = per_seq_rows(h9a, fase6, args.levels)
    avg = averages(rows, args.levels)
    write_csv(os.path.join(args.out_dir, "swap_per_seq.csv"), rows,
              ["seq", "level", "kind", "bd_rate", "bd_psnr", "ts_pct", "speedup"])
    write_csv(os.path.join(args.out_dir, "swap_average.csv"), avg,
              ["level", "kind", "bd_rate", "bd_psnr", "ts_pct", "speedup"])
    write_tex(os.path.join(args.out_dir, "swap_tables.tex"), avg, args.levels)
    console(avg, args.levels)
    print("\nWrote swap_per_seq.csv, swap_average.csv, swap_tables.tex to "
          + args.out_dir)


if __name__ == "__main__":
    main()
