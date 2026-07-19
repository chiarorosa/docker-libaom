#!/usr/bin/env python3
"""Cross-batch CTC analysis: CQ-regime decomposition, paired significance tests,
the 8-sequence Pareto frontier, and the two competing TS definitions.

Motivation. The per-config averages over the full CTC grid hide three things:

  A2  The ML and native levers have OPPOSITE dependence on CQ. Averaging over
      the four CQ points cancels a real effect: at high quality (CQ 20+32) the
      H9c swap is better than the native CNN, while at low rate it is not.
  D3  Differences of 0.02-0.03 pp in BD-rate are reported as if they were
      structure. A per-sequence paired test says which survive.
  A6  The published frontier used 3 of the 8 measured sequences.
  D4  Two TS definitions coexist in the docs, differing by up to ~2 pp -- the
      same magnitude as the gaps being argued about.

Everything here is analysis over encodes already on disk; nothing is re-encoded.
All configurations are referenced to the same anchor (libaom original, cpu0),
which lives only in the Fase 6 CSV -- the swap batches carry no anchor of their
own.

Usage:
  python analyze_frontier.py --out-dir results/benchmark/fase6_analysis
"""

import argparse
import csv
import itertools
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "benchmark"))
for _p in (HERE, BENCH):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from bd_rate import bd_rate  # noqa: E402

DEFAULT_CSVS = [
    "/workspace/results/benchmark/fase6/raw_results.csv",
    "/workspace/results/benchmark/fase6_swap/raw_results.csv",
    "/workspace/results/benchmark/fase6_swap_h9c/raw_results.csv",
]
# The two CQ regimes. High quality / high rate first.
REGIMES = [("cq20+32", (20, 32)), ("cq43+55", (43, 55))]


def kbps(row):
    fps = int(row["fps_num"]) / int(row["fps_den"])
    return int(row["bytes"]) * 8 * fps / int(row["frames"]) / 1000.0


def load(paths):
    """data[seq][config][cq] = {rate, psnr, time}; later files do not override."""
    data = {}
    for path in paths:
        if not os.path.exists(path):
            print("[warn] missing, skipped: " + path, file=sys.stderr)
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                cfg = data.setdefault(row["seq"], {}).setdefault(
                    row["config"], {})
                cq = int(row["cq"])
                if cq not in cfg:
                    cfg[cq] = {"rate": kbps(row), "psnr": float(row["psnr_y"]),
                               "time": float(row["time_s"])}
    return data


def curve(pts, cqs):
    cqs = [c for c in sorted(cqs) if c in pts]
    return ([pts[c]["rate"] for c in cqs], [pts[c]["psnr"] for c in cqs], cqs)


def evaluate(anchor, pts, cqs):
    """BD-rate + TS + speedup of `pts` vs `anchor`, restricted to `cqs`."""
    ra, qa, ca = curve(anchor, cqs)
    rt, qt, ct = curve(pts, cqs)
    shared = [c for c in ca if c in ct]
    if len(shared) < 2:
        return None
    ts = statistics.mean((anchor[c]["time"] - pts[c]["time"])
                         / anchor[c]["time"] * 100.0 for c in shared)
    spd = statistics.mean(anchor[c]["time"] / pts[c]["time"] for c in shared)
    return {"bd_rate": bd_rate(ra, qa, rt, qt), "ts_pct": ts, "speedup": spd,
            "n_cq": len(shared)}


def paired_test(deltas):
    """Two-sided paired t-test against zero. Returns (mean, se, t, n).

    scipy is available in the project venv, but the statistic is one line and
    keeping it explicit makes the degrees of freedom auditable.
    """
    n = len(deltas)
    if n < 2:
        return (float("nan"),) * 3 + (n,)
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    se = sd / (n ** 0.5)
    t = m / se if se > 0 else float("nan")
    return m, se, t, n


def p_from_t(t, df):
    """Two-sided p-value; scipy if present, else a readable bracket."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csvs", nargs="+", default=DEFAULT_CSVS)
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_analysis")
    p.add_argument("--anchor", default="anchor")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = load(args.csvs)
    seqs = sorted(s for s in data if args.anchor in data[s])
    all_cq = sorted({c for s in seqs for cfg in data[s].values() for c in cfg})
    print("sequences with anchor: {} ({})".format(len(seqs), ", ".join(seqs)))
    print("CQ grid: {}".format(all_cq))

    configs = sorted({c for s in seqs for c in data[s]} - {args.anchor})
    # Only configurations measured on every anchored sequence are comparable.
    full = [c for c in configs
            if all(len(data[s].get(c, {})) == len(all_cq) for s in seqs)]
    partial = [c for c in configs if c not in full]
    print("\nconfigs complete on {}/{} seqs: {}".format(
        len(seqs), len(seqs), len(full)))
    print("configs partial (excluded from tests): {}".format(
        ", ".join(partial) if partial else "none"))

    # ---- per (seq, config, regime) -------------------------------------
    rows = []
    for s in seqs:
        anc = data[s][args.anchor]
        for c in full:
            for name, cqs in [("all", tuple(all_cq))] + REGIMES:
                r = evaluate(anc, data[s][c], cqs)
                if r:
                    rows.append(dict(seq=s, config=c, regime=name, **r))

    with open(os.path.join(args.out_dir, "cq_decomposition.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seq", "config", "regime", "bd_rate",
                                          "ts_pct", "speedup", "n_cq"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    def avg(cfg, regime, key):
        v = [r[key] for r in rows if r["config"] == cfg and r["regime"] == regime]
        return statistics.mean(v) if v else float("nan")

    print("\n=== A2. BD-rate / TS by CQ regime (mean over {} seqs) ==="
          .format(len(seqs)))
    print("{:<20}{:>18}{:>18}{:>18}".format(
        "config", "all: BD / TS", "cq20+32: BD / TS", "cq43+55: BD / TS"))
    for c in full:
        print("{:<20}{:>9.3f} /{:>6.2f}{:>9.3f} /{:>6.2f}{:>9.3f} /{:>6.2f}"
              .format(c, avg(c, "all", "bd_rate"), avg(c, "all", "ts_pct"),
                      avg(c, "cq20+32", "bd_rate"), avg(c, "cq20+32", "ts_pct"),
                      avg(c, "cq43+55", "bd_rate"), avg(c, "cq43+55", "ts_pct")))

    # ---- A2b: TS per individual CQ (where the two levers cross) ---------
    print("\n=== A2b. TS (%) per individual CQ (mean over {} seqs) ==="
          .format(len(seqs)))
    print("{:<20}".format("config") + "".join(
        "{:>9}".format("cq" + str(q)) for q in all_cq))
    percq = []
    for c in full:
        rec = {"config": c}
        cells = []
        for q in all_cq:
            v = [(data[s][args.anchor][q]["time"] - data[s][c][q]["time"])
                 / data[s][args.anchor][q]["time"] * 100.0
                 for s in seqs if q in data[s].get(c, {})]
            m = statistics.mean(v) if v else float("nan")
            rec["ts_cq{}".format(q)] = m
            cells.append("{:>9.2f}".format(m))
        percq.append(rec)
        print("{:<20}".format(c) + "".join(cells))
    with open(os.path.join(args.out_dir, "ts_per_cq.csv"), "w",
              newline="") as f:
        keys = ["config"] + ["ts_cq{}".format(q) for q in all_cq]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in percq:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # ---- D3: paired tests, every swap config vs native at matched level --
    def level_of(cfg):
        _, _, lvl = cfg.rpartition("_cpu")
        return int(lvl) if lvl.isdigit() else 0

    tests = []
    for c in full:
        lvl = level_of(c)
        ref = "native_cpu{}".format(lvl) if lvl else None
        if not ref or ref == c or ref not in full:
            continue
        for regime, _ in [("all", None)] + [(n, None) for n, _ in REGIMES]:
            for key in ("bd_rate", "ts_pct"):
                d = []
                for s in seqs:
                    a = next((r for r in rows if r["seq"] == s
                              and r["config"] == c and r["regime"] == regime), None)
                    b = next((r for r in rows if r["seq"] == s
                              and r["config"] == ref and r["regime"] == regime), None)
                    if a and b:
                        d.append(a[key] - b[key])
                m, se, t, n = paired_test(d)
                tests.append({"config": c, "vs": ref, "regime": regime,
                              "metric": key, "mean_delta": m, "se": se,
                              "t": t, "p": p_from_t(t, n - 1), "n": n,
                              "n_better": sum(1 for x in d if x < 0)})

    with open(os.path.join(args.out_dir, "paired_tests.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "vs", "regime", "metric",
                                          "mean_delta", "se", "t", "p", "n",
                                          "n_better"])
        w.writeheader()
        for r in tests:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\n=== D3. Paired tests vs native at matched cpu-used "
          "(BD-rate; negative delta = swap better) ===")
    print("{:<20}{:<10}{:>10}{:>9}{:>9}{:>8}{:>10}".format(
        "config", "regime", "dBD(pp)", "se", "t", "p", "better"))
    for r in tests:
        if r["metric"] != "bd_rate":
            continue
        sig = "*" if r["p"] == r["p"] and r["p"] < 0.05 else " "
        print("{:<20}{:<10}{:>10.3f}{:>9.3f}{:>9.2f}{:>8.3f}{}{:>8}".format(
            r["config"], r["regime"], r["mean_delta"], r["se"], r["t"],
            r["p"], sig, "{}/{}".format(r["n_better"], r["n"])))

    # ---- A6: Pareto frontier over all complete configs ------------------
    pts = [{"config": c, "bd_rate": avg(c, "all", "bd_rate"),
            "ts_pct": avg(c, "all", "ts_pct"),
            "speedup": avg(c, "all", "speedup")} for c in full]
    for a in pts:
        # dominated: someone has BD <= and TS >=, strictly better in one axis
        a["dominated_by"] = ";".join(
            b["config"] for b in pts
            if b is not a and b["bd_rate"] <= a["bd_rate"]
            and b["ts_pct"] >= a["ts_pct"]
            and (b["bd_rate"] < a["bd_rate"] or b["ts_pct"] > a["ts_pct"]))
    with open(os.path.join(args.out_dir, "pareto_frontier.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "bd_rate", "ts_pct",
                                          "speedup", "dominated_by"])
        w.writeheader()
        for r in sorted(pts, key=lambda x: x["ts_pct"]):
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\n=== A6. Pareto frontier, {} seqs, {} configs "
          "(sorted by TS) ===".format(len(seqs), len(pts)))
    print("{:<20}{:>10}{:>9}{:>10}  {}".format(
        "config", "BD(%)", "TS(%)", "speedup", "dominated by"))
    for r in sorted(pts, key=lambda x: x["ts_pct"]):
        print("{:<20}{:>10.3f}{:>9.2f}{:>10.3f}  {}".format(
            r["config"], r["bd_rate"], r["ts_pct"], r["speedup"],
            r["dominated_by"] or "-- (non-dominated)"))

    # ---- D4: the two TS definitions -------------------------------------
    print("\n=== D4. TS definitions compared ===")
    print("canonical : mean over CQ of (1 - t_cfg/t_anchor), then over seqs")
    print("sintese   : mean over seqs of (1 - sum_CQ t_cfg / sum_CQ t_anchor)")
    print("{:<20}{:>12}{:>12}{:>10}".format(
        "config", "TS canon", "TS sintese", "delta pp"))
    d4 = []
    for c in full:
        canon, sint = [], []
        for s in seqs:
            anc, pt = data[s][args.anchor], data[s][c]
            shared = [q for q in all_cq if q in anc and q in pt]
            canon.append(statistics.mean(
                (anc[q]["time"] - pt[q]["time"]) / anc[q]["time"] * 100.0
                for q in shared))
            sint.append((1 - sum(pt[q]["time"] for q in shared)
                         / sum(anc[q]["time"] for q in shared)) * 100.0)
        a, b = statistics.mean(canon), statistics.mean(sint)
        d4.append({"config": c, "ts_canonical": a, "ts_sintese": b,
                   "delta_pp": b - a})
        print("{:<20}{:>12.2f}{:>12.2f}{:>10.2f}".format(c, a, b, b - a))
    with open(os.path.join(args.out_dir, "ts_definitions.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "ts_canonical",
                                          "ts_sintese", "delta_pp"])
        w.writeheader()
        for r in d4:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    print("\nWrote cq_decomposition.csv, paired_tests.csv, pareto_frontier.csv,"
          " ts_definitions.csv to " + args.out_dir)


if __name__ == "__main__":
    main()
