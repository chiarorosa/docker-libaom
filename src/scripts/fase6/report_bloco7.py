#!/usr/bin/env python3
"""Bloco 7 — relatorio de E1 (h9c em 8/8 seqs) e E4 (decomposicao do confound).

E1. As linhas h9c_tau90/95 em cpu-used=0 existiam em 6 das 8 sequencias CTC; o
E1 completou Crosswalk e NocturneDance. Aqui a curva BD-rate x TS do H9c e
fechada sobre as 8, ao lado do H9a implantado e do knob nativo, para sustentar
a Conclusao 2 (o extremo de baixo BD) sobre o conjunto inteiro.

E4. As medicoes h9c_tau{N} rodaram com o estudante H9a nos seus defaults
compilados (tau_none/split=0.9), logo mediam H9a+H9c empilhados, nao H9c. As
linhas h9ciso_tau90 repetem o ponto com o H9a neutralizado (tau=2/2/-1, nunca
dispara). A diferenca entre as duas e exatamente a contribuicao do H9a@default
que fora atribuida ao H9c. Ate agora isso repousava numa unica sequencia
(Neon1224); o E4 acrescenta tres.

Reproducao:
    /workspace/build/venv-ml/bin/python src/scripts/fase6/report_bloco7.py
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
from bd_rate import bd_rate  # noqa: E402

CQS = (20, 32, 43, 55)
E1_CONFIGS = ["h9c_tau90", "h9c_tau95", "ml_balanced", "ml_bal_h9d",
              "native_cpu1"]
E4_PAIRS = [("h9c_tau90", "h9ciso_tau90")]


def load(csv_path):
    """d[seq][config][cq] = {"rate": kbps, "psnr": psnr_y, "time": s}."""
    d = {}
    for r in csv.DictReader(open(csv_path, newline="")):
        fps = int(r["fps_num"]) / int(r["fps_den"])
        kbps = int(r["bytes"]) * 8 * fps / int(r["frames"]) / 1000.0
        d.setdefault(r["seq"], {}).setdefault(r["config"], {})[int(r["cq"])] = {
            "rate": kbps, "psnr": float(r["psnr_y"]), "time": float(r["time_s"]),
        }
    return d


def complete(cfgs, name):
    """The config's four-CQ curve, or None if any QP is missing."""
    pts = cfgs.get(name)
    return pts if pts and all(c in pts for c in CQS) else None


def metrics(anchor, pts):
    """(bd_rate, ts_pct, speedup) of pts against the anchor curve."""
    ra = [anchor[c]["rate"] for c in CQS]
    qa = [anchor[c]["psnr"] for c in CQS]
    rt = [pts[c]["rate"] for c in CQS]
    qt = [pts[c]["psnr"] for c in CQS]
    ts = statistics.mean((anchor[c]["time"] - pts[c]["time"]) / anchor[c]["time"]
                         * 100.0 for c in CQS)
    spd = statistics.mean(anchor[c]["time"] / pts[c]["time"] for c in CQS)
    return bd_rate(ra, qa, rt, qt), ts, spd


def report_e1(d):
    print("=" * 78)
    print("E1 - H9c em cpu-used=0 sobre as 8 sequencias CTC (vs anchor)")
    print("=" * 78)
    print("{:<15}{:<14}{:>11}{:>9}{:>10}".format(
        "Sequencia", "Config", "BD-rate(%)", "TS(%)", "Speedup"))
    acc = {}
    for seq in sorted(d):
        anchor = complete(d[seq], "anchor")
        if not anchor:
            continue
        first = True
        for cfg in E1_CONFIGS:
            pts = complete(d[seq], cfg)
            if not pts:
                continue
            bd, ts, spd = metrics(anchor, pts)
            acc.setdefault(cfg, []).append((bd, ts, spd))
            print("{:<15}{:<14}{:>+11.3f}{:>9.1f}{:>9.2f}x".format(
                seq if first else "", cfg, bd, ts, spd))
            first = False
        print("-" * 78)
    print("{:<15}{:<14}{:>11}{:>9}{:>10}".format(
        "MEDIA", "", "", "", ""))
    for cfg in E1_CONFIGS:
        v = acc.get(cfg)
        if not v:
            continue
        print("{:<15}{:<14}{:>+11.3f}{:>9.1f}{:>9.2f}x   (n={})".format(
            "", cfg, statistics.mean(x[0] for x in v),
            statistics.mean(x[1] for x in v),
            statistics.mean(x[2] for x in v), len(v)))


def report_e4(d):
    print()
    print("=" * 78)
    print("E4 - decomposicao do confound H9a/H9c (confundido vs isolado)")
    print("=" * 78)
    for conf, iso in E4_PAIRS:
        print("{:<15}{:>22}{:>22}{:>16}".format(
            "Sequencia", conf + " (H9a ativo)", iso + " (H9a inerte)",
            "atribuivel H9a"))
        rows = []
        for seq in sorted(d):
            anchor = complete(d[seq], "anchor")
            a = complete(d[seq], conf)
            b = complete(d[seq], iso)
            if not (anchor and a and b):
                continue
            bd_a, ts_a, _ = metrics(anchor, a)
            bd_b, ts_b, _ = metrics(anchor, b)
            share = (ts_a - ts_b) / ts_a * 100.0 if ts_a > 1e-9 else float("nan")
            rows.append((ts_a, ts_b, share))
            print("{:<15}{:>11.3f}% {:>9.1f}%{:>11.3f}% {:>9.1f}%{:>9.1f} pp"
                  "{:>7.0f}%".format(
                      seq, bd_a, ts_a, bd_b, ts_b, ts_a - ts_b, share))
        if rows:
            print("-" * 78)
            print("{:<15}{:>22}{:>22}{:>9.1f} pp{:>7.0f}%".format(
                "MEDIA", "", "", statistics.mean(r[0] - r[1] for r in rows),
                statistics.mean(r[2] for r in rows)))
            print()
            print("Leitura: 'atribuivel H9a' e a fracao do TS medido no ponto "
                  "confundido que\ndesaparece quando o H9a e neutralizado, ou "
                  "seja, que nunca foi do H9c.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",
                   default="/workspace/results/benchmark/fase6/raw_results.csv")
    args = p.parse_args()
    d = load(args.csv)
    report_e1(d)
    report_e4(d)


if __name__ == "__main__":
    main()
