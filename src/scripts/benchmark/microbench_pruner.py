#!/usr/bin/env python3
"""Pruner cost microbenchmark: per-call inference AND aggregate cost per encode.

Extends the 2026-07-17 measurement (docs/RESULTADOS_microbench_pruner.md), which
reported ns/call for the model forward pass alone and explicitly excluded two
things: the MLP's feature extraction, and the invocation frequency.

Both exclusions matter for the deployment claim. The native CNN is invoked once
per superblock and decides the whole quad-tree; the MLP is invoked once per
modeled node. So a per-call ratio cannot be read as "the deployed pruner is ~50x
cheaper" -- that requires the aggregate.

This script measures the aggregate directly rather than extrapolating: the C
instrumentation already accumulates total_ns and call counts per pruner,
including a separate accumulator for H9a feature preparation
(g_pt_h9a_feat, partition_strategy.c:156 / :2128-2134), which had never been
read. Totals over a whole encode need no assumption about nodes per superblock.

Caveat carried from the C: H9c's own call to student_node_features
(partition_strategy.c:2163) sits OUTSIDE any timer -- only av1_nn_predict is
timed (:2171-2173). H9c's feature cost is therefore not separable here; its
extraction is the same function H9a uses, so H9a's figure is the proxy.

Usage (container):
  python microbench_pruner.py --out results/benchmark/microbench/pruner_cost.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

CTC = "/workspace/src/samples/aomctc_test_set"
# (label, y4m, cq, bit_depth) -- two runs on different content/QP, as in the
# original measurement, to show the ratios are not a one-off.
RUNS = [
    ("Tango_cq32", CTC + "/Tango_3840x2160_5994fps_10bit_420.y4m", 32, 10),
    ("BoxingPractice_cq43",
     CTC + "/BoxingPractice_3840x2160_5994fps_10bit_420.y4m", 43, 10),
]
ROW_RE = re.compile(
    r"^\s*(\S+.*?)\s+calls=(\d+)\s+total_ms=\s*([\d.]+)\s+ns/call=\s*([\d.]+)")


def run_one(enc, seq, cq, bit_depth, frames, cpu_used, h9c):
    env = dict(os.environ, AV1_PRUNER_TIMING="1")
    if h9c:
        env["AV1_STUDENT_H9C_ENABLE"] = "1"
        env["AV1_STUDENT_H9C_TAU"] = "0.90"
    cmd = [
        enc, "--cpu-used={}".format(cpu_used), "--passes=1", "--end-usage=q",
        "--cq-level={}".format(cq), "--kf-min-dist=0", "--kf-max-dist=0",
        "--deltaq-mode=0", "--enable-tpl-model=0",
        "--enable-keyframe-filtering=0",
        "--tile-columns=1", "--tile-rows=0", "--threads=1", "--row-mt=0",
        "--bit-depth={}".format(bit_depth), "--limit={}".format(frames),
        "--psnr", "--obu", "-o", "/tmp/_microbench.obu", seq,
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    wall_s = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-800:] + "\n")
        raise SystemExit("encode failed: " + os.path.basename(seq))
    text = r.stderr.decode(errors="replace")
    out = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            out[m.group(1).strip()] = {
                "calls": int(m.group(2)), "total_ms": float(m.group(3)),
                "ns_per_call": float(m.group(4))}
    if not out:
        sys.stderr.write(text[-800:] + "\n")
        raise SystemExit("no PRUNER_TIMING block parsed")
    out["_wall_s"] = wall_s
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--cpu-used", type=int, default=1,
                   help="1+ so the native CNN speed feature is active")
    p.add_argument("--out",
                   default="/workspace/results/benchmark/microbench/pruner_cost.csv")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    CNN, INF, FEAT, H9C = ("native_cnn", "h9a_infer (MLP)",
                           "h9a_feature_prep", "h9c_infer (MLP)")
    rows = []
    for label, seq, cq, bd in RUNS:
        print("=== {} ===".format(label), flush=True)
        acc = run_one(args.enc, seq, cq, bd, args.frames, args.cpu_used, True)
        for k in (CNN, INF, FEAT, H9C):
            a = acc.get(k)
            if a:
                print("  {:<20} calls={:<9} total_ms={:9.3f}  ns/call={:9.1f}"
                      .format(k, a["calls"], a["total_ms"], a["ns_per_call"]))
                rows.append({"run": label, "pruner": k, **a})

        cnn, inf, feat = acc.get(CNN), acc.get(INF), acc.get(FEAT)
        if cnn and inf and feat:
            nodes = inf["calls"] / cnn["calls"]
            mlp_call = inf["ns_per_call"] + feat["ns_per_call"]
            mlp_total = inf["total_ms"] + feat["total_ms"]
            print("  --")
            print("  MLP calls per CNN call (nodes/SB) : {:.2f}".format(nodes))
            print("  ratio, inference only (per call)  : {:.1f}x"
                  .format(cnn["ns_per_call"] / inf["ns_per_call"]))
            print("  ratio, +feature prep (per call)   : {:.1f}x"
                  .format(cnn["ns_per_call"] / mlp_call))
            print("  ratio, AGGREGATE over the encode  : {:.2f}x"
                  .format(cnn["total_ms"] / mlp_total))
            print("  feature prep / inference          : {:.1f}x"
                  .format(feat["ns_per_call"] / inf["ns_per_call"]))
            # Absolute weight: a dramatic ratio between two terms that are both
            # a rounding error of the encode changes no deployment conclusion.
            wall_ms = acc["_wall_s"] * 1000.0
            print("  encode wall time                  : {:.1f} s"
                  .format(acc["_wall_s"]))
            print("  CNN path as % of encode           : {:.2f}%"
                  .format(100.0 * cnn["total_ms"] / wall_ms))
            print("  H9a path (infer+feat) as % of enc : {:.2f}%"
                  .format(100.0 * mlp_total / wall_ms))
            rows.append({"run": label, "pruner": "_derived",
                         "calls": 0, "total_ms": 0.0, "ns_per_call": 0.0,
                         "nodes_per_sb": round(nodes, 3),
                         "ratio_inference_only": round(
                             cnn["ns_per_call"] / inf["ns_per_call"], 2),
                         "ratio_with_features": round(
                             cnn["ns_per_call"] / mlp_call, 2),
                         "ratio_aggregate": round(
                             cnn["total_ms"] / mlp_total, 3),
                         "feat_over_infer": round(
                             feat["ns_per_call"] / inf["ns_per_call"], 2),
                         "encode_wall_s": round(acc["_wall_s"], 2),
                         "cnn_pct_of_encode": round(
                             100.0 * cnn["total_ms"] / (acc["_wall_s"] * 1000), 4),
                         "h9a_pct_of_encode": round(
                             100.0 * mlp_total / (acc["_wall_s"] * 1000), 4)})

    keys = ["run", "pruner", "calls", "total_ms", "ns_per_call", "nodes_per_sb",
            "ratio_inference_only", "ratio_with_features", "ratio_aggregate",
            "feat_over_infer", "encode_wall_s", "cnn_pct_of_encode",
            "h9a_pct_of_encode"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\nWrote " + args.out)


if __name__ == "__main__":
    main()
