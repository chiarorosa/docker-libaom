#!/usr/bin/env python3
"""Deployed overhead of every NPL-AV1 pruner as a share of the encoding time.

Extends the measurement of docs/RESULTADOS_microbench_pruner.md §6 (pre-search
pruner only) to the post-NONE and the extended pruners, with the per-pruner
accumulators of AV1_PRUNER_TIMING (feature extraction and inference, each
pruner in its own accumulator; build/libaom_perf_timing).

Protocol, as in §6.2 of that document: 3 frames, cpu-used=1, Tango cq32 and
BoxingPractice cq43. --threads=1 is mandatory: the accumulators are plain
static counters, and with two tile threads their call counts diverge (race
observed in the QA of 2026-10-04), so the share of the wall time is only
meaningful single-threaded. The remaining flags are the AOM-CTC All-Intra ones
of encode_ctc.encode.

Configurations (all at cpu-used=1, as deployed in the replacement campaigns):
    native        native CNN on, pre-search neutralized (tau 2/2/-1)
    post_none     CNN off, pre-search neutralized, post-NONE tau_stop=0.95
    effirst_ext   CNN off, pre-search efficiency-first, extended pruner on
The neutralized pre-search pruner still extracts and infers (its decisions never
fire), so its cost is reported apart and is NOT charged to the other pruners.

Reproduction (container av1_bench):
    /workspace/build/venv-ml/bin/python src/scripts/fase6/overhead_pruners_iscas.py
"""
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encode_ctc import TAU_BALANCED, parse_y4m  # noqa: E402

ENC = "/workspace/build/libaom_perf_timing/aomenc"
SEQ_DIR = "/workspace/src/samples/aomctc_test_set"
OUT = "/workspace/results/benchmark/overhead_iscas"
POINTS = [("Tango", 32), ("BoxingPractice", 43)]
NEUTRAL = {"AV1_STUDENT_TAU_NONE": "2", "AV1_STUDENT_TAU_SPLIT": "2",
           "AV1_STUDENT_TAU_REST": "-1"}
CONFIGS = {
    "native": dict(NEUTRAL),
    "post_none": dict(NEUTRAL, AV1_DISABLE_NATIVE_CNN="1",
                      AV1_STUDENT_H9C_ENABLE="1", AV1_STUDENT_H9C_TAU="0.95"),
    "effirst_ext": dict(TAU_BALANCED, AV1_DISABLE_NATIVE_CNN="1",
                        AV1_STUDENT_H9D_ENABLE="1"),
}
ACC = ["native_cnn", "h9a_infer", "h9a_feature_prep", "h9c_infer",
       "h9c_feature_prep", "h9d_infer", "h9d_feature_prep"]


def run(seq_path, cq, env_extra, bd):
    cmd = [ENC, "--cpu-used=1", "--passes=1", "--end-usage=q",
           "--cq-level=%d" % cq, "--kf-min-dist=0", "--kf-max-dist=0",
           "--deltaq-mode=0", "--enable-tpl-model=0",
           "--enable-keyframe-filtering=0", "--tile-columns=1",
           "--tile-rows=0", "--threads=1", "--row-mt=0",
           "--bit-depth=%d" % bd, "--limit=3", "--obu", "-o", "/dev/null",
           seq_path]
    env = dict(os.environ, AV1_PRUNER_TIMING="1", **env_extra)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode(errors="replace")[-500:])
    acc = {}
    for line in r.stderr.decode(errors="replace").splitlines():
        m = re.match(r"\s+(\S+).*calls=(\d+)\s+total_ms=\s*([\d.]+)", line)
        if m and m.group(1) in ACC:
            acc[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return wall, acc


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, cq in POINTS:
        f = [x for x in os.listdir(SEQ_DIR) if x.startswith(name + "_")][0]
        bd = parse_y4m(os.path.join(SEQ_DIR, f))[4]
        for cfg, env in CONFIGS.items():
            wall, acc = run(os.path.join(SEQ_DIR, f), cq, env, bd)
            row = {"seq": name, "cq": cq, "config": cfg,
                   "wall_s": round(wall, 3)}
            for k in ACC:
                calls, ms = acc.get(k, (0, 0.0))
                row[k + "_calls"] = calls
                row[k + "_ms"] = ms
            rows.append(row)
            pct = lambda *ks: 100.0 * sum(row[k + "_ms"] for k in ks) / (wall * 1000)
            print("%-15s cq%d %-12s wall=%7.1fs  cnn=%.3f%%  pre=%.3f%%  "
                  "post=%.3f%%  ext=%.3f%%" % (
                      name, cq, cfg, wall, pct("native_cnn"),
                      pct("h9a_infer", "h9a_feature_prep"),
                      pct("h9c_infer", "h9c_feature_prep"),
                      pct("h9d_infer", "h9d_feature_prep")), flush=True)
    with open(os.path.join(OUT, "overhead.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
