#!/usr/bin/env python3
"""H7+H8 combined benchmark: one anchor pass, many operating points.

run_benchmark.py re-encodes the anchor for every tau point; at cpu-used=0 4K
that triples the cost. This driver encodes the anchor ONCE, then evaluates each
operating point (student policies for H7, surrogate replay for H8) against that
single baseline, reporting BD-rate and encode-time speedup per point.

Each point is a dict: {name, enc, env, replay}. `replay` (optional) is a
per-cq surrogate-probability file produced by surrogate_replay.py; when set the
env gets AV1_STUDENT_PROBS_FILE and the encoder reads the surrogate's decisions
instead of running the distilled student (the H8 ceiling measurement).
"""

import argparse
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from bd_rate import bd_rate  # noqa: E402
from run_benchmark import probe, read_y, y_psnr, encode, decode  # noqa: E402


def eval_point(name, enc, dec_bin, seq, args, env, work, anchor):
    """Encode/decode seq at every cq; return (bd_rate, speedup, rows)."""
    w, h, total = probe(seq)
    n = min(args.frames, total - args.skip)
    src_y = read_y(seq, w, h, args.skip, n)
    rate, psnr, times, rows = [], [], [], []
    for cq in args.cqs:
        e = dict(env)
        if "replay_tmpl" in env:
            e = {k: v for k, v in env.items() if k != "replay_tmpl"}
            e["AV1_STUDENT_PROBS_FILE"] = env["replay_tmpl"].format(cq=cq)
        ivf = os.path.join(work, "{}_{}_{}.ivf".format(name, os.path.basename(
            seq).split("_")[0], cq))
        dts = [encode(enc, seq, w, h, cq, args.skip, n, args.cpu_used, ivf, e)
               for _ in range(args.repeats)]
        dt = statistics.median(dts)
        yuv = ivf[:-4] + ".yuv"
        decode(dec_bin, ivf, yuv)
        ps = y_psnr(src_y, read_y(yuv, w, h, 0, n))
        kbps = os.path.getsize(ivf) * 8 / n * 30.0 / 1000.0
        os.remove(yuv)
        rate.append(kbps)
        psnr.append(ps)
        times.append(dt)
        rows.append((name, cq, round(kbps, 2), round(ps, 4), round(dt, 3)))
        print("  [{}] cq{:>2}: {:.4f} dB, {:.1f} kbps, {:.1f}s".format(
            name, cq, ps, kbps, dt), flush=True)
    if anchor is None:
        return None, None, {"rate": rate, "psnr": psnr, "time": times}, rows
    bd = bd_rate(anchor["rate"], anchor["psnr"], rate, psnr)
    speed = sum(anchor["time"]) / max(sum(times), 1e-9)
    return bd, speed, {"rate": rate, "psnr": psnr, "time": times}, rows


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--decoder", default="/workspace/build/libaom_ml_check/aomdec")
    p.add_argument("--seq", required=True, help="single .yuv (held-out Jockey)")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--replay-dir", default="/workspace/results/benchmark/h8_probs",
                   help="dir holding surrogate replay files jockey_cq{cq}.bin")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/h7h8")
    args = p.parse_args(argv)

    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    # Operating points. Student policies (H7) run on the distilled per-size MLP;
    # the surrogate point (H8) replays precomputed probs through the same hook.
    rep = os.path.join(args.replay_dir, "jockey_cq{cq}.bin")
    points = [
        ("P0_oldpolicy", args.test_enc,
         {"AV1_STUDENT_TAU_NONE": "0.84", "AV1_STUDENT_TAU_SPLIT": "0.90",
          "AV1_STUDENT_TAU_REST": "-1"}),
        ("P_rect", args.test_enc,
         {"AV1_STUDENT_TAU_NONE": "0.95", "AV1_STUDENT_TAU_SPLIT": "0.90",
          "AV1_STUDENT_TAU_REST": "0.20"}),
        ("P_ref", args.test_enc,
         {"AV1_STUDENT_TAU_NONE_16": "0.95", "AV1_STUDENT_TAU_NONE_32": "0.95",
          "AV1_STUDENT_TAU_NONE_64": "0.80", "AV1_STUDENT_TAU_SPLIT": "0.90",
          "AV1_STUDENT_TAU_REST_16": "0.10", "AV1_STUDENT_TAU_REST_32": "0.20",
          "AV1_STUDENT_TAU_REST_64": "-1"}),
        ("H8_surrogate", args.test_enc,
         {"AV1_STUDENT_TAU_NONE": "0.90", "AV1_STUDENT_TAU_SPLIT": "0.90",
          "AV1_STUDENT_TAU_REST": "0.20", "replay_tmpl": rep}),
    ]
    # Drop H8 if replay files are absent (H7-only run).
    if not all(os.path.exists(rep.format(cq=cq)) for cq in args.cqs):
        print("[warn] replay files missing in {}; skipping H8_surrogate".format(
            args.replay_dir), flush=True)
        points = [pt for pt in points if pt[0] != "H8_surrogate"]

    print("=== ANCHOR (once) ===", flush=True)
    _, _, anchor, all_rows = eval_point(
        "anchor", args.anchor_enc, args.decoder, args.seq, args,
        {}, work, None)

    summary = []
    for name, enc, env in points:
        print("=== {} ===".format(name), flush=True)
        bd, speed, _, rows = eval_point(name, enc, args.decoder, args.seq,
                                        args, env, work, anchor)
        all_rows += rows
        summary.append((name, bd, speed))

    import csv
    with open(os.path.join(args.out_dir, "runs.csv"), "w", newline="") as f:
        csv.writer(f).writerows(
            [("encoder", "cq", "kbps", "y_psnr", "encode_s")] + all_rows)
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["point", "bd_rate_pct", "time_speedup_x"])
        wtr.writerows([(n, round(b, 3), round(s, 3)) for n, b, s in summary])

    print("\n=== H7+H8 SUMMARY (Jockey held-out, {} frames) ===".format(
        args.frames))
    print("{:<16} {:>10} {:>12}".format("point", "BD-rate%", "speedup x"))
    for n, b, s in summary:
        print("{:<16} {:>10.3f} {:>12.3f}".format(n, b, s))
    print("H7H8_DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
