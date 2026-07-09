#!/usr/bin/env python3
"""Benchmark the ML-pruned encoder vs an anchor: BD-rate + encode-time speedup.

For each (encoder, sequence, cq) it does a clean timed All-Intra encode (no
--psnr, so timing is not polluted and no CONFIG_INTERNAL_STATS build is needed),
then decodes the stream and computes Y-PSNR against the source in numpy. Per
sequence it reports BD-rate (test vs anchor) and the median encode-time speedup.

The test encoder is the PARTITION_ML_STUDENT build; thresholds are passed via the
AV1_STUDENT_TAU_NONE / AV1_STUDENT_TAU_SPLIT environment variables, so a tau
sweep needs no rebuild. The headline number is the held-out sequence.
"""

import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from bd_rate import bd_rate  # noqa: E402


def probe(path):
    m = re.search(r"(\d+)x(\d+)", os.path.basename(path))
    if not m:
        raise SystemExit("cannot parse WxH from " + path)
    w, h = int(m.group(1)), int(m.group(2))
    total = os.path.getsize(path) // (w * h * 3 // 2)
    return w, h, total


def read_y(path, w, h, start, n):
    """Return (n, h, w) uint8 luma planes from an 8-bit 4:2:0 .yuv."""
    fb = w * h * 3 // 2
    out = np.empty((n, h, w), dtype=np.uint8)
    with open(path, "rb") as f:
        for i in range(n):
            f.seek((start + i) * fb)
            out[i] = np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)
    return out


def y_psnr(src, dec):
    mse = np.mean((src.astype(np.float64) - dec.astype(np.float64)) ** 2)
    if mse <= 0:
        return 99.0
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def encode(enc, src, w, h, cq, skip, n, cpu, out_ivf, env_extra):
    # --limit counts input frames BEFORE --skip, so encode [skip, skip+n).
    cmd = [enc, "--usage=2", "--passes=1", "--threads=1",
           "--cpu-used={}".format(cpu), "--end-usage=q",
           "--cq-level={}".format(cq), "-w", str(w), "-h", str(h),
           "--skip={}".format(skip), "--limit={}".format(skip + n),
           "-o", out_ivf, src]
    env = dict(os.environ, **env_extra)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-400:] + "\n")
        raise SystemExit("encode failed: {} cq{}".format(enc, cq))
    return dt


def decode(dec_bin, ivf, out_yuv):
    r = subprocess.run([dec_bin, ivf, "-o", out_yuv, "--rawvideo"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-400:] + "\n")
        raise SystemExit("decode failed: " + ivf)


def run_encoder(tag, enc, dec_bin, seqs, args, env_extra, work, writer):
    """Encode every (seq, cq); return {seq: {"rate":[], "psnr":[], "time":[]}}."""
    res = {}
    for seq in seqs:
        w, h, total = probe(seq)
        base = os.path.splitext(os.path.basename(seq))[0]
        n = min(args.frames, total - args.skip)
        src_y = read_y(seq, w, h, args.skip, n)
        res.setdefault(base, {"rate": [], "psnr": [], "time": []})
        for cq in args.cqs:
            ivf = os.path.join(work, "{}_{}_{}.ivf".format(tag, base, cq))
            times = [encode(enc, seq, w, h, cq, args.skip, n, args.cpu_used,
                            ivf, env_extra) for _ in range(args.repeats)]
            dt = statistics.median(times)
            yuv = os.path.join(work, "{}_{}_{}.yuv".format(tag, base, cq))
            decode(dec_bin, ivf, yuv)
            dec_y = read_y(yuv, w, h, 0, n)
            psnr = y_psnr(src_y, dec_y)
            bits = os.path.getsize(ivf) * 8
            kbps = bits / n * 30.0 / 1000.0  # nominal 30 fps
            res[base]["rate"].append(kbps)
            res[base]["psnr"].append(psnr)
            res[base]["time"].append(dt)
            writer.writerow({"encoder": tag, "sequence": base, "cq": cq,
                             "kbps": round(kbps, 2), "y_psnr": round(psnr, 4),
                             "encode_s": round(dt, 3), "frames": n})
            os.remove(yuv)
            print("  [{}] {} cq{:>2}: {:.4f} dB, {:.1f} kbps, {:.1f}s".format(
                tag, base[:20], cq, psnr, kbps, dt), flush=True)
    return res


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_baseline/aomenc")
    p.add_argument("--decoder", default="/workspace/build/libaom_ml_check/aomdec")
    p.add_argument("--samples-dir", default="/workspace/src/samples")
    p.add_argument("--seqs", nargs="+", default=None,
                   help="explicit .yuv paths; default globs samples-dir")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--repeats", type=int, default=2,
                   help="timing repeats; median is reported")
    p.add_argument("--tau-none", type=float, default=0.9)
    p.add_argument("--tau-split", type=float, default=0.9)
    p.add_argument("--tau-rest", type=float, default=-1.0,
                   help="P(REST) threshold for the rect-off action; -1 = off")
    p.add_argument("--env", action="append", default=[],
                   help="extra KEY=VAL for the test encoder (repeatable), e.g. "
                        "per-level overrides AV1_STUDENT_TAU_NONE_16=0.9")
    p.add_argument("--out-dir", default="/workspace/results/benchmark")
    args = p.parse_args(argv)

    seqs = args.seqs or sorted(glob.glob(os.path.join(args.samples_dir, "*.yuv")))
    if not seqs:
        raise SystemExit("no sequences")
    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)

    runs_csv = os.path.join(args.out_dir, "runs.csv")
    rf = open(runs_csv, "w", newline="")
    rw = csv.DictWriter(rf, fieldnames=["encoder", "sequence", "cq", "kbps",
                                        "y_psnr", "encode_s", "frames"])
    rw.writeheader()

    test_env = {"AV1_STUDENT_TAU_NONE": str(args.tau_none),
                "AV1_STUDENT_TAU_SPLIT": str(args.tau_split),
                "AV1_STUDENT_TAU_REST": str(args.tau_rest)}
    for kv in args.env:
        k, _, v = kv.partition("=")
        test_env[k] = v

    print("ANCHOR:", args.anchor_enc)
    anchor = run_encoder("anchor", args.anchor_enc, args.decoder, seqs, args,
                         {}, work, rw)
    print("TEST", test_env, ":")
    test = run_encoder("test", args.test_enc, args.decoder, seqs, args,
                       test_env, work, rw)
    rf.close()

    taus_label = ";".join("{}={}".format(k, v) for k, v in sorted(
        test_env.items()))
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as sf:
        sw = csv.writer(sf)
        sw.writerow(["sequence", "bd_rate_pct", "time_speedup_x", "taus"])
        print("\n=== SUMMARY ({}) ===".format(taus_label))
        print("{:<28} {:>10} {:>12}".format("sequence", "BD-rate%", "speedup x"))
        for base in anchor:
            bd = bd_rate(anchor[base]["rate"], anchor[base]["psnr"],
                         test[base]["rate"], test[base]["psnr"])
            speed = (sum(anchor[base]["time"]) / max(sum(test[base]["time"]),
                                                     1e-9))
            sw.writerow([base, round(bd, 3), round(speed, 3), taus_label])
            print("{:<28} {:>10.3f} {:>12.3f}".format(base[:27], bd, speed))
    print("\nruns ->", runs_csv, "\nsummary ->", summary_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
