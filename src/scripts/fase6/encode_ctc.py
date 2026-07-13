#!/usr/bin/env python3
"""Fase 6 — CTC final-results encoder (All Intra, Class A1, 4K, 10-bit).

Encodes each CTC A1 sequence under the AOM-CTC All-Intra configuration (good
mode, usage=0) at the four thesis QPs, for six configurations:

    anchor        libaom original                       cpu-used=0
    ml_balanced   H9a pruner, balanced point (P_rect)   cpu-used=0
    ml_aggr       H9a pruner, aggressive point (A3)      cpu-used=0
    native_cpu1   libaom original                       cpu-used=1
    native_cpu2   libaom original                       cpu-used=2
    native_cpu3   libaom original                       cpu-used=3

The ML points share the deployed H9a student (build/libaom_perf) and differ
only in the policy thresholds (AV1_STUDENT_TAU_*). The anchor and native points
run the untouched encoder (build/libaom_perf_anchor); the native points are the
encoder's own speed knob, the practitioner's "why not just use cpu-used=1?"
baseline that the ML proposal is positioned against.

One raw row per encode is appended to raw_results.csv:
    seq, config, cq, fps_num, fps_den, frames, bytes, psnr_y, time_s
The run is resumable (rows already present are skipped) so it survives a killed
container. report_ctc.py turns the raw rows into the thesis tables.

Encode command mirrors AOM-CTC §4.1 (All Intra) with the §4 4K tiling rule for
class A1: --tile-columns=1 (log2 => 2 columns) --threads=2 --row-mt=0. QP is
specified on libaom's --cq-level scale (this build has no --qp); the thesis QP
grid 20/32/43/55 is kept from the ML validation (Fase 5).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Operating points -------------------------------------------------------
# ML policy thresholds are the two frozen Fase 6 points (see ANDAMENTO §4.2).
TAU_BALANCED = {  # P_rect: conservative BD, tau_none 0.95 + rect-off
    "AV1_STUDENT_TAU_NONE": "0.95",
    "AV1_STUDENT_TAU_SPLIT": "0.90",
    "AV1_STUDENT_TAU_REST": "0.20",
}
TAU_AGGRESSIVE = {  # A3: max deployable TS
    "AV1_STUDENT_TAU_NONE": "0.60",
    "AV1_STUDENT_TAU_SPLIT": "0.85",
    "AV1_STUDENT_TAU_REST": "0.40",
}


def configs(anchor_enc, ml_enc):
    """Return the six (name, encoder, env, cpu_used) benchmark configs."""
    return [
        ("anchor", anchor_enc, {}, 0),
        ("ml_balanced", ml_enc, dict(TAU_BALANCED), 0),
        ("ml_aggr", ml_enc, dict(TAU_AGGRESSIVE), 0),
        ("native_cpu1", anchor_enc, {}, 1),
        ("native_cpu2", anchor_enc, {}, 2),
        ("native_cpu3", anchor_enc, {}, 3),
    ]


# --- y4m header -------------------------------------------------------------
def parse_y4m(path):
    """Read (w, h, fps_num, fps_den, bit_depth) from the y4m header line."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", errors="replace")
    if not header.startswith("YUV4MPEG2"):
        raise SystemExit("not a y4m file: " + path)
    w = h = None
    fps_num, fps_den = 30, 1
    bit_depth = 8
    for tok in header.split():
        if tok[0] == "W":
            w = int(tok[1:])
        elif tok[0] == "H":
            h = int(tok[1:])
        elif tok[0] == "F":
            num, _, den = tok[1:].partition(":")
            fps_num, fps_den = int(num), int(den or 1)
        elif tok[0] == "C":
            m = re.search(r"p(\d+)", tok)  # C420p10 -> 10, C420 -> 8
            bit_depth = int(m.group(1)) if m else 8
    if w is None or h is None:
        raise SystemExit("cannot parse WxH from y4m header: " + path)
    return w, h, fps_num, fps_den, bit_depth


# --- encode -----------------------------------------------------------------
def encode(enc, seq, cq, frames, bit_depth, cpu_used, env_extra, out_obu):
    """Run one AOM-CTC All-Intra encode; return (wall_time_s, psnr_y)."""
    cmd = [
        enc,
        "--cpu-used={}".format(cpu_used),
        "--passes=1", "--end-usage=q",
        "--cq-level={}".format(cq),
        "--kf-min-dist=0", "--kf-max-dist=0",
        "--deltaq-mode=0", "--enable-tpl-model=0",
        "--enable-keyframe-filtering=0",
        "--tile-columns=1", "--tile-rows=0", "--threads=2", "--row-mt=0",
        "--bit-depth={}".format(bit_depth),
        "--limit={}".format(frames),
        "--psnr", "--obu",
        "-o", out_obu, seq,
    ]
    env = dict(os.environ, **env_extra)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-600:] + "\n")
        raise SystemExit("encode failed: {} cq{} cpu{}".format(enc, cq, cpu_used))
    text = r.stderr.decode(errors="replace")
    m = re.search(r"Overall/Avg/Y/U/V\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text)
    if not m:
        sys.stderr.write(text[-600:] + "\n")
        raise SystemExit("could not parse PSNR-Y for {} cq{}".format(seq, cq))
    return dt, float(m.group(1))


# --- resumable CSV ----------------------------------------------------------
CSV_FIELDS = ["seq", "config", "cq", "fps_num", "fps_den", "frames",
              "bytes", "psnr_y", "time_s"]


def load_done(csv_path):
    """Return the set of (seq, config, cq) already recorded."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["seq"], row["config"], int(row["cq"])))
    return done


def append_row(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seqs", nargs="+", default=None,
                   help="only sequences whose name contains one of these "
                        "substrings (default: all .y4m in --seq-dir)")
    p.add_argument("--work", default=None,
                   help="scratch dir for .obu (default <out-dir>/_work)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = args.work or os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    if args.seqs:
        seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no .y4m sequences in " + args.seq_dir)
    cfgs = configs(args.anchor_enc, args.ml_enc)

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("Fase 6 CTC benchmark: {} seqs x {} configs x {} cqs = {} encodes"
          .format(len(seqs), len(cfgs), len(args.cqs), total), flush=True)
    print("already done: {}/{}".format(len(done), total), flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, enc, env, cpu in cfgs:
                if (name, cname, cq) in done:
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cname, cq))
                dt, psnr_y = encode(enc, seq, cq, args.frames, bd, cpu, env,
                                    out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                append_row(csv_path, {
                    "seq": name, "config": cname, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, cname, cq))
                print("  cq{:>2} {:<12} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nFASE6_ENCODE_DONE  ({} rows in {})".format(len(done), csv_path),
          flush=True)


if __name__ == "__main__":
    main()
