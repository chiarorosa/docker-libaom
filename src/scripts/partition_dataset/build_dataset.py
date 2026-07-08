#!/usr/bin/env python3
"""Automate partition-dataset generation over N sequences x N QPs.

For each raw .yuv sequence and each QP, this driver:
  1. Picks K frames spread evenly across the whole clip (temporal stratified
     sampling) -- NOT consecutive frames, which in All-Intra are near-duplicates
     and would make the dataset redundant.
  2. Encodes each picked frame independently as an intra keyframe with the
     LOG_PARTITION_DATA build (--usage=2, --threads=1, fixed --cpu-used), all
     appended into one per-(seq,QP) binary.
  3. Converts that binary into one per-(seq,QP) .pkl.
  4. Records a manifest row (dims, frames used, QP, base_qindex, cpu-used,
     sample and class counts, file paths) -- the reproducibility ledger.

Ground-truth note: use --cpu-used 0 so the recorded partitioning reflects the
near-exhaustive RD search, minimally polluted by speed heuristics/ML pruning.
Datasets built with different --cpu-used are NOT comparable; it is logged per row.

Sequence dimensions are parsed from the filename (e.g. Name_3840x2160_...yuv);
frame count is derived from file size assuming 8-bit 4:2:0.

Example:
  python build_dataset.py --samples-dir /workspace/src/samples \
      --out-dir /workspace/results/dataset --qps 20 32 43 55 \
      --frames 5 --cpu-used 0
"""

import argparse
import csv
import glob
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CONVERTER = os.path.join(HERE, "convert_partition_data.py")

LUMA_SIZE = 64 * 64
SAMPLE = struct.Struct("<I4H5B3x{}s".format(LUMA_SIZE))
PART_NAMES = ["NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B",
              "VERT_A", "VERT_B", "HORZ_4", "VERT_4"]
DIMS = [8, 16, 32, 64]

MANIFEST_FIELDS = (
    ["sequence", "width", "height", "total_frames", "num_frames_used",
     "frames_used", "cq_level", "base_qindex", "cpu_used", "threads",
     "num_samples"]
    + ["dim{}".format(d) for d in DIMS]
    + ["part_{}".format(p) for p in PART_NAMES]
    + ["bin_path", "pkl_path", "timestamp"]
)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-dir", default="/workspace/src/samples")
    p.add_argument("--out-dir", default="/workspace/results/dataset")
    p.add_argument("--qps", type=int, nargs="+", default=[20, 32, 43, 55],
                   help="cq-level values (0..63) to sweep")
    p.add_argument("--frames", type=int, default=5,
                   help="K frames per sequence, spread across the whole clip")
    p.add_argument("--cpu-used", type=int, default=0,
                   help="aomenc speed; 0 = most exhaustive RD (best ground truth)")
    p.add_argument("--threads", type=int, default=1,
                   help="keep 1 for deterministic logging")
    p.add_argument("--keep-bin", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="keep the per-(seq,QP) .bin after converting to .pkl; "
                        "--no-keep-bin deletes it to save disk (the .pkl is the "
                        "product, the .bin is redundant afterwards)")
    p.add_argument("--aomenc", default="/workspace/build/libaom_logpart/aomenc")
    p.add_argument("--python", default=sys.executable,
                   help="Python (with numpy) used for the converter step")
    p.add_argument("--seqs", nargs="+", default=None,
                   help="Explicit .yuv paths; default globs samples-dir/*.yuv")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (seqs, dims, frame offsets) and exit")
    return p.parse_args(argv)


def probe_sequence(path):
    """Return (width, height, total_frames) from filename dims + file size."""
    m = re.search(r"(\d+)x(\d+)", os.path.basename(path))
    if not m:
        raise SystemExit(
            "Cannot parse WxH from filename: {}. Rename to include e.g. "
            "_3840x2160_.".format(os.path.basename(path)))
    w, h = int(m.group(1)), int(m.group(2))
    frame_bytes = w * h * 3 // 2  # 8-bit 4:2:0
    total = os.path.getsize(path) // frame_bytes
    return w, h, total


def spread_offsets(total, k):
    """K frame indices spread evenly across [0, total-1] (deduped)."""
    if total <= 0:
        return []
    if k <= 1 or total == 1:
        return [0]
    k = min(k, total)
    offs = sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})
    return offs


def count_bin(path):
    """Return (num_samples, base_qindex, dim_counts, part_counts)."""
    dim_c, part_c = Counter(), Counter()
    base_q = None
    with open(path, "rb") as f:
        while True:
            buf = f.read(SAMPLE.size)
            if len(buf) < SAMPLE.size:
                break
            rec = SAMPLE.unpack(buf)
            q, bdim, part = rec[5], rec[8], rec[9]
            if base_q is None:
                base_q = q
            dim_c[bdim] += 1
            part_c[part] += 1
    n = sum(dim_c.values())
    return n, base_q, dim_c, part_c


def counts_from_pkl(pkl_path):
    """Recover (num_samples, base_qindex, dim_counts, part_counts) from a .pkl,
    used to rebuild a manifest row when resuming over an already-converted
    (seq,QP) whose .bin was deleted."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    part = list(d["partition"].tolist())
    bdim = d["block_dim"].tolist()
    q = d["qindex"].tolist()
    n = len(part)
    base_q = int(q[0]) if n else None
    return n, base_q, Counter(bdim), Counter(part)


def make_row(base, w, h, total, offs, qp, base_q, n, dim_c, part_c, bin_path,
             pkl_path, cpu_used, threads):
    row = {
        "sequence": base, "width": w, "height": h, "total_frames": total,
        "num_frames_used": len(offs), "frames_used": ";".join(map(str, offs)),
        "cq_level": qp, "base_qindex": base_q, "cpu_used": cpu_used,
        "threads": threads, "num_samples": n, "bin_path": bin_path,
        "pkl_path": pkl_path,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for d in DIMS:
        row["dim{}".format(d)] = dim_c.get(d, 0)
    for i, name in enumerate(PART_NAMES):
        row["part_{}".format(name)] = part_c.get(i, 0)
    return row


def extract_frame(seq, frame_bytes, offset, dst):
    """Copy one raw frame at `offset` to `dst` via a direct seek (dd skip),
    avoiding a sequential read of the whole file for large offsets."""
    r = subprocess.run(
        ["dd", "if=" + seq, "of=" + dst, "bs={}".format(frame_bytes),
         "skip={}".format(offset), "count=1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0 or os.path.getsize(dst) != frame_bytes:
        sys.stderr.write(r.stderr.decode(errors="replace")[-500:] + "\n")
        raise SystemExit("dd extract failed: {} frame {}".format(
            os.path.basename(seq), offset))


def encode_frame(args, frame_yuv, w, h, qp, bin_path, tmp_ivf):
    """Encode a single pre-extracted frame (1-frame .yuv) as an intra keyframe,
    appending its partition samples to bin_path."""
    env = dict(os.environ, AV1_PARTITION_LOG=bin_path)
    cmd = [
        args.aomenc, "--usage=2", "--passes=1",
        "--threads={}".format(args.threads), "--cpu-used={}".format(args.cpu_used),
        "--end-usage=q", "--cq-level={}".format(qp),
        "-w", str(w), "-h", str(h), "--limit=1",
        "-o", tmp_ivf, frame_yuv,
    ]
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-500:] + "\n")
        raise SystemExit("aomenc failed on {} qp={}".format(
            os.path.basename(frame_yuv), qp))


def main(argv):
    args = parse_args(argv)
    seqs = args.seqs or sorted(glob.glob(os.path.join(args.samples_dir, "*.yuv")))
    if not seqs:
        raise SystemExit("No .yuv found in {}".format(args.samples_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    plan = []
    for seq in seqs:
        w, h, total = probe_sequence(seq)
        offs = spread_offsets(total, args.frames)
        plan.append((seq, w, h, total, offs))

    print("Plan: {} sequence(s) x {} QP(s) {} = {} encodes @ cpu-used={}".format(
        len(seqs), len(args.qps), args.qps,
        sum(len(o) for *_, o in plan) * len(args.qps), args.cpu_used))
    for seq, w, h, total, offs in plan:
        print("  {:<40} {}x{} frames={} -> use {} @ {}".format(
            os.path.basename(seq), w, h, total, len(offs), offs))
    if args.dry_run:
        print("(dry-run: nothing encoded)")
        return

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    new = not os.path.exists(manifest_path)
    # Resume support: stems (Name_cqNN) already recorded in the manifest.
    done_stems = set()
    if not new:
        with open(manifest_path, newline="") as rf:
            for r in csv.DictReader(rf):
                done_stems.add(
                    os.path.splitext(os.path.basename(r.get("pkl_path", "")))[0])
    mf = open(manifest_path, "a", newline="")
    writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
    if new:
        writer.writeheader()
    tmp_ivf = os.path.join(args.out_dir, "_tmp.ivf")

    for seq, w, h, total, offs in plan:
        base = os.path.splitext(os.path.basename(seq))[0]
        frame_bytes = w * h * 3 // 2  # 8-bit 4:2:0

        # Extract the target frames ONCE per sequence (direct seek, ~12 MB each)
        # and reuse them across every QP -- avoids re-reading the whole file per
        # (QP, offset) via --skip.
        frame_dir = tempfile.mkdtemp(prefix="av1frames_")
        try:
            frame_files = []
            for off in offs:
                fp = os.path.join(frame_dir, "f{}.yuv".format(off))
                extract_frame(seq, frame_bytes, off, fp)
                frame_files.append(fp)
            print("[{}] extracted {} frame(s) {} -> encoding {} QP(s)".format(
                base, len(offs), offs, len(args.qps)), flush=True)

            for qp in args.qps:
                stem = "{}_cq{:02d}".format(base, qp)
                bin_path = os.path.join(args.out_dir, stem + ".bin")
                pkl_path = os.path.join(args.out_dir, stem + ".pkl")

                # Resume: skip a (seq,QP) whose .pkl already exists. Backfill its
                # manifest row from the .pkl if it is missing (e.g. salvaged run).
                if os.path.exists(pkl_path):
                    if stem not in done_stems:
                        n, base_q, dim_c, part_c = counts_from_pkl(pkl_path)
                        writer.writerow(make_row(
                            base, w, h, total, offs, qp, base_q, n, dim_c,
                            part_c, bin_path, pkl_path, args.cpu_used,
                            args.threads))
                        mf.flush()
                        done_stems.add(stem)
                    print("  cq={} already done (pkl exists), skipping".format(qp),
                          flush=True)
                    continue

                if os.path.exists(bin_path):
                    os.remove(bin_path)  # fresh per (seq,QP)
                print("  cq={} encoding {} frame(s)...".format(qp, len(offs)),
                      flush=True)
                for fp in frame_files:
                    encode_frame(args, fp, w, h, qp, bin_path, tmp_ivf)

                n, base_q, dim_c, part_c = count_bin(bin_path)
                subprocess.run([args.python, CONVERTER, "--input", bin_path,
                                "--output", pkl_path, "--seq", base], check=True)
                if not args.keep_bin:
                    os.remove(bin_path)  # .pkl is the product; free the disk
                    print("  (removed {} to save space)".format(
                        os.path.basename(bin_path)))

                writer.writerow(make_row(
                    base, w, h, total, offs, qp, base_q, n, dim_c, part_c,
                    bin_path, pkl_path, args.cpu_used, args.threads))
                mf.flush()
                done_stems.add(stem)
                print("  -> {} samples, base_qindex={}, pkl={}".format(
                    n, base_q, pkl_path))
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    mf.close()
    if os.path.exists(tmp_ivf):
        os.remove(tmp_ivf)
    print("Done. Manifest:", manifest_path)


if __name__ == "__main__":
    main(sys.argv[1:])
