#!/usr/bin/env python3
"""C <-> Python parity check for the student feature extractor (Fase D gate).

Encodes one frame with the PARTITION_ML_STUDENT build while
AV1_STUDENT_FEATURE_DUMP makes the C side append one record per scored node
(int32 n, mi_row, mi_col, qindex + float32 feats[NUM_FEATURES]). This script
then recomputes every vector with features.node_features() from the same
source pixels and reports the worst absolute deviation per feature.

Expected result: bit-identical for the integer-derived features; <= 1-2 ulp
(float32) on log1p outputs, which is harmless to the MLP. Any larger deviation
means the two implementations diverged and MUST be fixed before H7.

Run inside the container, e.g.:
  venv-ml/bin/python check_feature_parity.py \
      --aomenc /workspace/build/libaom_ml_check/aomenc
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features as featmod  # noqa: E402

REC = struct.Struct("<4i{}f".format(featmod.NUM_FEATURES))


def make_yuv(path, w, h, seed=0):
    """Textured synthetic 4:2:0 frame: smooth gradients + noise + hard edges,
    so every feature (variance, gradients, band structure) is exercised."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    y = (96 + 64 * np.sin(xx / 17.0) * np.cos(yy / 23.0) +
         rng.normal(0, 24, size=(h, w)))
    y[:, w // 3:w // 3 + 8] = 235  # vertical bar (VERT cue)
    y[h // 2:h // 2 + 6, :] = 16   # horizontal bar (HORZ cue)
    y = np.clip(y, 0, 255).astype(np.uint8)
    uv = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(y.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())
    return y


def read_y(path, w, h):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aomenc", default="/workspace/build/libaom_ml_check/aomenc")
    p.add_argument("--yuv", default="/tmp/parity_448x256.yuv",
                   help="source; synthesized if missing (WxH parsed from name)")
    p.add_argument("--dump", default="/tmp/parity_feats.bin")
    p.add_argument("--cq", type=int, default=32)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-6,
                   help="max |C - Python| accepted (float32 ulp headroom)")
    args = p.parse_args(argv)

    import re
    m = re.search(r"(\d+)x(\d+)", os.path.basename(args.yuv))
    if not m:
        raise SystemExit("cannot parse WxH from " + args.yuv)
    w, h = int(m.group(1)), int(m.group(2))
    if os.path.exists(args.yuv):
        y = read_y(args.yuv, w, h)
    else:
        y = make_yuv(args.yuv, w, h)

    if os.path.exists(args.dump):
        os.remove(args.dump)
    env = dict(os.environ, AV1_STUDENT_FEATURE_DUMP=args.dump)
    cmd = [args.aomenc, "--usage=2", "--passes=1", "--threads=1",
           "--cpu-used={}".format(args.cpu_used), "--end-usage=q",
           "--cq-level={}".format(args.cq), "-w", str(w), "-h", str(h),
           "--limit=1", "-o", "/tmp/parity.ivf", args.yuv]
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-500:] + "\n")
        raise SystemExit("encode failed")

    blob = open(args.dump, "rb").read()
    if len(blob) % REC.size:
        raise SystemExit("dump size {} not a multiple of record size {}".format(
            len(blob), REC.size))
    n_rec = len(blob) // REC.size
    if n_rec == 0:
        raise SystemExit("no records: is the build PARTITION_ML_STUDENT=1?")

    worst = np.zeros(featmod.NUM_FEATURES)
    seen = set()
    checked = 0
    for i in range(n_rec):
        vals = REC.unpack_from(blob, i * REC.size)
        n, mi_row, mi_col, qindex = vals[:4]
        cf = np.array(vals[4:], dtype=np.float32)
        key = (n, mi_row, mi_col)
        if key in seen:
            continue  # rd_pick_partition may revisit a node; one check suffices
        seen.add(key)
        sb_r, sb_c = (mi_row & ~15) * 4, (mi_col & ~15) * 4
        sb = y[sb_r:sb_r + 64, sb_c:sb_c + 64]
        r_cell = (mi_row & 15) // (n // 4)
        c_cell = (mi_col & 15) // (n // 4)
        pf = featmod.node_features(sb, n, r_cell, c_cell, qindex)
        worst = np.maximum(worst, np.abs(cf.astype(np.float64) -
                                         pf.astype(np.float64)))
        checked += 1

    print("checked {} unique nodes ({} records)".format(checked, n_rec))
    bad = 0
    for k in range(featmod.NUM_FEATURES):
        flag = ""
        if worst[k] > args.atol:
            flag = "  <-- MISMATCH"
            bad += 1
        print("  [{:>2}] {:<18} max|dC-dPy| = {:.3e}{}".format(
            k, featmod.FEATURE_NAMES[k], worst[k], flag))
    if bad:
        raise SystemExit("PARITY FAILED: {} feature(s) above atol={}".format(
            bad, args.atol))
    print("PARITY OK (atol={})".format(args.atol))


if __name__ == "__main__":
    main(sys.argv[1:])
