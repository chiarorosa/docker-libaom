#!/usr/bin/env python3
"""H8: score the ConvNeXt surrogate over source frames and dump per-node probs.

The surrogate's input is source luma + qindex only, so its in-loop decisions
can be precomputed EXACTLY outside the encoder. The PARTITION_ML_STUDENT hook
reads this file back (AV1_STUDENT_PROBS_FILE) and applies the very same pruning
policy, which turns the encoder into a measurement device for the surrogate's
BD-rate ceiling -- no convolutional inference in C, no distillation loss.

Record layout (little-endian), one per whole 64x64 unit per encoded frame:
  int32 frame (encode order, 0-based), int32 sb_row, int32 sb_col,
  float32 probs[21][3]  -- nodes ordered [64] ++ [32 row-major] ++ [16 rm],
                           each [P(NONE), P(SPLIT), P(REST)].

The qindex fed to the surrogate mirrors aomenc --end-usage=q: qindex = 4*cq.
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from model import PartitionSurrogate  # noqa: E402
from partition_defs import MODEL_LEVELS  # noqa: E402
import student as studentmod  # noqa: E402


def read_y(path, w, h, idx):
    fb = w * h * 3 // 2
    with open(path, "rb") as f:
        f.seek(idx * fb)
        return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surrogate", default="/workspace/results/models/"
                                          "surrogate_fs/surrogate_best.pt")
    p.add_argument("--yuv", required=True)
    p.add_argument("--cq", type=int, required=True)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args(argv)

    m = re.search(r"(\d+)x(\d+)", os.path.basename(args.yuv))
    if not m:
        raise SystemExit("cannot parse WxH from " + args.yuv)
    w, h = int(m.group(1)), int(m.group(2))
    sb_rows, sb_cols = h // 64, w // 64  # whole units only (matches the gate)
    qindex = 4 * args.cq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.surrogate, map_location=device)
    sa = ckpt.get("args", {})
    net = PartitionSurrogate(sa.get("variant", "tiny"),
                             sa.get("fusion_dim", 128)).to(device).eval()
    net.load_state_dict(ckpt["model"])

    qplane = np.full((64, 64), qindex / 255.0, dtype=np.float32)
    n_rec = 0
    with open(args.out, "wb") as out:
        for fi in range(args.frames):
            y = read_y(args.yuv, w, h, args.skip + fi)
            keys, batch = [], []

            def flush():
                nonlocal n_rec
                if not batch:
                    return
                x = torch.from_numpy(np.stack(batch)).to(device)
                with torch.no_grad():
                    logits = net(x)
                    probs = {d: F.softmax(logits[d].float(), dim=-1)
                             .cpu().numpy() for d, _ in MODEL_LEVELS}
                for k, (sr, sc) in enumerate(keys):
                    rec = [struct.pack("<3i", fi, sr, sc)]
                    for dim, side in MODEL_LEVELS:
                        for r in range(side):
                            for c in range(side):
                                p3 = studentmod.collapse_probs(
                                    probs[dim][k, r, c])
                                rec.append(struct.pack(
                                    "<3f", *(float(v) for v in p3)))
                    out.write(b"".join(rec))
                    n_rec += 1
                keys.clear()
                batch.clear()

            for sr in range(sb_rows):
                for sc in range(sb_cols):
                    luma = y[sr * 64:(sr + 1) * 64, sc * 64:(sc + 1) * 64]
                    batch.append(np.stack(
                        [luma.astype(np.float32) / 255.0, qplane]))
                    keys.append((sr, sc))
                    if len(batch) >= args.batch:
                        flush()
            flush()
            print("frame {}: {} records total".format(fi, n_rec), flush=True)
    print("wrote {} ({} records)".format(args.out, n_rec))


if __name__ == "__main__":
    main(sys.argv[1:])
