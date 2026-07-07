#!/usr/bin/env python3
"""Validate an av1_partition_data.bin against its source YUV.

Checks:
  1. Structural integrity: fixed 4116-byte records, contiguous sample_id,
     partition in [0,9], block_dim in {8,16,32,64}.
  2. Pixel accuracy (the ML-critical check): for blocks fully inside the frame,
     the recorded luma bytes must equal the source Y-plane pixels at
     (mi_row*4, mi_col*4). This proves each block carries the correct region,
     aligned to its RDO partition label.
  3. Exports PNGs of a few blocks spanning different sizes/classes for eyeballing.

Usage:
  python validate_partition_data.py --bin results/beauty_partition.bin \
      --source src/samples/Beauty_3840x2160_120fps_420_8bit_YUV_RAW.yuv \
      --width 3840 --height 2160 --out-dir results/beauty_blocks
"""

import argparse
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

LUMA_DIM = 64
LUMA_SIZE = LUMA_DIM * LUMA_DIM
SAMPLE = struct.Struct("<I4H5B3x{}s".format(LUMA_SIZE))
MI_SIZE = 4  # pixels per mode-info unit
PART_NAMES = ["NONE", "HORZ", "VERT", "SPLIT", "HORZ_A", "HORZ_B",
              "VERT_A", "VERT_B", "HORZ_4", "VERT_4"]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True)
    p.add_argument("--source", required=True, help="Raw .yuv (4:2:0 8-bit)")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--out-dir", default=None, help="Directory for sample PNGs")
    p.add_argument("--num-png", type=int, default=2,
                   help="PNGs to export per (block_dim, partition) combo")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    W, H = args.width, args.height

    if SAMPLE.size != 4116:
        raise SystemExit("Struct size {} != 4116".format(SAMPLE.size))
    fsz = os.path.getsize(args.bin)
    if fsz % SAMPLE.size:
        raise SystemExit("Not 4116-aligned: {} bytes".format(fsz))
    n = fsz // SAMPLE.size

    # Frame-0 luma plane from the raw YUV.
    with open(args.source, "rb") as f:
        yplane = np.frombuffer(f.read(W * H), dtype=np.uint8).reshape(H, W)

    parts, dims = Counter(), Counter()
    qs, ws, hs = set(), set(), set()
    prev_id = -1
    id_ok = True
    checked = mismatched = edge_skipped = 0
    png_budget = defaultdict(int)
    png_saved = 0
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    with open(args.bin, "rb") as f:
        for _ in range(n):
            (sid, fw, fh, mr, mc, q, bd, bs, bdim, part,
             luma) = SAMPLE.unpack(f.read(SAMPLE.size))
            if sid != prev_id + 1:
                id_ok = False
            prev_id = sid
            if not (0 <= part <= 9):
                raise SystemExit("bad partition {} @id {}".format(part, sid))
            if bdim not in (8, 16, 32, 64):
                raise SystemExit("bad block_dim {} @id {}".format(bdim, sid))
            parts[part] += 1
            dims[bdim] += 1
            qs.add(q); ws.add(fw); hs.add(fh)

            block = np.frombuffer(luma, dtype=np.uint8).reshape(
                LUMA_DIM, LUMA_DIM)[:bdim, :bdim]

            # Pixel-accuracy: only for blocks fully inside the frame.
            y0, x0 = mr * MI_SIZE, mc * MI_SIZE
            if y0 + bdim <= H and x0 + bdim <= W:
                ref = yplane[y0:y0 + bdim, x0:x0 + bdim]
                checked += 1
                if not np.array_equal(block, ref):
                    mismatched += 1
                    if mismatched <= 3:
                        print("  MISMATCH id={} dim={} @({},{})".format(
                            sid, bdim, y0, x0))
            else:
                edge_skipped += 1

            # Export a few diverse blocks as PNG.
            if args.out_dir and png_budget[(bdim, part)] < args.num_png:
                png_budget[(bdim, part)] += 1
                name = "blk_{:02d}px_{}_{}_id{}.png".format(
                    bdim, PART_NAMES[part], png_budget[(bdim, part)], sid)
                Image.fromarray(block, mode="L").save(
                    os.path.join(args.out_dir, name))
                png_saved += 1

    print("samples: {}  (file 4116-aligned, no leftover bytes)".format(n))
    print("sample_id contiguous 0..n-1:", id_ok)
    print("frame W/H recorded:", sorted(ws), sorted(hs),
          "(source {}x{})".format(W, H))
    print("qindex set:", sorted(qs))
    print("block_dim histogram:", dict(sorted(dims.items())))
    print("partition histogram:",
          {PART_NAMES[k]: v for k, v in sorted(parts.items())})
    print("pixel-accuracy: checked {} interior blocks, {} mismatched, "
          "{} edge blocks skipped".format(checked, mismatched, edge_skipped))
    if args.out_dir:
        print("PNGs exported:", png_saved, "->", args.out_dir)

    if mismatched:
        raise SystemExit("FAIL: {} interior blocks did not match source".format(
            mismatched))
    print("RESULT: PASS -- recorded luma matches source; labels aligned.")


if __name__ == "__main__":
    main(sys.argv[1:])
