#!/usr/bin/env python3
"""Convert av1_partition_data.bin (produced by a LOG_PARTITION_DATA libaom build)
into a serialized dataset (.pkl) for the ConvNeXt-AV1 partition surrogate.

The binary is a flat sequence of fixed-size C structs (PartitionSample, see
av1/encoder/partition_search.c). Each record holds the source luma of a square
block, the RDO partition decision (ground truth), the QP and the frame size.

C struct layout (little-endian, no implicit padding -> 4144 bytes), matching
PartitionSample in av1/encoder/partition_search.c:80-100:

    int64  none_dist        # [E] RD context of the PARTITION_NONE candidate
    int64  none_rdcost
    uint32 sample_id
    uint32 none_rate
    uint16 frame_width
    uint16 frame_height
    uint16 mi_row
    uint16 mi_col
    uint16 dc_q
    uint8  qindex
    uint8  bit_depth
    uint8  bsize
    uint8  block_dim        # valid side in pixels: 8/16/32/64
    uint8  partition        # PARTITION_TYPE, 0..9
    uint8  above_bsize
    uint8  left_bsize
    uint8  neigh_avail
    uint8  pad[6]
    uint8  luma[64*64]      # top-left block_dim x block_dim region is valid

Usage:
    python convert_partition_data.py --input av1_partition_data.bin \
        --output dataset.pkl [--block-dim 32] [--qindex 128] [--seq foreman]
"""

import argparse
import pickle
import struct
import sys
from collections import Counter

import numpy as np

# Must match PartitionSample in av1/encoder/partition_search.c exactly.
# Layout (little-endian): [E] none_dist(q) none_rdcost(q); sample_id(I)
# none_rate(I); frame_w,frame_h,mi_row,mi_col,dc_q (5H); qindex,bit_depth,bsize,
# block_dim,partition,above_bsize,left_bsize,neigh_avail (8B); 6 pad; luma(4096s).
LUMA_DIM = 64
LUMA_SIZE = LUMA_DIM * LUMA_DIM
SAMPLE_STRUCT = struct.Struct("<qqII5H8B6x{}s".format(LUMA_SIZE))
SAMPLE_SIZE = SAMPLE_STRUCT.size  # expected: 4144
EXPECTED_SIZE = 4144

# PARTITION_TYPE names (av1/common/enums.h), index == label value.
PARTITION_NAMES = [
    "NONE", "HORZ", "VERT", "SPLIT",
    "HORZ_A", "HORZ_B", "VERT_A", "VERT_B",
    "HORZ_4", "VERT_4",
]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True,
                   help="Path to av1_partition_data.bin")
    p.add_argument("--output", "-o", required=True,
                   help="Path to the output .pkl dataset")
    p.add_argument("--block-dim", type=int, default=None,
                   choices=[8, 16, 32, 64],
                   help="Keep only samples of this block side (pixels)")
    p.add_argument("--qindex", type=int, default=None,
                   help="Keep only samples with this base_qindex")
    p.add_argument("--seq", default=None,
                   help="Optional sequence name recorded in dataset metadata")
    p.add_argument("--dtype", choices=["float32", "uint8"], default="float32",
                   help="Pixel dtype in the dataset (float32 normalized to "
                        "[0,1], or raw uint8)")
    return p.parse_args(argv)


def iter_samples(path):
    """Yield raw PartitionSample tuples from the binary file, validating size."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        filesize = f.tell()
        f.seek(0)
        if SAMPLE_SIZE != EXPECTED_SIZE:
            raise SystemExit(
                "Struct size mismatch: got {}, expected {}. The C layout and "
                "this parser are out of sync.".format(SAMPLE_SIZE, EXPECTED_SIZE))
        if filesize % SAMPLE_SIZE != 0:
            raise SystemExit(
                "Corrupt/truncated file: size {} is not a multiple of the "
                "{}-byte sample.".format(filesize, SAMPLE_SIZE))
        n = filesize // SAMPLE_SIZE
        for _ in range(n):
            yield SAMPLE_STRUCT.unpack(f.read(SAMPLE_SIZE))


def build_dataset(args):
    luma_list, labels, qindex, block_dim = [], [], [], []
    frame_w, frame_h, mi_row, mi_col = [], [], [], []
    sample_ids = []
    # H9 rate-distortion context ([B] neighbor, [C] quant, [E] none RD).
    above_bsz, left_bsz, neigh_av, dc_qs = [], [], [], []
    none_rate, none_dist, none_rdcost = [], [], []
    class_counts, dim_counts = Counter(), Counter()
    kept = total = 0

    for rec in iter_samples(args.input):
        (n_dist, n_rdcost, sample_id, n_rate, fw, fh, mr, mc, dc_q, qidx,
         bit_depth, bsize, bdim, part, above_b, left_b, navl, luma_bytes) = rec
        total += 1

        if part > 9:
            raise SystemExit(
                "Invalid partition label {} at sample {} -- data corruption?"
                .format(part, sample_id))
        if args.block_dim is not None and bdim != args.block_dim:
            continue
        if args.qindex is not None and qidx != args.qindex:
            continue

        full = np.frombuffer(luma_bytes, dtype=np.uint8).reshape(LUMA_DIM, LUMA_DIM)
        block = full[:bdim, :bdim].copy()
        if args.dtype == "float32":
            block = block.astype(np.float32) / 255.0

        luma_list.append(block)
        labels.append(part)
        qindex.append(qidx)
        block_dim.append(bdim)
        frame_w.append(fw)
        frame_h.append(fh)
        mi_row.append(mr)
        mi_col.append(mc)
        sample_ids.append(sample_id)
        above_bsz.append(above_b)
        left_bsz.append(left_b)
        neigh_av.append(navl)
        dc_qs.append(dc_q)
        none_rate.append(n_rate)
        none_dist.append(n_dist)
        none_rdcost.append(n_rdcost)
        class_counts[PARTITION_NAMES[part]] += 1
        dim_counts[bdim] += 1
        kept += 1

    # Stack into a dense (N, dim, dim) array when every block has the same size
    # (e.g. when --block-dim is used); otherwise keep an object array.
    if luma_list and len({b.shape for b in luma_list}) == 1:
        luma_arr = np.stack(luma_list)
    else:
        luma_arr = np.array(luma_list, dtype=object)

    dataset = {
        "luma": luma_arr,
        "partition": np.array(labels, dtype=np.uint8),
        "qindex": np.array(qindex, dtype=np.uint8),
        "block_dim": np.array(block_dim, dtype=np.uint8),
        "frame_width": np.array(frame_w, dtype=np.uint16),
        "frame_height": np.array(frame_h, dtype=np.uint16),
        "mi_row": np.array(mi_row, dtype=np.uint16),
        "mi_col": np.array(mi_col, dtype=np.uint16),
        # Per-process record counter; resets to 0 at each new frame (each frame
        # is a separate aomenc process). Frame boundaries = where sample_id == 0.
        # Enables regrouping per-block samples into per-frame superblock trees.
        "sample_id": np.array(sample_ids, dtype=np.uint32),
        # H9 rate-distortion context.
        "above_bsize": np.array(above_bsz, dtype=np.uint8),   # [B]
        "left_bsize": np.array(left_bsz, dtype=np.uint8),     # [B]
        "neigh_avail": np.array(neigh_av, dtype=np.uint8),    # [B] bit0 up bit1 lf
        "dc_q": np.array(dc_qs, dtype=np.uint16),             # [C]
        "none_rate": np.array(none_rate, dtype=np.uint32),    # [E]
        "none_dist": np.array(none_dist, dtype=np.int64),     # [E]
        "none_rdcost": np.array(none_rdcost, dtype=np.int64), # [E]
        "meta": {
            "sequence": args.seq,
            "source": args.input,
            "pixel_dtype": args.dtype,
            "num_samples": kept,
            "partition_names": PARTITION_NAMES,
            "class_counts": dict(class_counts),
            "block_dim_counts": dict(dim_counts),
            "filter_block_dim": args.block_dim,
            "filter_qindex": args.qindex,
        },
    }
    return dataset, total, kept, class_counts, dim_counts


def main(argv):
    args = parse_args(argv)
    dataset, total, kept, class_counts, dim_counts = build_dataset(args)

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Read {} samples, kept {} after filters.".format(total, kept))
    print("By block size:", dict(sorted(dim_counts.items())))
    print("By partition:", dict(class_counts))
    print("Wrote dataset to {}".format(args.output))


if __name__ == "__main__":
    main(sys.argv[1:])
