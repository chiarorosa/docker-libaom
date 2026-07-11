#!/usr/bin/env python3
"""Convert the working .pkl dataset to an archival .npz form (Zenodo/DOI).

Why: the .pkl payload is Python-pickle (version-fragile, executes code on load)
and stores luma as float32 (4x larger than needed). This produces, per input
pkl, one compressed .npz that is:
  * smaller  -- luma stored as uint8 [0,255] (the float32 pkl is exactly
                uint8/255; round(x*255) recovers it losslessly);
  * portable -- pure numpy arrays, loadable with allow_pickle=False, no code
                execution, stable across numpy versions;
  * complete -- carries the same fields the pipeline uses (luma + labels +
                coordinates + the H9 rate-distortion context B/C/E).

Ragged luma encoding (blocks have different sizes, so no dense array):
  luma_flat    uint8  [sum_i dim_i*dim_i]   all blocks concatenated, row-major
  luma_offsets int64  [N+1]                 sample i = luma_flat[off[i]:off[i+1]]
                                            reshaped to (block_dim[i], block_dim[i])
All other arrays are fixed-shape and stored natively. See docs/ZENODO_datasheet.md
for the schema and a reference loader.

Usage (does not run automatically):
  venv-ml/bin/python pkl_to_npz.py --in-dir results/dataset_h9 \
      --out-dir results/dataset_h9_npz
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# Scalar/per-sample fields carried through unchanged (name -> archival dtype).
SCALAR_FIELDS = {
    "partition": np.uint8,
    "qindex": np.uint8,
    "block_dim": np.uint8,
    "frame_width": np.uint16,
    "frame_height": np.uint16,
    "mi_row": np.uint16,
    "mi_col": np.uint16,
    "sample_id": np.uint32,
    # H9 rate-distortion context.
    "above_bsize": np.uint8,
    "left_bsize": np.uint8,
    "neigh_avail": np.uint8,
    "dc_q": np.uint16,
    "none_rate": np.uint32,
    "none_dist": np.int64,
    "none_rdcost": np.int64,
}


def _luma_to_uint8(block):
    """Recover 8-bit pixels from a stored block (float32 [0,1] or uint8)."""
    b = np.asarray(block)
    if np.issubdtype(b.dtype, np.floating):
        return np.rint(b * 255.0).clip(0, 255).astype(np.uint8)
    return b.astype(np.uint8)


def convert_one(pkl_path, out_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    n = len(d["partition"])
    luma = d["luma"]

    # Build the ragged uint8 luma buffer + offsets.
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    parts = []
    for i in range(n):
        u8 = _luma_to_uint8(luma[i]).reshape(-1)
        parts.append(u8)
        offsets[i + 1] = offsets[i] + u8.size
    luma_flat = (np.concatenate(parts) if parts
                 else np.zeros(0, dtype=np.uint8))

    out = {"luma_flat": luma_flat, "luma_offsets": offsets}
    for name, dt in SCALAR_FIELDS.items():
        if name in d:
            out[name] = np.asarray(d[name]).astype(dt)

    # Preserve provenance metadata as a JSON string (numpy-native, no pickle).
    import json
    meta = dict(d.get("meta", {}))
    meta["archival_note"] = ("luma stored as uint8 [0,255]; ragged via "
                             "luma_flat + luma_offsets; converted by pkl_to_npz.py")
    out["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(out_path, **out)
    return n, luma_flat.nbytes


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--out-dir", default="/workspace/results/dataset_h9_npz")
    p.add_argument("--overwrite", action="store_true",
                   help="reconvert even if the .npz already exists")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    pkls = sorted(glob.glob(os.path.join(args.in_dir, "*.pkl")))
    if not pkls:
        raise SystemExit("no .pkl in " + args.in_dir)
    # Copy the manifest verbatim (authoritative ledger).
    man = os.path.join(args.in_dir, "manifest.csv")
    if os.path.exists(man):
        import shutil
        shutil.copy2(man, os.path.join(args.out_dir, "manifest.csv"))

    total_n = 0
    for pk in pkls:
        stem = os.path.splitext(os.path.basename(pk))[0]
        out = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out) and not args.overwrite:
            print("skip (exists):", stem, flush=True)
            continue
        n, luma_bytes = convert_one(pk, out)
        total_n += n
        print("  {:<50} {:>9} samples  luma {:>6.1f} MB".format(
            stem, n, luma_bytes / 1e6), flush=True)
    print("done: {} pkl(s), {} samples -> {}".format(
        len(pkls), total_n, args.out_dir))


if __name__ == "__main__":
    main(sys.argv[1:])
