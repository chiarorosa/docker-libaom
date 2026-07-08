#!/usr/bin/env python3
"""Dataset assembly for the ConvNeXt-AV1 partition surrogate.

Turns the flat per-block PartitionSample records (stored in per-(seq,QP) .pkl
files) into per-superblock training examples: one 64x64 luma input paired with
the ground-truth partition decision at every quadtree level (64/32/16/8).

Design points:
  * Dynamic discovery. The set of .pkl files is read from manifest.csv (or a
    glob fallback) -- never hard-coded -- so the same code scales from the 4
    construction sequences to the full UVG 4K dataset with no edits.
  * Per-sequence split. Train/val are separated by sequence name so the
    reported number is a genuine held-out generalization test.
  * Frame-aware regrouping. Each frame is a separate aomenc process, so
    sample_id resets to 0 at each new frame; frame index = cumsum(sample_id==0).
    Samples are grouped by (frame, superblock) via mi_row/mi_col.
  * Assembled tensors are cached to a single .npz per split (the regroup pass
    reads several GB of pickles; caching makes re-training cheap).
"""

import argparse
import csv
import glob
import hashlib
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from partition_defs import (  # noqa: E402
    LEVELS, SB_SIZE_MI, SB_SIZE_PX, NUM_PARTITION_TYPES, sb_cell, legal_classes,
)

IGNORE = -1  # label sentinel for "no sample at this quadtree cell"


# --------------------------------------------------------------------------
# Discovery + split
# --------------------------------------------------------------------------
def discover_pkls(dataset_dir):
    """Return list of dicts {path, sequence, cq} for every dataset .pkl.

    Prefers manifest.csv (authoritative ledger); falls back to globbing *.pkl
    and parsing the sequence/cq from the filename stem (Name_..._cqNN.pkl)."""
    manifest = os.path.join(dataset_dir, "manifest.csv")
    entries = []
    if os.path.exists(manifest):
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                path = row.get("pkl_path", "")
                # Manifest stores container-absolute paths; resolve to the given
                # dataset_dir so this works on host or in-container.
                if not os.path.exists(path):
                    path = os.path.join(dataset_dir, os.path.basename(path))
                if os.path.exists(path):
                    entries.append({"path": path,
                                    "sequence": row.get("sequence", ""),
                                    "cq": int(row.get("cq_level", -1))})
    if not entries:
        for path in sorted(glob.glob(os.path.join(dataset_dir, "*.pkl"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            seq, cq = stem, -1
            if "_cq" in stem:
                seq, _, tail = stem.rpartition("_cq")
                try:
                    cq = int(tail)
                except ValueError:
                    pass
            entries.append({"path": path, "sequence": seq, "cq": cq})
    return entries


def split_entries(entries, val_seqs, train_seqs=None):
    """Partition entries by sequence name (case-insensitive substring match).

    An entry is validation if any val token matches its sequence; otherwise it
    is training, unless train_seqs is given, in which case training is
    restricted to matching sequences (everything else is dropped)."""
    def matches(seq, tokens):
        s = seq.lower()
        return any(t.lower() in s for t in tokens)

    train, val = [], []
    for e in entries:
        if matches(e["sequence"], val_seqs):
            val.append(e)
        elif train_seqs is None or matches(e["sequence"], train_seqs):
            train.append(e)
    return train, val


# --------------------------------------------------------------------------
# Regroup one pkl into superblocks
# --------------------------------------------------------------------------
def _frame_index(sample_id):
    """Per-sample frame index; a frame boundary is sample_id resetting to 0."""
    return np.cumsum(sample_id == 0) - 1


def _load_and_group(path):
    """Load one .pkl and group sample indices by (frame, superblock).
    Returns (arrays_dict, groups) where groups maps (frame, sb_row, sb_col) to a
    list of sample indices."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "sample_id" not in d:
        raise SystemExit(
            "{}: pkl has no 'sample_id' (regenerate with the updated converter; "
            "frame regrouping needs it).".format(path))
    arr = {
        "luma": d["luma"],
        "part": np.asarray(d["partition"]),
        "qidx": np.asarray(d["qindex"]),
        "bdim": np.asarray(d["block_dim"]),
        "mir": np.asarray(d["mi_row"]).astype(np.int64),
        "mic": np.asarray(d["mi_col"]).astype(np.int64),
    }
    frame = _frame_index(np.asarray(d["sample_id"]))
    sb_row = arr["mir"] // SB_SIZE_MI
    sb_col = arr["mic"] // SB_SIZE_MI
    groups = {}
    for i in range(len(arr["part"])):
        groups.setdefault((int(frame[i]), int(sb_row[i]), int(sb_col[i])),
                          []).append(i)
    return arr, groups


def iter_superblock_members(path):
    """Yield rich superblock records from one .pkl:
      {luma:(64,64)uint8, qindex:int,
       members:[(block_dim, cell_r, cell_c, block_luma, label), ...]}
    Members carry each visited node's own luma crop and grid cell, used by the
    distillation step to pair per-block features with the surrogate's per-cell
    prediction. Superblocks lacking a 64x64 root are skipped."""
    arr, groups = _load_and_group(path)
    luma, bdim, part, qidx = arr["luma"], arr["bdim"], arr["part"], arr["qidx"]
    mir, mic = arr["mir"], arr["mic"]
    for (_, sr, sc), idxs in groups.items():
        root = next((i for i in idxs if bdim[i] == SB_SIZE_PX), None)
        if root is None:
            continue
        block = np.asarray(luma[root])
        if block.shape != (SB_SIZE_PX, SB_SIZE_PX):
            continue
        members, ok = [], True
        for i in idxs:
            dim = int(bdim[i])
            if dim not in (d for d, _ in LEVELS):
                continue
            r, c = sb_cell(dim, mir[i], mic[i], sr * SB_SIZE_MI, sc * SB_SIZE_MI)
            side = SB_SIZE_PX // dim
            if not (0 <= r < side and 0 <= c < side):
                ok = False
                break
            members.append((dim, r, c, np.asarray(luma[i]), int(part[i])))
        if not ok:
            continue
        yield {"luma": block.astype(np.uint8), "qindex": int(qidx[root]),
               "members": members}


def assemble_pkl(path):
    """Yield superblock dicts {luma, qindex, labels:{dim:(side,side)}} for
    surrogate training; labels use IGNORE for cells no RD node visited."""
    for sb in iter_superblock_members(path):
        labels = {dim: np.full((side, side), IGNORE, dtype=np.int64)
                  for dim, side in LEVELS}
        for dim, r, c, _luma, label in sb["members"]:
            labels[dim][r, c] = label
        yield {"luma": sb["luma"], "qindex": sb["qindex"], "labels": labels}


# --------------------------------------------------------------------------
# Assemble + cache a whole split
# --------------------------------------------------------------------------
def _signature(entries):
    h = hashlib.sha1()
    for e in sorted(entries, key=lambda x: x["path"]):
        st = os.stat(e["path"])
        h.update(os.path.basename(e["path"]).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def assemble_split(entries, cache_dir=None, tag="split", rebuild=False,
                   limit=None, verbose=True):
    """Assemble (and cache) all superblocks for a list of pkl entries.

    Returns a dict of arrays: luma (N,64,64)uint8, qindex (N,)uint8, and one
    label array per level lab{dim} (N,side,side)int8 with IGNORE for empty
    cells. `limit` caps the superblock count (smoke tests)."""
    sig = _signature(entries) if entries else "empty"
    cache = (os.path.join(cache_dir, "{}_{}.npz".format(tag, sig))
             if cache_dir else None)
    if cache and os.path.exists(cache) and not rebuild:
        if verbose:
            print("[data] loading cached split:", cache)
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    luma, qidx = [], []
    labs = {dim: [] for dim, _ in LEVELS}
    for e in entries:
        n0 = len(luma)
        for sb in assemble_pkl(e["path"]):
            luma.append(sb["luma"])
            qidx.append(sb["qindex"])
            for dim, _ in LEVELS:
                labs[dim].append(sb["labels"][dim])
            if limit and len(luma) >= limit:
                break
        if verbose:
            print("[data] {}: +{} superblocks ({} total)".format(
                os.path.basename(e["path"]), len(luma) - n0, len(luma)))
        if limit and len(luma) >= limit:
            break

    out = {
        "luma": np.asarray(luma, dtype=np.uint8),
        "qindex": np.asarray(qidx, dtype=np.uint8),
    }
    for dim, _ in LEVELS:
        out["lab{}".format(dim)] = np.asarray(labs[dim], dtype=np.int8)
    if cache and not limit:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, **out)
        if verbose:
            print("[data] cached split ->", cache)
    return out


def level_class_counts(assembled):
    """Per-level histogram of partition labels (ignores empty cells).
    Returns {dim: np.array(NUM_PARTITION_TYPES)}. Used for class weighting."""
    counts = {}
    for dim, _ in LEVELS:
        lab = assembled["lab{}".format(dim)].reshape(-1)
        lab = lab[lab >= 0]
        counts[dim] = np.bincount(lab, minlength=NUM_PARTITION_TYPES)
    return counts


# --------------------------------------------------------------------------
# torch Dataset (imported lazily so this module is usable without torch)
# --------------------------------------------------------------------------
def make_torch_dataset(assembled):
    import torch
    from torch.utils.data import Dataset

    class SuperblockDataset(Dataset):
        def __init__(self, data):
            self.luma = data["luma"]
            self.qindex = data["qindex"]
            self.labs = {dim: data["lab{}".format(dim)] for dim, _ in LEVELS}

        def __len__(self):
            return len(self.luma)

        def __getitem__(self, i):
            luma = torch.from_numpy(
                self.luma[i].astype(np.float32) / 255.0)
            q = float(self.qindex[i]) / 255.0
            qplane = torch.full((SB_SIZE_PX, SB_SIZE_PX), q, dtype=torch.float32)
            x = torch.stack([luma, qplane], dim=0)  # (2,64,64)
            labels = {dim: torch.from_numpy(self.labs[dim][i].astype(np.int64))
                      for dim, _ in LEVELS}
            return x, labels

    return SuperblockDataset(assembled)


def main(argv):
    p = argparse.ArgumentParser(description="Assemble/inspect the surrogate "
                                            "superblock dataset.")
    p.add_argument("--dataset-dir", default="/workspace/results/dataset")
    p.add_argument("--val-seqs", nargs="+", default=["Jockey"])
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--cache-dir", default="/workspace/results/models/cache")
    p.add_argument("--limit", type=int, default=None,
                   help="cap superblocks per split (smoke test)")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)

    entries = discover_pkls(args.dataset_dir)
    print("Discovered {} pkl(s) in {}".format(len(entries), args.dataset_dir))
    for e in entries:
        print("  {:<45} cq={}".format(e["sequence"][:44], e["cq"]))
    train, val = split_entries(entries, args.val_seqs, args.train_seqs)
    print("Split: {} train pkl(s), {} val pkl(s) (val seqs={})".format(
        len(train), len(val), args.val_seqs))

    for tag, ents in (("train", train), ("val", val)):
        if not ents:
            print("[{}] no entries".format(tag))
            continue
        data = assemble_split(ents, args.cache_dir, tag=tag,
                              rebuild=args.rebuild, limit=args.limit)
        print("[{}] {} superblocks; luma {}".format(
            tag, len(data["luma"]), data["luma"].shape))
        counts = level_class_counts(data)
        for dim, _ in LEVELS:
            legal = sorted(legal_classes(dim))
            print("  level {:>2}px counts (legal {}): {}".format(
                dim, legal, counts[dim].tolist()))


if __name__ == "__main__":
    main(sys.argv[1:])
