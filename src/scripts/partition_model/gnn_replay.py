#!/usr/bin/env python3
"""Approach B: score the deployable pixel-only GNN over SOURCE frames and dump
per-node probs in the EXACT format the encoder's H8 replay hook reads
(AV1_STUDENT_PROBS_FILE). The GNN's input is source luma + qindex only (feat_mode
"pixelquant" -- block B, the decision-dependent neighbor-partition context, is
never fed), so its in-loop decisions can be precomputed EXACTLY outside the
encoder, same trick as surrogate_replay.py. This turns the encoder into a
measurement device for the GNN's real BD-rate x speedup -- no GNN inference in C.

Record layout (little-endian), one per whole 64x64 superblock per encoded frame:
  int32 frame (encode order, 0-based), int32 sb_row, int32 sb_col,
  float32 probs[21][3]  -- nodes ordered [64] ++ [32 row-major] ++ [16 rm],
                           each [P(NONE), P(SPLIT), P(REST)].

The qindex fed to the GNN mirrors aomenc --end-usage=q: qindex = 4*cq. The
per-node dc_q context feature (block C) uses the AV1 luma DC dequant step for
that qindex, given via a built-in cq->dc_q map (override with --dc-q).
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

import features as featmod  # noqa: E402
import graph_data as gd  # noqa: E402
import gnn_model as gm  # noqa: E402

# AV1 luma DC dequant step per qindex (qindex = 4*cq), extracted from the dataset.
DCQ_MAP = {20: 74, 32: 140, 43: 265, 55: 522}


def read_y(path, w, h, idx):
    fb = w * h * 3 // 2
    with open(path, "rb") as f:
        f.seek(idx * fb)
        return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)


def _node_keys():
    """21 nodes in output order: [64] ++ [32 row-major] ++ [16 row-major]."""
    keys = [(64, 0, 0)]
    for r in range(2):
        for c in range(2):
            keys.append((32, r, c))
    for r in range(4):
        for c in range(4):
            keys.append((16, r, c))
    return keys


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gnn-bundle", required=True)
    p.add_argument("--yuv", required=True)
    p.add_argument("--cq", type=int, required=True)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128,
                   help="superblocks per forward pass (block-diagonal batch)")
    p.add_argument("--dc-q", type=int, default=None,
                   help="override the built-in cq->dc_q map (needed if --cq "
                        "isn't one of {}".format(sorted(DCQ_MAP)))
    args = p.parse_args(argv)

    m = re.search(r"(\d+)x(\d+)", os.path.basename(args.yuv))
    if not m:
        raise SystemExit("cannot parse WxH from " + args.yuv)
    w, h = int(m.group(1)), int(m.group(2))
    sb_rows, sb_cols = h // 64, w // 64  # whole units only (matches the gate)
    qindex = 4 * args.cq
    dc_q = args.dc_q if args.dc_q is not None else DCQ_MAP.get(args.cq)
    if dc_q is None:
        raise SystemExit("no built-in dc_q for cq={}; pass --dc-q".format(args.cq))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.gnn_bundle, map_location=device)
    assert bundle.get("head") == "gnn3", "unexpected bundle format: {}".format(
        args.gnn_bundle)
    assert bundle.get("feat_mode", "h9a") == "pixelquant", (
        "gnn_replay.py requires a pixelquant (deployable) bundle, got feat_mode={}"
        .format(bundle.get("feat_mode")))
    net = gm.build_gnn(bundle["num_features"], bundle["hidden"],
                       bundle["layer"], bundle["n_layers"]).to(device).eval()
    net.load_state_dict(bundle["state_dict"])

    node_keys = _node_keys()
    n_nodes = len(node_keys)
    edge_list = gd.sb_edges(node_keys)
    base_edges = (np.asarray(edge_list, dtype=np.int64).T if edge_list
                  else np.zeros((2, 0), np.int64))
    feat_dim = gd.FEAT_DIMS["pixelquant"]

    n_rec = 0
    with open(args.out, "wb") as out:
        for fi in range(args.frames):
            y = read_y(args.yuv, w, h, args.skip + fi)
            keys, batch_x = [], []

            def flush():
                nonlocal n_rec
                if not batch_x:
                    return
                n_sb = len(batch_x)
                x = torch.from_numpy(np.concatenate(batch_x, axis=0)).to(device)
                if base_edges.size:
                    ei_np = np.concatenate(
                        [base_edges + i * n_nodes for i in range(n_sb)], axis=1)
                else:
                    ei_np = np.zeros((2, 0), np.int64)
                ei = torch.from_numpy(ei_np).to(device)
                with torch.no_grad():
                    logits = net(x, ei)
                    probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
                for k, (sr, sc) in enumerate(keys):
                    rec = [struct.pack("<3i", fi, sr, sc)]
                    node_probs = probs[k * n_nodes:(k + 1) * n_nodes]
                    for p3 in node_probs:
                        rec.append(struct.pack(
                            "<3f", *(float(v) for v in p3)))
                    out.write(b"".join(rec))
                    n_rec += 1
                keys.clear()
                batch_x.clear()

            for sr in range(sb_rows):
                for sc in range(sb_cols):
                    luma = y[sr * 64:(sr + 1) * 64, sc * 64:(sc + 1) * 64]
                    feats = np.empty((n_nodes, feat_dim), dtype=np.float32)
                    for ni, (dim, cr, cc) in enumerate(node_keys):
                        ctx = {"dc_q": dc_q,
                               "mi_row": sr * 16 + cr * (dim // 4),
                               "mi_col": sc * 16 + cc * (dim // 4),
                               "frame_w": w, "frame_h": h,
                               "neigh_avail": 0}
                        f36 = featmod.node_features_h9a(
                            luma, dim, cr, cc, qindex, ctx)
                        feats[ni] = gd.slice_feat(f36, "pixelquant")
                    batch_x.append(feats)
                    keys.append((sr, sc))
                    if len(batch_x) >= args.batch:
                        flush()
            flush()
            print("frame {}: {} records total".format(fi, n_rec), flush=True)
    print("wrote {} ({} records)".format(args.out, n_rec))


if __name__ == "__main__":
    main(sys.argv[1:])
