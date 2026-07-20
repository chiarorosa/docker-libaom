#!/usr/bin/env python3
"""Fase 3 -- train the deployable per-block-size student directly on the H9a
feature vector (A pixels + B neighbor context + C quant/position = 36 features)
with hard-label cross-entropy (no teacher). Reproduces, in the deployable
artifact, the winning Gate 2 configuration.

The distilled ConvNeXt teacher is deliberately not used: it is pixel-only and
cannot carry the B/C signal that dataset_h9 exists to capture (see
docs/ANDAMENTO_tese.md 1.2 / 3). The output bundle is a superset of the pixel
student's, adding num_features and feature_set so export_weights.py and
simulate_pruning.py score with the right input width.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from distill import train_student  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def collect_by_dim(entries, per_pkl=None, limit=None, feature_set="h9a"):
    """Per-block-size {'feat':(N,36 or 42), 'truth':(N,)} arrays over the
    H9-instrumented dataset. per_pkl caps superblocks taken from each pkl
    (diverse sampling across seqs/QPs); limit is a global superblock cap.
    feature_set='h9a_b1' adds the 6 hereditary RD-context features (B1)."""
    acc = {dim: {"feat": [], "truth": []} for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: no RD context; re-extract with the H9 "
                                 "instrumentation.".format(e["path"]))
            if feature_set == "h9a_b1":
                node_ctx = {(dim, r, c): ctx for (dim, r, c, _luma, _label), ctx
                           in zip(sb["members"], sb["ctx"])}
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in acc:
                    continue
                if feature_set == "h9a_b1":
                    f = featmod.node_features_h9a_b1(
                        sb["luma"], dim, r, c, sb["qindex"], sb["ctx"][k],
                        node_ctx)
                else:
                    f = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                                  sb["qindex"], sb["ctx"][k])
                acc[dim]["feat"].append(f)
                acc[dim]["truth"].append(studentmod.collapse_label(label))
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and n_sb >= limit:
                return _finalize(acc), n_sb
    return _finalize(acc), n_sb


def _finalize(acc):
    return {dim: {"feat": np.asarray(v["feat"], dtype=np.float32),
                  "truth": np.asarray(v["truth"], dtype=np.int64)}
            for dim, v in acc.items()}


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--out-dir", default="/workspace/results/models/student_h9a")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-pkl", type=int, default=3000,
                   help="cap superblocks per pkl (diverse sampling); 0 = all")
    p.add_argument("--limit", type=int, default=None,
                   help="global superblock cap (smoke)")
    p.add_argument("--class-weight", action="store_true",
                   help="per-level inverse-frequency class weighting (B4). "
                        "Default off reproduces the deployed student_h9a; on "
                        "trains the class-weighted variant student_h9a_cw (see "
                        "docs/RESULTADOS_modelagem_B4_ponderacao_classe.md).")
    p.add_argument("--feature-set", choices=["h9a", "h9a_b1"], default="h9a",
                   help="'h9a' (default, 36 feats, deployed behavior) or "
                        "'h9a_b1' (42 feats: h9a + 6 hereditary RD-context "
                        "features from the parent and earlier siblings, B1).")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, args.val_seqs, args.train_seqs)
    datamod.assert_real_luma(train_e)
    print("train pkls: {} (train seqs {})".format(
        len(train_e), args.train_seqs), flush=True)
    per_pkl = args.per_pkl or None
    data, n_sb = collect_by_dim(train_e, per_pkl=per_pkl, limit=args.limit,
                               feature_set=args.feature_set)
    print("collected {} superblocks".format(n_sb), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    nfa = (featmod.NUM_FEATURES_H9A_B1 if args.feature_set == "h9a_b1"
          else featmod.NUM_FEATURES_H9A)
    bundle = {"hidden": args.hidden, "students": {}, "norm": {},
              "num_features": nfa, "feature_set": args.feature_set,
              "class_weight": bool(args.class_weight)}
    for dim, _ in MODEL_LEVELS:
        rec = data[dim]
        if len(rec["truth"]) == 0:
            print("[dim {}] no samples, skipping".format(dim))
            continue
        rec = {"feat": rec["feat"], "truth": rec["truth"],
               "teacher": np.full((len(rec["truth"]), 3), 1.0 / 3.0,
                                  dtype=np.float32)}
        net, norm = train_student(rec, args.hidden, device, args.epochs,
                                  args.lr, alpha=1.0, temp=1.0,
                                  use_class_weight=args.class_weight,
                                  in_features=nfa)
        with torch.no_grad():
            fb = torch.tensor(rec["feat"], dtype=torch.float32, device=device)
            pred = net(fb).argmax(-1).cpu().numpy()
        counts = np.bincount(rec["truth"], minlength=3).tolist()
        acc = float((pred == rec["truth"]).mean())
        print("[dim {:>2}] n={} counts(N/S/R)={} truth-acc={:.3f}".format(
            dim, len(pred), counts, acc), flush=True)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm

    torch.save(bundle, os.path.join(args.out_dir, "students.pt"))
    print("Saved ->", os.path.join(args.out_dir, "students.pt"))


if __name__ == "__main__":
    main(sys.argv[1:])
