#!/usr/bin/env python3
"""RPP -- information ladder over the causal-context columns (ICASSP study).

The question is not which pruner is faster. It is WHERE the information that
separates a low-loss partition decision from a costly one actually lives. The
thesis answers that in aggregate: 24 compact descriptors beat isolated variance,
and adding 12 causal columns beats them again. Two confounders make that ladder
weaker than it looks, and this script removes both.

  1. TRAINING PROCEDURE. The 24-column arm scored by `oracle_regret.py` is
     `student_real`, a DISTILLED artifact (`distill.py` from `surrogate_real`,
     alpha=0.5 CE + KL against the ConvNeXt teacher). The 36-column arm,
     `student_h9a`, is direct hard-label cross-entropy with no teacher. So the
     measured step confounds the feature set with the training objective. Here
     every rung is trained by ONE fixed recipe.

  2. RUN-TO-RUN VARIANCE. Nothing in the training path is seeded, so a single
     run per rung cannot tell a real gap from initialization noise. Here every
     rung is trained under several seeds and the spread is reported.

The ladder also splits the 12 causal columns, which the thesis only ever
measured together:

    A      0..23   21 luma descriptors + q_norm (17) + pos_r/pos_c (22,23)
    A+B    0..31   + causal partitioning neighborhood of the above/left blocks
    A+C    0..23,32..35   + effective dequant step, frame position, node depth
    A+B+C  0..35   the full free vector

Note that A already carries a quantization index and an intra-superblock
position. Whatever A+B+C buys over A is therefore NOT the model discovering
quantization: that column was there from the start. Splitting B from C is what
turns "causal context helps" into a statement about WHICH causal context.

Superblocks are read from disk ONCE and reused across every rung and seed; the
pkl read, not the training, is what costs.

Usage (container):
  venv-ml/bin/python src/scripts/partition_model/rpp_ladder.py \
    --out-dir results/models/rpp_ladder --seeds 0 1 2
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
from distill import train_student  # noqa: E402
from train_student_h9 import collect_by_dim, TRAIN_SEQS, VAL_SEQS  # noqa: E402


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--out-dir",
                   default="/workspace/results/models/rpp_ladder")
    p.add_argument("--rungs", nargs="+", default=["A", "A_B", "A_C", "A_B_C"],
                   choices=sorted(featmod.RPP_SUBSETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # Held fixed across every rung: this is the whole point of the experiment.
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-pkl", type=int, default=3000,
                   help="cap superblocks per pkl; matches the deployed "
                        "student_h9a training command")
    p.add_argument("--limit", type=int, default=None, help="smoke cap")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, args.val_seqs,
                                       args.train_seqs)
    datamod.assert_real_luma(train_e)
    print("train pkls: {} ({})".format(len(train_e), args.train_seqs),
          flush=True)

    t0 = time.time()
    data, n_sb = collect_by_dim(train_e, per_pkl=(args.per_pkl or None),
                                limit=args.limit, feature_set="h9a")
    print("collected {} superblocks in {:.1f} min".format(
        n_sb, (time.time() - t0) / 60.0), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for rung in args.rungs:
        cols = featmod.RPP_SUBSETS[rung]
        for seed in args.seeds:
            tag = "{}_s{}".format(rung, seed)
            out = os.path.join(args.out_dir, tag)
            os.makedirs(out, exist_ok=True)
            bundle = {"hidden": args.hidden, "students": {}, "norm": {},
                      "num_features": len(cols), "feature_set": "h9a",
                      "rung": rung, "cols": cols, "seed": seed,
                      "class_weight": False}
            for dim, _ in MODEL_LEVELS:
                src = data[dim]
                if len(src["truth"]) == 0:
                    continue
                # Seed per (rung, level) so the initialization a rung sees does
                # not depend on how many rungs ran before it.
                torch.manual_seed(seed * 1000 + dim)
                np.random.seed(seed * 1000 + dim)
                rec = {"feat": src["feat"][:, cols], "truth": src["truth"],
                       "teacher": np.full((len(src["truth"]), 3), 1.0 / 3.0,
                                          dtype=np.float32)}
                net, norm = train_student(rec, args.hidden, device,
                                          args.epochs, args.lr, alpha=1.0,
                                          temp=1.0, use_class_weight=False,
                                          in_features=len(cols))
                with torch.no_grad():
                    fb = torch.tensor(rec["feat"], dtype=torch.float32,
                                      device=device)
                    pred = net(fb).argmax(-1).cpu().numpy()
                acc = float((pred == rec["truth"]).mean())
                bundle["students"][dim] = net.state_dict()
                bundle["norm"][dim] = norm
                rows.append({"rung": rung, "seed": seed, "dim": dim,
                             "n_features": len(cols), "n": len(pred),
                             "train_acc": round(acc, 5)})
                print("[{} dim {:>2}] n={} feats={} train-acc={:.3f}".format(
                    tag, dim, len(pred), len(cols), acc), flush=True)
            torch.save(bundle, os.path.join(out, "students.pt"))

    with open(os.path.join(args.out_dir, "training.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rung", "seed", "dim", "n_features",
                                          "n", "train_acc"])
        w.writeheader()
        w.writerows(rows)
    print("Saved -> {} ({} bundles)".format(
        args.out_dir, len(args.rungs) * len(args.seeds)))
    print("Training accuracy is reported for provenance only; the ladder is "
          "adjudicated by oracle_regret.py, not by per-node accuracy.")


if __name__ == "__main__":
    main(sys.argv[1:])
