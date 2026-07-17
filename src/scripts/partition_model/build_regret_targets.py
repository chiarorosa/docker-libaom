#!/usr/bin/env python3
"""Monta o dataset de treino da Solução 4: para cada nó de decisão dos superblocos
do dataset_h9, empilha a feature H9a (36) e o alvo regret_rel (regret.node_regrets).
Espelha train_student_h9.collect_by_dim, trocando o rótulo pelo regret contínuo."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import regret as regretmod  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def collect_regret_by_dim(entries, per_pkl=None, limit=None, exact_only=True):
    acc = {dim: {"feat": [], "regret": [], "exact": []} for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD (dataset_h9 requerido)."
                                 .format(e["path"]))
            for nr in regretmod.node_regrets(sb["members"], sb["ctx"]):
                dim = nr["dim"]
                if dim not in acc:
                    continue
                if exact_only and not nr["exact"]:
                    continue
                k = nr["k"]
                _dim, r, c, _luma, _label = sb["members"][k]
                f = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                              sb["qindex"], sb["ctx"][k])
                acc[dim]["feat"].append(f)
                acc[dim]["regret"].append(nr["regret_rel"])
                acc[dim]["exact"].append(nr["exact"])
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and n_sb >= limit:
                return _finalize(acc)
    return _finalize(acc)


def _finalize(acc):
    return {dim: {"feat": np.asarray(v["feat"], dtype=np.float32),
                  "regret": np.asarray(v["regret"], dtype=np.float32),
                  "exact": np.asarray(v["exact"], dtype=bool)}
            for dim, v in acc.items()}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--per-pkl", type=int, default=3000)
    p.add_argument("--exact-only", action="store_true", default=True)
    p.add_argument("--include-censored", dest="exact_only", action="store_false")
    p.add_argument("--out", default="/workspace/results/models/regret/targets_train.npz")
    args = p.parse_args(argv)

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs, args.train_seqs)
    ents = train_e if args.split == "train" else val_e
    datamod.assert_real_luma(ents)
    per_pkl = args.per_pkl or None
    out = collect_regret_by_dim(ents, per_pkl=per_pkl, exact_only=args.exact_only)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    flat = {}
    for dim, v in out.items():
        flat["feat_{}".format(dim)] = v["feat"]
        flat["regret_{}".format(dim)] = v["regret"]
        print("[dim {:>2}] n={} regret[min/med/max]={:.4f}/{:.4f}/{:.4f}".format(
            dim, len(v["regret"]),
            float(v["regret"].min()) if len(v["regret"]) else 0.0,
            float(np.median(v["regret"])) if len(v["regret"]) else 0.0,
            float(v["regret"].max()) if len(v["regret"]) else 0.0), flush=True)
    np.savez_compressed(args.out, **flat)
    print("Saved ->", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
