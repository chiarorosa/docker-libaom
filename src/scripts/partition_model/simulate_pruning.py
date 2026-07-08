#!/usr/bin/env python3
"""Oracle simulation of the partition-pruning policy -- the Fase C gate.

Before touching any C, replay the student's pruning decisions over held-out
ground truth to estimate, per (tau_none, tau_split) operating point:
  * search reduction: fraction of the baseline's logged partition nodes whose
    RD search would be eliminated (a NONE-commit at a node cuts its whole
    subtree; this is where the speedup comes from), and
  * RD risk: how often the policy forces a decision that contradicts the true
    RD choice -- especially true-SPLIT nodes wrongly committed to NONE.

The pruning policy mirrors Fase D exactly:
  P(NONE)  > tau_none  -> commit NONE  (av1_disable_all_splits): subtree cut.
  P(SPLIT) > tau_split -> force split  (av1_set_square_split_only): recurse.
  otherwise            -> full search.
Student probabilities are computed once per node, so the tau sweep is cheap.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from partition_defs import LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402

CHILD_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def collect_nodes(entries, limit=None):
    """List of superblocks; each is dict[(dim,r,c)] -> {truth, feat, qidx}."""
    sbs = []
    n = 0
    for e in entries:
        for sb in datamod.iter_superblock_members(e["path"]):
            nodes = {}
            for dim, r, c, luma, label in sb["members"]:
                nodes[(dim, r, c)] = {
                    "truth": label,
                    "feat": featmod.block_features(luma, sb["qindex"]),
                }
            sbs.append(nodes)
            n += 1
            if limit and n >= limit:
                return sbs
    return sbs


def score_nodes(sbs, bundle, device):
    """Attach student 3-class probs to every node, computed once."""
    nets = {}
    for dim, _ in LEVELS:
        if dim not in bundle["students"]:
            continue
        net = studentmod.make_student(featmod.NUM_FEATURES, bundle["hidden"])
        net.load_state_dict(bundle["students"][dim])
        net.to(device).eval()
        nets[dim] = net
    for dim, _ in LEVELS:
        if dim not in nets:
            continue
        idx = [(si, key) for si, sb in enumerate(sbs) for key in sb
               if key[0] == dim]
        if not idx:
            continue
        feats = np.stack([sbs[si][key]["feat"] for si, key in idx])
        with torch.no_grad():
            probs = F.softmax(
                nets[dim](torch.tensor(feats, dtype=torch.float32, device=device)),
                dim=-1).cpu().numpy()
        for (si, key), p in zip(idx, probs):
            sbs[si][key]["prob"] = p  # [P(NONE), P(SPLIT), P(REST)]
    return sbs


def _subtree_size(nodes, key):
    dim, r, c = key
    total = 0
    if dim > 8:
        for dr, dc in CHILD_OFFSETS:
            ck = (dim // 2, 2 * r + dr, 2 * c + dc)
            if ck in nodes:
                total += 1 + _subtree_size(nodes, ck)
    return total


def simulate(sbs, tau_none, tau_split):
    """Aggregate policy effects over all superblocks for one operating point."""
    stat = {"evaluated": 0, "saved": 0,
            "none_forced": 0, "none_forced_wrong": 0,
            "split_forced": 0, "split_forced_wrong": 0,
            "true_split": 0, "true_split_cut": 0,
            "true_rect": 0, "true_rect_cut": 0}

    def visit(nodes, key):
        node = nodes.get(key)
        if node is None:
            return
        stat["evaluated"] += 1
        truth = node["truth"]
        prob = node.get("prob")
        dim = key[0]
        children = [(dim // 2, 2 * key[1] + dr, 2 * key[2] + dc)
                    for dr, dc in CHILD_OFFSETS] if dim > 8 else []
        present_children = [c for c in children if c in nodes]

        if prob is None:
            for c in present_children:
                visit(nodes, c)
            return

        p_none, p_split = float(prob[0]), float(prob[1])
        if p_none > tau_none:
            saved = sum(1 + _subtree_size(nodes, c) for c in present_children)
            stat["saved"] += saved
            stat["none_forced"] += 1
            if truth != 0:
                stat["none_forced_wrong"] += 1
            if truth == 3:
                stat["true_split"] += 1
                stat["true_split_cut"] += 1
            elif truth not in (0, 3):
                stat["true_rect"] += 1
                stat["true_rect_cut"] += 1
            return  # subtree cut
        if truth == 3:
            stat["true_split"] += 1
        elif truth not in (0, 3):
            stat["true_rect"] += 1
        if p_split > tau_split and dim > 8:
            stat["split_forced"] += 1
            if truth != 3:
                stat["split_forced_wrong"] += 1
        for c in present_children:
            visit(nodes, c)

    for nodes in sbs:
        visit(nodes, (64, 0, 0))
    return stat


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset")
    p.add_argument("--students", default="/workspace/results/models/student/"
                                        "students.pt")
    p.add_argument("--val-seqs", nargs="+", default=["Jockey"])
    p.add_argument("--tau-none", type=float, nargs="+",
                   default=[0.6, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--tau-split", type=float, nargs="+",
                   default=[0.6, 0.7, 0.8, 0.9])
    p.add_argument("--out-csv", default="/workspace/results/models/student/"
                                        "oracle_sim.csv")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.students, map_location=device)
    entries = datamod.discover_pkls(args.dataset_dir)
    _, val_e = datamod.split_entries(entries, args.val_seqs, None)
    print("oracle sim over {} val pkl(s) (seqs {})".format(
        len(val_e), args.val_seqs))
    sbs = collect_nodes(val_e, limit=args.limit)
    sbs = score_nodes(sbs, bundle, device)
    total_nodes = sum(len(s) for s in sbs)
    print("superblocks: {}, nodes: {}".format(len(sbs), total_nodes))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    header = ("tau_none tau_split  search_red%  none_wrong%  splitLost%  "
              "rectLost%  splitForce_wrong%")
    print(header)
    for tn in args.tau_none:
        for ts in args.tau_split:
            s = simulate(sbs, tn, ts)
            red = 100.0 * s["saved"] / max(s["evaluated"], 1)
            nwrong = 100.0 * s["none_forced_wrong"] / max(s["none_forced"], 1)
            slost = 100.0 * s["true_split_cut"] / max(s["true_split"], 1)
            rlost = 100.0 * s["true_rect_cut"] / max(s["true_rect"], 1)
            sfw = 100.0 * s["split_forced_wrong"] / max(s["split_forced"], 1)
            rows.append((tn, ts, red, nwrong, slost, rlost, sfw))
            print("  {:.2f}     {:.2f}      {:6.2f}      {:6.2f}     {:6.2f}    "
                  "{:6.2f}     {:6.2f}".format(tn, ts, red, nwrong, slost, rlost,
                                               sfw))
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_none", "tau_split", "search_reduction_pct",
                    "none_forced_wrong_pct", "true_split_lost_pct",
                    "true_rect_lost_pct", "split_forced_wrong_pct"])
        w.writerows(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
