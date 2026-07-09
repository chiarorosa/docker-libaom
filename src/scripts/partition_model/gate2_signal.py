#!/usr/bin/env python3
"""Fase 2 -- offline signal gate: does cheap RD context beat the variance floor?

For each feature subset (variance / pixels24 / H9a / H9b / H9c) this trains a
per-block-size MLP on the 3-class collapsed label and scores held-out
superblocks, then replays the NONE-commit + rect-off pruning policy (the oracle
cost simulation) over a tau sweep. It reports, per subset, the search-COST
reduction achievable under matched RD-risk caps.

Decision (Gate 2): if H9a and/or H9b reach a materially higher cost reduction
than `variance` at the same risk, cheap RD context carries signal the pixels
lack -> proceed to the C integration. If not, stop: the ceiling is informational
and only the none_rdcost (H9c) study remains.

This is deliberately Python-only and model-light: it isolates the FEATURES, not
the surrogate/distillation, so the verdict is about information, not capacity.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from simulate_pruning import simulate, metrics, _subtree_size  # noqa: E402


def collect(entries, limit=None):
    """Superblocks with full 41-feature H9 vectors + truth per node."""
    sbs = []
    for e in entries:
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: no RD context; re-extract with the H9 "
                                 "instrumentation.".format(e["path"]))
            nodes = {}
            for k, (dim, r, c, luma, label) in enumerate(sb["members"]):
                ctx = dict(sb["ctx"][k])
                ctx["bsize_enum"] = -1
                nodes[(dim, r, c)] = {
                    "truth": label,
                    "feat": featmod.node_features_h9(sb["luma"], dim, r, c,
                                                     sb["qindex"], ctx),
                }
            sbs.append({"nodes": nodes})
            if limit and len(sbs) >= limit:
                return sbs
    return sbs


def make_mlp(n_in, hidden=(64, 32)):
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    layers += [nn.Linear(prev, 3)]
    return nn.Sequential(*layers)


def train_and_score(train_sbs, val_sbs, cols, device, epochs=25, lr=1e-3):
    """Train one MLP per block size on `cols` features; attach val probs."""
    cols = np.asarray(cols)
    for dim, _ in MODEL_LEVELS:
        X, y = [], []
        for sb in train_sbs:
            for (d, r, c), nd in sb["nodes"].items():
                if d == dim:
                    X.append(nd["feat"][cols])
                    y.append(studentmod.collapse_label(nd["truth"]))
        if not X:
            continue
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.long)
        mean, std = X.mean(0), X.std(0).clamp_min(1e-6)
        Xn = ((X - mean) / std).to(device)
        yb = y.to(device)
        net = make_mlp(len(cols)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        n = len(Xn)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, 8192):
                idx = perm[s:s + 8192]
                loss = F.cross_entropy(net(Xn[idx]), yb[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net.eval()
        # Score val nodes of this size.
        idx = [(si, key) for si, sb in enumerate(val_sbs)
               for key in sb["nodes"] if key[0] == dim]
        if not idx:
            continue
        feats = np.array([val_sbs[si]["nodes"][key]["feat"][cols]
                          for si, key in idx])
        with torch.no_grad():
            fb = (torch.tensor(feats, dtype=torch.float32).to(device) -
                  mean.to(device)) / std.to(device)
            probs = F.softmax(net(fb), -1).cpu().numpy()
        for (si, key), p in zip(idx, probs):
            val_sbs[si]["nodes"][key]["prob"] = p


def variance_threshold_probs(val_sbs, v0=1000.0):
    """Non-learned baseline: P(NONE)=exp(-var/v0) from feature 0 (log_var).
    Only the modeled levels (16/32/64) are scored; 8x8 leaves stay unscored so
    the simulator treats them as terminal, exactly like the learned models."""
    modeled = {d for d, _ in MODEL_LEVELS}
    for sb in val_sbs:
        for key, nd in sb["nodes"].items():
            if key[0] not in modeled:
                continue
            var = np.expm1(nd["feat"][0])       # feature 0 is log1p(var)
            flat = float(np.exp(-var / v0))
            nd["prob"] = np.array([flat, 1.0 - flat, 0.0])


def best_cost_at_risk(sbs, taus_none, tau_rest, max_split_lost, max_none_wrong):
    """Max cost reduction over a tau_none sweep meeting the risk caps."""
    best = 0.0
    best_pt = None
    for tn in taus_none:
        for tr in tau_rest:
            m = metrics(simulate(sbs, tn, 2.0, tr))  # tau_split=2: NONE+rect only
            if (m["split_lost"] <= max_split_lost and
                    m["none_wrong"] <= max_none_wrong):
                if m["cost_red"] > best:
                    best, best_pt = m["cost_red"], (tn, tr, m)
    return best, best_pt


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--val-seqs", nargs="+", default=["HoneyBee", "FlowerPan",
                                                     "Lips"])
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--subsets", nargs="+",
                   default=["variance", "pixels24", "H9a", "H9b", "H9c"])
    p.add_argument("--max-split-lost", type=float, default=1.0)
    p.add_argument("--max-none-wrong", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv",
                   default="/workspace/results/models/gate2_signal.csv")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs,
                                           args.train_seqs)
    datamod.assert_real_luma(train_e)
    print("train pkls: {}, val pkls: {} (val {})".format(
        len(train_e), len(val_e), args.val_seqs), flush=True)
    train_sbs = collect(train_e, limit=args.limit)
    val_base = collect(val_e, limit=args.limit)
    print("train superblocks: {}, val: {}".format(len(train_sbs), len(val_base)),
          flush=True)

    taus_none = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    tau_rest = [-1.0, 0.1, 0.2, 0.3]
    rows = []
    print("\nsubset      cost_red%  @(tau_none,tau_rest)  none_wrong% splitLost%",
          flush=True)

    # Non-learned variance-threshold reference.
    import copy
    vt = copy.deepcopy(val_base)
    variance_threshold_probs(vt)
    b, pt = best_cost_at_risk(vt, taus_none, tau_rest, args.max_split_lost,
                              args.max_none_wrong)
    print("var-thresh   {:6.2f}    tau={:.2f}/{:.2f}       {:6.2f}    {:6.2f}"
          .format(b, pt[0], pt[1], pt[2]["none_wrong"], pt[2]["split_lost"])
          if pt else "var-thresh    (no feasible point)", flush=True)
    rows.append(["var-threshold", round(b, 3), pt[0] if pt else None,
                 pt[1] if pt else None])

    for name in args.subsets:
        cols = featmod.H9_SUBSETS[name]
        val_sbs = copy.deepcopy(val_base)
        train_and_score(train_sbs, val_sbs, cols, device)
        b, pt = best_cost_at_risk(val_sbs, taus_none, tau_rest,
                                  args.max_split_lost, args.max_none_wrong)
        if pt:
            print("{:<11} {:6.2f}    tau={:.2f}/{:.2f}       {:6.2f}    {:6.2f}"
                  .format(name, b, pt[0], pt[1], pt[2]["none_wrong"],
                          pt[2]["split_lost"]), flush=True)
        else:
            print("{:<11}  (no feasible point)".format(name), flush=True)
        rows.append([name, round(b, 3), pt[0] if pt else None,
                     pt[1] if pt else None])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "cost_reduction_pct", "tau_none", "tau_rest"])
        w.writerows(rows)
    print("\nwrote", args.out_csv)
    print("\nGATE 2 verdict: compare H9a/H9b cost_red% to var-threshold and "
          "pixels24. A clear margin => cheap RD context beats the pixel floor.")


if __name__ == "__main__":
    main(sys.argv[1:])
