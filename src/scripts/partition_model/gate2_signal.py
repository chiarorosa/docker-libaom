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


def train_and_score(train_sbs, val_sbs, cols, device, epochs=30, lr=1e-3,
                    seeds=3):
    """Train one MLP per block size on `cols` features; attach val probs.
    Averages `seeds` independent runs (a cheap ensemble) so the reported
    operating points reflect the features, not a single noisy fit."""
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
        n = len(Xn)
        idx = [(si, key) for si, sb in enumerate(val_sbs)
               for key in sb["nodes"] if key[0] == dim]
        if not idx:
            continue
        feats = np.array([val_sbs[si]["nodes"][key]["feat"][cols]
                          for si, key in idx])
        fb = (torch.tensor(feats, dtype=torch.float32).to(device) -
              mean.to(device)) / std.to(device)
        prob_sum = np.zeros((len(idx), 3))
        for seed in range(seeds):
            torch.manual_seed(1000 + seed)
            net = make_mlp(len(cols)).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(epochs):
                perm = torch.randperm(n, device=device)
                for s in range(0, n, 8192):
                    b = perm[s:s + 8192]
                    loss = F.cross_entropy(net(Xn[b]), yb[b])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                prob_sum += F.softmax(net(fb), -1).cpu().numpy()
        probs = prob_sum / seeds
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


def sweep_curve(sbs, taus_none, tau_rest):
    """All (cost_red, risk) points over the tau grid; NONE+rect policy."""
    pts = []
    for tn in taus_none:
        for tr in tau_rest:
            m = metrics(simulate(sbs, tn, 2.0, tr))
            pts.append((tn, tr, m["cost_red"], m["none_wrong"],
                        m["split_lost"], m["rect_off_wrong"]))
    return pts


def cost_at_risk(pts, max_split_lost, max_rect_off_wrong=5.0):
    """Max cost reduction among sweep points within the BD-relevant risk caps:
    true-SPLIT loss (catastrophic, from NONE-commit) and rect-off error (from
    disabling rectangular candidates). none_wrong is intentionally not bounded --
    it is dominated by cheap NONE/rect confusions that barely move BD-rate."""
    ok = [c for _, _, c, _nw, sl, rw in pts
          if sl <= max_split_lost and rw <= max_rect_off_wrong]
    return max(ok) if ok else 0.0


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
    # Matched-risk operating points: cost reduction achievable while keeping
    # true-SPLIT loss under each cap (none_wrong loosely bounded).
    caps = [0.5, 1.0, 2.0, 3.0]
    import copy

    curves = {}
    vt = copy.deepcopy(val_base)
    variance_threshold_probs(vt)
    curves["var-thresh"] = sweep_curve(vt, taus_none, tau_rest)
    for name in args.subsets:
        cols = featmod.H9_SUBSETS[name]
        val_sbs = copy.deepcopy(val_base)
        train_and_score(train_sbs, val_sbs, cols, device)
        curves[name] = sweep_curve(val_sbs, taus_none, tau_rest)

    # Report: cost reduction (%) at matched risk (SPLIT-lost cap, rect-off
    # error <= 5%).
    print("\ncost_red% at matched risk (SPLIT-lost cap; rect_off_wrong<=5%):",
          flush=True)
    hdr = "subset       " + "".join("  <=SL{:.1f}%".format(c) for c in caps)
    print(hdr, flush=True)
    order = ["var-thresh"] + list(args.subsets)
    rows = []
    for name in order:
        pts = curves[name]
        cells = [cost_at_risk(pts, c, 5.0) for c in caps]
        print("{:<11}".format(name) +
              "".join("  {:8.2f}".format(v) for v in cells), flush=True)
        rows.append([name] + [round(v, 3) for v in cells])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset"] + ["cost_red_SL{}".format(c) for c in caps])
        w.writerows(rows)
    # Full sweep dump for inspection.
    with open(args.out_csv.replace(".csv", "_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "tau_none", "tau_rest", "cost_red", "none_wrong",
                    "split_lost", "rect_off_wrong"])
        for name in order:
            for row in curves[name]:
                w.writerow([name] + [round(x, 3) for x in row])
    print("\nwrote", args.out_csv)
    print("\nGATE 2: H9a/H9b cost_red% clearly above var-thresh & pixels24 at "
          "matched SPLIT-lost => cheap RD context beats the pixel floor.")


if __name__ == "__main__":
    main(sys.argv[1:])
