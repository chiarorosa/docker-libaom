#!/usr/bin/env python3
"""Oracle simulation of the partition-pruning policy -- the Fase C gate.

Before touching any C, replay the pruning decisions over held-out ground truth
to estimate, per (tau_none, tau_split, tau_rest) operating point:
  * node reduction: fraction of the baseline's logged partition nodes whose RD
    search would be eliminated (a NONE-commit cuts the whole subtree) -- kept
    for comparability with the H1-H6 numbers;
  * COST reduction (v2, the time proxy that picks the operating point): each
    node's full search evaluates ~9 shape candidates (NONE + rect + AB + 4-way;
    4 at 8x8), each costing ~n^2 pixels. The three actions map to costs:
      NONE-commit  -> the node pays NONE only (1 candidate), subtree is cut;
      force-SPLIT  -> the node pays no shape candidate (recursion only);
      rect-off     -> the node pays NONE only, but children are still searched.
    Node counting is blind to the dominant rect/AB/4-way term (~8/9 of the
    within-node work); cost counting is what correlates with encode time.
  * RD risk: how often the policy forces a decision that contradicts the true
    RD choice -- especially true-SPLIT nodes wrongly committed to NONE, and
    true-rect nodes whose rect candidates were disabled.

Node probabilities can come from either model, so this doubles as a diagnostic:
  --students  : score with the distilled per-size MLPs (what ships in C).
  --surrogate : score with the multi-level ConvNeXt directly (the ceiling; if it
                is much better, the handcrafted features are the bottleneck).

The pruning policy mirrors Fase D exactly (per-level thresholds, Pre-H7 A3):
  P(NONE)  > tau_none[dim]  -> commit NONE (av1_disable_all_splits).
  P(SPLIT) > tau_split[dim] -> force split (av1_set_square_split_only).
  P(REST)  < tau_rest[dim]  -> disable rect/AB/4-way (av1_disable_rect_partitions)
                               while still searching NONE vs SPLIT (Pre-H7 A1).
  otherwise                 -> full search.
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
import torch.nn.functional as F  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402

CHILD_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]

# Shape candidates a full search evaluates at a node (cpu-used=0 All-Intra):
# NONE + HORZ/VERT + 4 AB + 2 four-way = 9 for >=16px; 8x8 has no AB/4-way and
# sub-8x8 is never explored in this regime -> NONE/HORZ/VERT + SPLIT bookkeeping
# ~ 4. Each candidate covers the block's n^2 pixels, hence the dim^2 weight.
CANDS = {64: 9, 32: 9, 16: 9, 8: 4}


def node_cost(dim):
    return CANDS[dim] * dim * dim


def collect_superblocks(entries, feature_set="pixels24", limit=None):
    """List of superblocks: {nodes:{(dim,r,c):{truth,feat}}, luma, qindex}.
    feature_set: 'pixels24' -> node_features (24); 'h9a' -> node_features_h9[:36]
    with the per-node RD context (blocks B/C)."""
    sbs = []
    for e in entries:
        for sb in datamod.iter_superblock_members(e["path"]):
            if feature_set == "h9a" and not sb.get("has_rd"):
                raise SystemExit("{}: no RD context; re-extract with the H9 "
                                 "instrumentation.".format(e["path"]))
            nodes = {}
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if feature_set == "h9a":
                    feat = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                                      sb["qindex"], sb["ctx"][k])
                else:
                    feat = featmod.node_features(sb["luma"], dim, r, c,
                                                 sb["qindex"])
                nodes[(dim, r, c)] = {"truth": label, "feat": feat}
            sbs.append({"nodes": nodes, "luma": sb["luma"],
                        "qindex": sb["qindex"]})
            if limit and len(sbs) >= limit:
                return sbs
    return sbs


def score_with_student(sbs, bundle, device):
    """Attach 3-class probs from the distilled per-size MLPs (once per node)."""
    nets = {}
    nfeat = bundle.get("num_features", featmod.NUM_FEATURES)
    for dim, _ in MODEL_LEVELS:
        if dim in bundle["students"]:
            net = studentmod.make_student(nfeat, bundle["hidden"])
            net.load_state_dict(bundle["students"][dim])
            nets[dim] = net.to(device).eval()
    for dim, _ in MODEL_LEVELS:
        if dim not in nets:
            continue
        idx = [(si, key) for si, sb in enumerate(sbs) for key in sb["nodes"]
               if key[0] == dim]
        if not idx:
            continue
        feats = np.stack([sbs[si]["nodes"][key]["feat"] for si, key in idx])
        with torch.no_grad():
            probs = F.softmax(nets[dim](torch.tensor(
                feats, dtype=torch.float32, device=device)), dim=-1).cpu().numpy()
        for (si, key), p in zip(idx, probs):
            sbs[si]["nodes"][key]["prob"] = p
    return sbs


def score_with_variance(sbs, v0=1000.0):
    """Non-learned baseline: P(NONE)=exp(-var/v0) from feature 0 (log_var).
    Only modeled levels are scored; 8x8 leaves stay terminal, as the models do."""
    modeled = {d for d, _ in MODEL_LEVELS}
    for sb in sbs:
        for key, nd in sb["nodes"].items():
            if key[0] not in modeled:
                continue
            var = float(np.expm1(nd["feat"][0]))     # feature 0 is log1p(var)
            flat = float(np.exp(-var / v0))
            nd["prob"] = np.array([flat, 1.0 - flat, 0.0])
    return sbs


def classification_report(sbs):
    """Per-size macro-F1 and SPLIT-recall from the scored probs (argmax) vs the
    3-class collapsed truth (Gate 3a). truth is the raw 10-class PARTITION_TYPE;
    collapse_label maps it to [NONE, SPLIT, REST]."""
    per = {d: {"pred": [], "true": []} for d, _ in MODEL_LEVELS}
    for sb in sbs:
        for (dim, r, c), nd in sb["nodes"].items():
            if dim not in per or "prob" not in nd:
                continue
            per[dim]["pred"].append(int(np.argmax(nd["prob"])))
            per[dim]["true"].append(studentmod.collapse_label(nd["truth"]))
    print("\nper-size classification (Gate 3a): macro-F1 | SPLIT-recall")
    for dim, _ in MODEL_LEVELS:
        pred = np.array(per[dim]["pred"])
        true = np.array(per[dim]["true"])
        if len(true) == 0:
            continue
        f1s = []
        for cls in range(3):
            tp = int(((pred == cls) & (true == cls)).sum())
            fp = int(((pred == cls) & (true != cls)).sum())
            fn = int(((pred != cls) & (true == cls)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        split_tp = int(((pred == 1) & (true == 1)).sum())
        split_fn = int(((pred != 1) & (true == 1)).sum())
        srec = split_tp / (split_tp + split_fn) if split_tp + split_fn else 0.0
        print("  {:>2}px  macroF1={:.3f}  SPLITrecall={:.3f}".format(
            dim, float(np.mean(f1s)), srec))


def score_with_surrogate(sbs, surrogate, device, batch=256):
    """Attach 3-class probs from the multi-level ConvNeXt (per-node, collapsed)."""
    buf_idx, buf_x = [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to(device)
        with torch.no_grad():
            logits = surrogate(x)
            probs = {dim: F.softmax(logits[dim].float(), dim=-1).cpu().numpy()
                     for dim, _ in MODEL_LEVELS}
        for k, si in enumerate(buf_idx):
            for (dim, r, c) in sbs[si]["nodes"]:
                if dim not in probs:
                    continue  # 8x8 leaves are not modeled
                sbs[si]["nodes"][(dim, r, c)]["prob"] = studentmod.collapse_probs(
                    probs[dim][k, r, c])
        buf_idx.clear()
        buf_x.clear()

    for si, sb in enumerate(sbs):
        luma = sb["luma"].astype(np.float32) / 255.0
        qplane = np.full((64, 64), sb["qindex"] / 255.0, dtype=np.float32)
        buf_x.append(np.stack([luma, qplane]))
        buf_idx.append(si)
        if len(buf_x) >= batch:
            flush()
    flush()
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


def _as_level_map(tau):
    """Accept a scalar or a {dim: value} map; return a {64,32,16} map."""
    if isinstance(tau, dict):
        return tau
    return {dim: tau for dim, _ in MODEL_LEVELS}


def simulate(sbs, tau_none, tau_split, tau_rest=-1.0):
    """Replay the 3-action policy. tau_* may be scalars or per-level maps.
    tau_rest < 0 disables the rect-off action (v1-compatible)."""
    tn, ts, tr = (_as_level_map(t) for t in (tau_none, tau_split, tau_rest))
    stat = {"evaluated": 0, "saved": 0, "none_forced": 0, "none_forced_wrong": 0,
            "split_forced": 0, "split_forced_wrong": 0, "true_split": 0,
            "true_split_cut": 0, "true_rect": 0, "true_rect_cut": 0,
            "scored": 0, "rect_off": 0, "rect_off_wrong": 0,
            "base_cost": 0, "cost": 0}

    def visit(nodes, key):
        node = nodes.get(key)
        if node is None:
            return
        stat["evaluated"] += 1
        truth = node["truth"]
        prob = node.get("prob")
        dim = key[0]
        present = [(dim // 2, 2 * key[1] + dr, 2 * key[2] + dc)
                   for dr, dc in CHILD_OFFSETS] if dim > 8 else []
        present = [c for c in present if c in nodes]
        if prob is None:  # 8x8 leaf: never pruned, full (small) search
            stat["cost"] += node_cost(dim)
            for c in present:
                visit(nodes, c)
            return
        stat["scored"] += 1
        p_none, p_split, p_rest = (float(prob[0]), float(prob[1]),
                                   float(prob[2]))
        if p_none > tn[dim]:
            # av1_disable_all_splits: the node pays NONE only, subtree cut.
            stat["cost"] += dim * dim
            stat["saved"] += sum(1 + _subtree_size(nodes, c) for c in present)
            stat["none_forced"] += 1
            if truth != 0:
                stat["none_forced_wrong"] += 1
            if truth == 3:
                stat["true_split"] += 1
                stat["true_split_cut"] += 1
            elif truth not in (0, 3):
                stat["true_rect"] += 1
                stat["true_rect_cut"] += 1
            return
        if truth == 3:
            stat["true_split"] += 1
        elif truth not in (0, 3):
            stat["true_rect"] += 1
        if p_split > ts[dim]:
            # av1_set_square_split_only: no shape candidate at this node.
            stat["split_forced"] += 1
            if truth != 3:
                stat["split_forced_wrong"] += 1
        elif p_rest < tr[dim]:
            # av1_disable_rect_partitions: NONE still searched, rect family off.
            stat["cost"] += dim * dim
            stat["rect_off"] += 1
            if truth not in (0, 3):
                stat["rect_off_wrong"] += 1
        else:
            stat["cost"] += node_cost(dim)
        for c in present:
            visit(nodes, c)

    def baseline(nodes, key):
        node = nodes.get(key)
        if node is None:
            return
        stat["base_cost"] += node_cost(key[0])
        if key[0] > 8:
            for dr, dc in CHILD_OFFSETS:
                baseline(nodes, (key[0] // 2, 2 * key[1] + dr, 2 * key[2] + dc))

    for sb in sbs:
        baseline(sb["nodes"], (64, 0, 0))
        visit(sb["nodes"], (64, 0, 0))
    return stat


def metrics(s):
    """Derived percentages from a simulate() stat dict."""
    node_red = 100.0 * s["saved"] / max(s["saved"] + s["evaluated"], 1)
    cost_red = 100.0 * (s["base_cost"] - s["cost"]) / max(s["base_cost"], 1)
    return {
        "node_red": node_red,
        "cost_red": cost_red,
        "none_wrong": 100.0 * s["none_forced_wrong"] / max(s["none_forced"], 1),
        "split_lost": 100.0 * s["true_split_cut"] / max(s["true_split"], 1),
        "rect_lost": 100.0 * s["true_rect_cut"] / max(s["true_rect"], 1),
        "split_forced_wrong":
            100.0 * s["split_forced_wrong"] / max(s["split_forced"], 1),
        "rect_off_cov": 100.0 * s["rect_off"] / max(s["scored"], 1),
        "rect_off_wrong": 100.0 * s["rect_off_wrong"] / max(s["rect_off"], 1),
    }


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset")
    p.add_argument("--students", default="/workspace/results/models/student/"
                                        "students.pt")
    p.add_argument("--surrogate", default=None,
                   help="if set, score nodes with this ConvNeXt ckpt instead of "
                        "the distilled students (ceiling diagnostic)")
    p.add_argument("--val-seqs", nargs="+",
                   default=["HoneyBee", "FlowerPan", "Lips"])
    p.add_argument("--tau-none", type=float, nargs="+",
                   default=[0.6, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--tau-split", type=float, nargs="+", default=[0.7, 0.9])
    p.add_argument("--tau-rest", type=float, nargs="+", default=[-1.0],
                   help="P(REST) thresholds for the rect-off action; "
                        "-1 disables it (v1-compatible)")
    p.add_argument("--refine", action="store_true",
                   help="after the grid sweep, per-level coordinate descent "
                        "maximizing cost reduction under the risk caps")
    p.add_argument("--max-split-lost", type=float, default=1.0,
                   help="refine cap: %% of true-SPLIT nodes cut")
    p.add_argument("--max-none-wrong", type=float, default=8.0,
                   help="refine cap: %% of NONE-commits that contradict truth")
    p.add_argument("--max-rect-off-wrong", type=float, default=8.0,
                   help="refine cap: %% of rect-off actions at true-rect nodes")
    p.add_argument("--out-csv", default="/workspace/results/models/student/"
                                        "oracle_sim.csv")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--baseline", choices=["variance"], default=None,
                   help="score with the non-learned variance threshold instead "
                        "of a model (Gate 3 comparison)")
    p.add_argument("--v0", type=float, default=1000.0,
                   help="variance-baseline scale: P(NONE)=exp(-var/v0)")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, val_e = datamod.split_entries(entries, args.val_seqs, None)
    datamod.assert_real_luma(val_e)

    if args.baseline == "variance":
        feature_set, mode = "pixels24", "variance"
    elif args.surrogate:
        feature_set, mode = "pixels24", "surrogate"
    else:
        bundle = torch.load(args.students, map_location=device)
        feature_set, mode = bundle.get("feature_set", "pixels24"), "student"

    sbs = collect_superblocks(val_e, feature_set=feature_set, limit=args.limit)
    total_nodes = sum(len(s["nodes"]) for s in sbs)

    if mode == "surrogate":
        from model import PartitionSurrogate
        ckpt = torch.load(args.surrogate, map_location=device)
        sa = ckpt.get("args", {})
        net = PartitionSurrogate(sa.get("variant", "tiny"),
                                 sa.get("fusion_dim", 128)).to(device).eval()
        net.load_state_dict(ckpt["model"])
        sbs = score_with_surrogate(sbs, net, device)
        model_tag = "SURROGATE " + args.surrogate
    elif mode == "variance":
        sbs = score_with_variance(sbs, args.v0)
        model_tag = "VARIANCE-BASELINE v0={}".format(args.v0)
    else:
        sbs = score_with_student(sbs, bundle, device)
        model_tag = "STUDENT {} (feature_set={})".format(args.students,
                                                          feature_set)
    print("scored with:", model_tag)
    print("superblocks: {}, nodes: {} (val seqs {})".format(
        len(sbs), total_nodes, args.val_seqs))
    classification_report(sbs)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    hdr = ("tau_none tau_split tau_rest | node_red% cost_red% | none_wrong% "
           "splitLost% rectLost% | rectOffCov% rectOffWrong%")

    def report(tn, ts, tr, label=""):
        m = metrics(simulate(sbs, tn, ts, tr))
        fmt_tau = lambda t: ("{:.2f}".format(t) if not isinstance(t, dict) else
                             "/".join("{:.2f}".format(t[d])
                                      for d, _ in MODEL_LEVELS))
        rows.append((fmt_tau(tn), fmt_tau(ts), fmt_tau(tr),
                     m["node_red"], m["cost_red"], m["none_wrong"],
                     m["split_lost"], m["rect_lost"], m["rect_off_cov"],
                     m["rect_off_wrong"], label))
        print("  {:>14} {:>14} {:>14} | {:7.2f} {:8.2f} | {:6.2f} {:6.2f} "
              "{:6.2f} | {:6.2f} {:6.2f}  {}".format(
                  fmt_tau(tn), fmt_tau(ts), fmt_tau(tr), m["node_red"],
                  m["cost_red"], m["none_wrong"], m["split_lost"],
                  m["rect_lost"], m["rect_off_cov"], m["rect_off_wrong"],
                  label))
        return m

    print(hdr)
    for tn in args.tau_none:
        for ts in args.tau_split:
            for tr in args.tau_rest:
                report(tn, ts, tr)

    if args.refine:
        def feasible(m):
            return (m["split_lost"] <= args.max_split_lost and
                    m["none_wrong"] <= args.max_none_wrong and
                    m["rect_off_wrong"] <= args.max_rect_off_wrong)

        # Start from the most conservative point; per-level coordinate descent
        # maximizing cost reduction subject to the risk caps.
        cur = {"tn": _as_level_map(max(args.tau_none)),
               "ts": _as_level_map(max(args.tau_split)),
               "tr": _as_level_map(min(args.tau_rest))}
        best = metrics(simulate(sbs, cur["tn"], cur["ts"], cur["tr"]))
        cand = {"tn": args.tau_none, "ts": args.tau_split, "tr": args.tau_rest}
        improved = True
        while improved:
            improved = False
            for knob in ("tn", "ts", "tr"):
                for dim, _ in MODEL_LEVELS:
                    for v in cand[knob]:
                        if v == cur[knob][dim]:
                            continue
                        trial = {k: dict(m) for k, m in cur.items()}
                        trial[knob][dim] = v
                        m = metrics(simulate(sbs, trial["tn"], trial["ts"],
                                             trial["tr"]))
                        if feasible(m) and m["cost_red"] > best["cost_red"]:
                            cur, best, improved = trial, m, True
        print("\nrefined per-level operating point "
              "(caps: splitLost<={}, noneWrong<={}, rectOffWrong<={}):".format(
                  args.max_split_lost, args.max_none_wrong,
                  args.max_rect_off_wrong))
        print(hdr)
        report(cur["tn"], cur["ts"], cur["tr"], label="REFINED")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_none", "tau_split", "tau_rest", "node_reduction_pct",
                    "cost_reduction_pct", "none_forced_wrong_pct",
                    "true_split_lost_pct", "true_rect_lost_pct",
                    "rect_off_coverage_pct", "rect_off_wrong_pct", "label"])
        w.writerows(rows)

    caps = [0.5, 1.0, 2.0]
    print("\ncost_red% at matched risk (split-lost cap):")
    print("  cap%   cost_red%")
    for cap in caps:
        feas = [r[4] for r in rows if r[6] <= cap]  # r[4]=cost_red, r[6]=split_lost
        print("  {:<5} {:8.2f}".format(cap, max(feas) if feas else 0.0))

    print("wrote", args.out_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
