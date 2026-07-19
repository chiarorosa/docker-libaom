#!/usr/bin/env python3
"""Calibration of the DEPLOYED student's softmax on the held-out test split.

Item A4 of the action plan: the project has no calibration analysis, yet the
thesis operating point IS a threshold on this softmax -- the C pruner commits
NONE when P(NONE) > tau_none and forces split when P(SPLIT) > tau_split
(partition_strategy.c:2142-2145). Everywhere "tau=0.95" is read as "95%
confidence" without that having been measured.

This runs the deployed weights (results/models/student_h9a/students.pt, the same
bundle export_weights.py dumps to C) over the test .pkl, exactly reproducing the
C inference path: node_features_h9a -> the net (which has normalization folded
into its first layer, distill.py:139-144, so it consumes RAW features like the
encoder) -> softmax. It then measures, against the true collapsed label:

  * top-label ECE and a reliability diagram (standard calibration summary);
  * per-class one-vs-rest reliability for NONE and SPLIT (the two acted-upon
    classes);
  * threshold precision: among nodes with P(class) > tau, the empirical fraction
    that truly are that class -- this is the number "tau confidence" claims to be.

All offline, over encodes already on disk; nothing is re-encoded.

Usage (container):
  python calibration.py --out-dir results/models/student_h9a/calibration
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

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402

CLASS_NAMES = ["NONE", "SPLIT", "REST"]
TEST_SEQS = ["Jockey", "RaceNight", "RiverBank"]
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def load_students(path, device):
    """Rebuild each per-block-size deployed net from the bundle. The net already
    has normalization folded into its first layer, so it consumes RAW features."""
    bundle = torch.load(path, map_location=device, weights_only=False)
    hidden = bundle["hidden"]
    nin = bundle.get("num_features", featmod.NUM_FEATURES_H9A)
    nets = {}
    for dim, sd in bundle["students"].items():
        net = studentmod.make_student(nin, hidden).to(device)
        net.load_state_dict(sd)
        net.eval()
        nets[int(dim)] = net
    return nets, nin


def collect_probs(entries, nets, device, per_pkl):
    """Return per-dim {'p':(N,3) softmax, 'y':(N,) collapsed label}."""
    modeled = {d for d, _ in MODEL_LEVELS}
    feats = {d: [] for d in modeled}
    ys = {d: [] for d in modeled}
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: no RD context".format(e["path"]))
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in modeled:
                    continue
                feats[dim].append(featmod.node_features_h9a(
                    sb["luma"], dim, r, c, sb["qindex"], sb["ctx"][k]))
                ys[dim].append(studentmod.collapse_label(label))
            took += 1
            if per_pkl and took >= per_pkl:
                break
        print("  {} done ({} SB cap={})".format(
            os.path.basename(e["path"]), took, per_pkl), flush=True)
    out = {}
    for dim in modeled:
        if not feats[dim] or dim not in nets:
            continue
        X = torch.tensor(np.asarray(feats[dim], dtype=np.float32), device=device)
        with torch.no_grad():
            p = torch.softmax(nets[dim](X), -1).cpu().numpy()
        out[dim] = {"p": p, "y": np.asarray(ys[dim], dtype=np.int64)}
    return out


def top_label_ece(p, y, n_bins=10):
    """Standard top-label ECE + per-bin (conf, acc, count) for a reliability plot."""
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt:
            c_conf = float(conf[m].mean())
            c_acc = float(correct[m].mean())
            ece += cnt / n * abs(c_acc - c_conf)
        else:
            c_conf = c_acc = float("nan")
        bins.append({"lo": lo, "hi": hi, "count": cnt, "conf": c_conf,
                     "acc": c_acc})
    return ece, bins


def class_reliability(p, y, cls, n_bins=10):
    """One-vs-rest reliability for `cls`: per bin of P(cls), the empirical
    frequency of the true class being `cls`. ECE for this class as well."""
    pc = p[:, cls]
    is_cls = (y == cls).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, n = [], 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (pc > lo) & (pc <= hi) if i else (pc >= lo) & (pc <= hi)
        cnt = int(m.sum())
        if cnt:
            c_conf = float(pc[m].mean())
            c_freq = float(is_cls[m].mean())
            ece += cnt / n * abs(c_freq - c_conf)
        else:
            c_conf = c_freq = float("nan")
        bins.append({"lo": lo, "hi": hi, "count": cnt, "conf": c_conf,
                     "freq": c_freq})
    return ece, bins


def threshold_precision(p, y, cls, taus):
    """Among nodes with P(cls) > tau, the empirical fraction truly == cls, and
    the coverage (share of all nodes past the threshold). This is what
    'tau confidence' asserts and the C pruner actually relies on."""
    pc = p[:, cls]
    is_cls = (y == cls)
    n = len(y)
    rows = []
    for t in taus:
        m = pc > t
        cov = int(m.sum())
        prec = float(is_cls[m].mean()) if cov else float("nan")
        rows.append({"tau": t, "precision": prec, "coverage": cov,
                     "coverage_pct": 100.0 * cov / n})
    return rows


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    ap.add_argument("--students",
                    default="/workspace/results/models/student_h9a/students.pt")
    ap.add_argument("--test-seqs", nargs="+", default=TEST_SEQS)
    ap.add_argument("--per-pkl", type=int, default=0,
                    help="cap superblocks per pkl (0 = all)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out-dir",
                    default="/workspace/results/models/student_h9a/calibration")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets, nin = load_students(args.students, device)
    print("loaded deployed student: {} feats, dims {}".format(
        nin, sorted(nets)), flush=True)

    entries = datamod.discover_pkls(args.dataset_dir)
    _, test_e = datamod.split_entries(entries, args.test_seqs, train_seqs=[])
    datamod.assert_real_luma(test_e)
    print("test pkls: {} (seqs {})".format(len(test_e), args.test_seqs),
          flush=True)
    per = args.per_pkl or None
    bydim = collect_probs(test_e, nets, device, per)

    # Pooled over all modeled dims, plus per-dim.
    allp = np.concatenate([bydim[d]["p"] for d in bydim])
    ally = np.concatenate([bydim[d]["y"] for d in bydim])
    groups = [("all", allp, ally)] + [
        ("dim{}".format(d), bydim[d]["p"], bydim[d]["y"]) for d in sorted(bydim)]

    rel_rows, thr_rows, ece_rows = [], [], []
    md = ["# Calibração da softmax do estudante implantado (A4)",
          "",
          "**Data:** 2026-07-19  ",
          "**Split de teste:** {} ({} nós de decisão)".format(
              ", ".join(args.test_seqs), len(ally)),
          "**Modelo:** `results/models/student_h9a/students.pt` (pesos implantados"
          " em C)  ",
          "**Reprodução:** `python src/scripts/partition_model/calibration.py`",
          ""]

    md += ["## 1. ECE (top-label) e por classe", "",
           "ECE = erro esperado de calibração; 0 = perfeitamente calibrado. "
           "Por classe é one-vs-rest.", "",
           "| grupo | n | ECE top | ECE NONE | ECE SPLIT | ECE REST |",
           "|---|--:|--:|--:|--:|--:|"]
    for name, p, y in groups:
        e_top, _ = top_label_ece(p, y, args.bins)
        e_cls = []
        for cls in range(3):
            ec, _ = class_reliability(p, y, cls, args.bins)
            e_cls.append(ec)
        md.append("| {} | {:,} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, len(y), e_top, e_cls[0], e_cls[1], e_cls[2]))
        ece_rows.append({"group": name, "n": len(y), "ece_top": e_top,
                         "ece_none": e_cls[0], "ece_split": e_cls[1],
                         "ece_rest": e_cls[2]})

    # Threshold precision for NONE and SPLIT -- the acted-upon decisions.
    md += ["", "## 2. Precisão no limiar — o que \"τ de confiança\" realmente vale",
           "",
           "Entre os nós com P(classe) > τ, a fração que **de fato** é dessa "
           "classe (precisão), e a cobertura (fração dos nós além do limiar). "
           "Se a precisão em τ=0,95 for < 0,95, o limiar não é 95% de confiança."]
    for cls, cname in [(0, "NONE"), (1, "SPLIT")]:
        md += ["", "### 2.{} P({}) > τ  (pooled, todos os níveis)".format(
            cls + 1, cname), "",
            "| τ | precisão | cobertura | cobertura % |",
            "|--:|--:|--:|--:|"]
        for r in threshold_precision(allp, ally, cls, TAU_GRID):
            md.append("| {:.2f} | {} | {:,} | {:.2f}% |".format(
                r["tau"],
                "{:.4f}".format(r["precision"]) if r["precision"] == r["precision"]
                else "--", r["coverage"], r["coverage_pct"]))
            thr_rows.append({"group": "all", "cls": cname, **r})
        # per-dim at the two deployment taus
        for d in sorted(bydim):
            for r in threshold_precision(bydim[d]["p"], bydim[d]["y"], cls,
                                         TAU_GRID):
                thr_rows.append({"group": "dim{}".format(d), "cls": cname, **r})

    md += ["", "## 3. Diagrama de confiabilidade (one-vs-rest)", "",
           "Por faixa de P(classe): probabilidade média predita (`conf`) contra "
           "frequência empírica da classe (`freq`). Calibrado ⇔ conf ≈ freq.",
           ""]
    for cls, cname in [(0, "NONE"), (1, "SPLIT")]:
        _, bins = class_reliability(allp, ally, cls, args.bins)
        md += ["### 3.{} classe {}".format(cls + 1, cname), "",
               "| faixa P | n | conf | freq |", "|---|--:|--:|--:|"]
        for b in bins:
            md.append("| {:.1f}–{:.1f} | {:,} | {} | {} |".format(
                b["lo"], b["hi"], b["count"],
                "{:.3f}".format(b["conf"]) if b["conf"] == b["conf"] else "--",
                "{:.3f}".format(b["freq"]) if b["freq"] == b["freq"] else "--"))
            rel_rows.append({"cls": cname, **b})
        md.append("")

    with open(os.path.join(args.out_dir, "calibration_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    for fname, rows, keys in [
        ("ece.csv", ece_rows,
         ["group", "n", "ece_top", "ece_none", "ece_split", "ece_rest"]),
        ("threshold_precision.csv", thr_rows,
         ["group", "cls", "tau", "precision", "coverage", "coverage_pct"]),
        ("reliability.csv", rel_rows,
         ["cls", "lo", "hi", "count", "conf", "freq"])]:
        with open(os.path.join(args.out_dir, fname), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(r[k], 5) if isinstance(r.get(k), float)
                                else r.get(k, "")) for k in keys})

    print("\n".join(md[7:]))
    print("\nWrote calibration_report.md + 3 CSVs to " + args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
