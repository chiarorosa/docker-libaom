#!/usr/bin/env python3
"""Gate 1-B (Approach B): ablação controlada L=0 (MLP) vs L>=1 (GNN) na validação.
Para cada bundle, prediz as classes por nó no conjunto de validação e reporta, por
tamanho, acurácia, macro-F1 e SPLIT-recall. Se L>=1 não superar L=0 (nem no oráculo,
ver simulate_pruning --gnn-bundle), a estrutura não agrega -> teto informacional."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import data as datamod  # noqa: E402
import graph_data as gd  # noqa: E402
import gnn_model as gm  # noqa: E402

VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def _macro_f1_and_split_recall(y_true, y_pred, n_classes=3):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    # SPLIT é a classe 1 (student.collapse_label).
    tp1 = np.sum((y_pred == 1) & (y_true == 1))
    fn1 = np.sum((y_pred != 1) & (y_true == 1))
    split_recall = tp1 / (tp1 + fn1) if (tp1 + fn1) else 0.0
    return float(np.mean(f1s)), float(split_recall)


def _predict(bundle, graphs, device):
    net = gm.build_gnn(bundle["num_features"], bundle["hidden"],
                       bundle["layer"], bundle["n_layers"])
    net.load_state_dict(bundle["state_dict"])
    net = net.to(device).eval()
    yt, yp, lv = [], [], []
    with torch.no_grad():
        for g in graphs:
            x = torch.tensor(g["x"], device=device)
            ei = torch.tensor(g["edge_index"], device=device)
            pred = F.softmax(net(x, ei), -1).argmax(-1).cpu().numpy()
            yt.append(g["y"]); yp.append(pred); lv.append(g["level"])
    return (np.concatenate(yt), np.concatenate(yp), np.concatenate(lv))


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--bundles", nargs="+", required=True,
                   help="pares tag=caminho, ex.: k0=.../gnn_k0/gnn.pt k2=.../gnn_k2/gnn.pt")
    p.add_argument("--per-pkl", type=int, default=2000)
    p.add_argument("--out", default="/workspace/results/models/gnn_gate1.csv")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, val_e = datamod.split_entries(entries, VAL_SEQS, None)
    datamod.assert_real_luma(val_e)
    graphs = gd.build_graph_dataset(val_e, per_pkl=args.per_pkl or None)
    print("grafos de validação:", len(graphs), flush=True)

    rows = ["tag,dim,n,acc,macroF1,split_recall"]
    for spec in args.bundles:
        tag, path = spec.split("=", 1)
        bundle = torch.load(path, map_location=device)
        assert bundle.get("head") == "gnn3"
        yt, yp, lv = _predict(bundle, graphs, device)
        for dim in (64, 32, 16):
            m = lv == dim
            if not m.any():
                continue
            acc = float((yp[m] == yt[m]).mean())
            f1, sr = _macro_f1_and_split_recall(yt[m], yp[m])
            rows.append("{},{},{},{:.4f},{:.4f},{:.4f}".format(
                tag, dim, int(m.sum()), acc, f1, sr))
            print("[{} dim {:>2}] n={} acc={:.3f} macroF1={:.3f} "
                  "split_recall={:.3f}".format(tag, dim, int(m.sum()), acc, f1, sr),
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Gate 1-B ->", args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
