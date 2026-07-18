#!/usr/bin/env python3
"""Approach B (Stage 1): treina o TreeGNN. Roda-se duas vezes com hiperparâmetros
IDÊNTICOS, mudando só n_layers (0 = baseline MLP independente, >=1 = com estrutura),
para a ablação controlada do Gate 1-B."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import data as datamod  # noqa: E402
import graph_data as gd  # noqa: E402
import gnn_model as gm  # noqa: E402

TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]


def train_gnn(graphs, hidden, layer, n_layers, device, epochs, lr, batch_sbs):
    net = gm.build_gnn(36, hidden, layer, n_layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    last = float("nan")
    for _ in range(epochs):
        order = np.random.permutation(len(graphs))
        for s in range(0, len(graphs), batch_sbs):
            batch = [graphs[i] for i in order[s:s + batch_sbs]]
            b = gd.collate(batch)
            x = torch.tensor(b["x"], device=device)
            y = torch.tensor(b["y"], device=device)
            ei = torch.tensor(b["edge_index"], device=device)
            opt.zero_grad()
            loss = loss_fn(net(x, ei), y)
            loss.backward()
            opt.step()
            last = float(loss.item())
    return net.state_dict(), last


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--n-layers", type=int, required=True,
                   help="0 = baseline MLP independente; >=1 = GNN com estrutura")
    p.add_argument("--layer", default="sage", choices=["sage", "gat", "gin"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-sbs", type=int, default=256)
    p.add_argument("--per-pkl", type=int, default=2000)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, VAL_SEQS, TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    graphs = gd.build_graph_dataset(train_e, per_pkl=args.per_pkl or None)
    print("grafos de treino:", len(graphs), flush=True)
    sd, loss = train_gnn(graphs, args.hidden, args.layer, args.n_layers, device,
                         args.epochs, args.lr, args.batch_sbs)
    out_dir = args.out_dir or "/workspace/results/models/gnn_L{}".format(args.n_layers)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"hidden": args.hidden, "layer": args.layer,
                "n_layers": args.n_layers, "state_dict": sd, "num_features": 36,
                "feature_set": "h9a", "head": "gnn3"},
               os.path.join(out_dir, "gnn.pt"))
    print("[layer={} L={}] loss final={:.5f} -> {}".format(
        args.layer, args.n_layers, loss, os.path.join(out_dir, "gnn.pt")),
        flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
