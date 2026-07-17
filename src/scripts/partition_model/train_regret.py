#!/usr/bin/env python3
"""Treina a NN regressora de regret da Solução 4: um MLP por tamanho de bloco
(mesma topologia [64,32] do estudante H9a), com CABEÇA DE REGRESSÃO ÚNICA. Alvo =
log1p(regret_rel). Perda Huber. Entrada = 36 features H9a (fonte única). O bundle é
compatível com export_weights/simulate (num_features/feature_set), com head='regret'."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import build_regret_targets as brt  # noqa: E402
import data as datamod  # noqa: E402


def build_regressor(in_features, hidden):
    layers, d = [], in_features
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, 1)]
    net = nn.Sequential(*layers)

    class Reg(nn.Module):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def forward(self, x):
            return self.body(x).squeeze(-1)

    return Reg(net)


def _fit_dim(feat, target, hidden, device, epochs, lr):
    mean = feat.mean(0)
    std = feat.std(0) + 1e-6
    xn = (feat - mean) / std
    x = torch.tensor(xn, dtype=torch.float32, device=device)
    y = torch.tensor(np.log1p(target), dtype=torch.float32, device=device)
    net = build_regressor(feat.shape[1], hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=1.0)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    return net, (mean.astype(np.float32), std.astype(np.float32)), float(loss.item())


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--out-dir", default="/workspace/results/models/regret")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per-pkl", type=int, default=3000)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, brt.VAL_SEQS, brt.TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    data = brt.collect_regret_by_dim(train_e, per_pkl=args.per_pkl or None,
                                     exact_only=True)

    os.makedirs(args.out_dir, exist_ok=True)
    bundle = {"hidden": args.hidden, "students": {}, "norm": {},
              "num_features": 36, "feature_set": "h9a", "head": "regret"}
    for dim, _ in MODEL_LEVELS:
        rec = data[dim]
        if len(rec["regret"]) == 0:
            print("[dim {}] sem amostras".format(dim))
            continue
        net, norm, loss = _fit_dim(rec["feat"], rec["regret"], args.hidden,
                                   device, args.epochs, args.lr)
        print("[dim {:>2}] n={} huber={:.5f}".format(
            dim, len(rec["regret"]), loss), flush=True)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm
    torch.save(bundle, os.path.join(args.out_dir, "students.pt"))
    print("Saved ->", os.path.join(args.out_dir, "students.pt"))


if __name__ == "__main__":
    main(sys.argv[1:])
