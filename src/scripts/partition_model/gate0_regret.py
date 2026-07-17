#!/usr/bin/env python3
"""Gate 0 (Solução 4): antes de treinar, valida que o sinal de regret é
computável a partir do dataset_h9 — cobertura de none_rdcost, variância do
regret, ordenação esperada e fração censurada (folhas retangulares). Ver spec §4."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as datamod  # noqa: E402
import build_regret_targets as brt  # noqa: E402
from partition_defs import MODEL_LEVELS  # noqa: E402

# Critérios de PASSAGEM (propostas iniciais; calibráveis).
MIN_EXACT_FRACTION = 0.40   # >=40% dos nós de decisão são exatos (sem censura)
MIN_REGRET_STD = 1e-3       # regret tem variância real (não é degenerado)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--per-pkl", type=int, default=500)
    p.add_argument("--out", default="/workspace/results/models/regret/gate0.csv")
    args = p.parse_args(argv)

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, brt.VAL_SEQS, brt.TRAIN_SEQS)
    datamod.assert_real_luma(train_e)
    alln = brt.collect_regret_by_dim(train_e, per_pkl=args.per_pkl,
                                     exact_only=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ok = True
    rows = ["dim,n,exact_frac,regret_std,regret_med,frac_zero"]
    for dim, _ in MODEL_LEVELS:
        v = alln[dim]
        n = len(v["regret"])
        if n == 0:
            print("[dim {}] SEM NÓS -> FALHA".format(dim))
            ok = False
            continue
        exact_frac = float(v["exact"].mean())
        std = float(v["regret"].std())
        med = float(np.median(v["regret"]))
        frac_zero = float((v["regret"] == 0).mean())
        rows.append("{},{},{:.4f},{:.6f},{:.6f},{:.4f}".format(
            dim, n, exact_frac, std, med, frac_zero))
        dim_ok = (exact_frac >= MIN_EXACT_FRACTION) and (std >= MIN_REGRET_STD)
        ok = ok and dim_ok
        print("[dim {:>2}] n={} exact={:.2%} std={:.4f} med={:.4f} "
              "zero={:.2%} -> {}".format(dim, n, exact_frac, std, med,
                                         frac_zero, "OK" if dim_ok else "FALHA"),
              flush=True)
    with open(args.out, "w") as f:
        f.write("\n".join(rows) + "\n")
    print("Gate 0:", "PASSOU" if ok else "FALHOU", "->", args.out)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main(sys.argv[1:])
