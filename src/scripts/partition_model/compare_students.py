#!/usr/bin/env python3
"""Compara dois bundles de estudante H9a no mesmo conjunto held-out: recall por
classe e macro-F1 por tamanho de bloco. Usado para avaliar o B4 (H9a_otimo =
H9a + ponderação de classe por nível) contra o H9a implantado.

O alvo do B4 é o split_recall em 16x16 (0,0259 medido, A8/gate1). O risco é
suprimir P(NONE) confiante (distill.py:114-116) e degradar a calibração (A4).
Este script mede o primeiro; a calibração roda por calibration.py.

Uso: python compare_students.py --a student_h9a --b student_h9a_otimo
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import simulate_pruning as sp  # noqa: E402
import student as studentmod  # noqa: E402


def per_dim_stats(sbs):
    """Por tamanho: recall NONE/SPLIT/REST e macro-F1 (argmax vs verdade 3-cls)."""
    out = {}
    for dim, _ in MODEL_LEVELS:
        pred, true = [], []
        for sb in sbs:
            for (d, r, c), nd in sb["nodes"].items():
                if d == dim and "prob" in nd:
                    pred.append(int(np.argmax(nd["prob"])))
                    true.append(studentmod.collapse_label(nd["truth"]))
        pred, true = np.array(pred), np.array(true)
        if not len(true):
            continue
        rec, f1s = {}, []
        for cls, nm in [(0, "NONE"), (1, "SPLIT"), (2, "REST")]:
            tp = int(((pred == cls) & (true == cls)).sum())
            fp = int(((pred == cls) & (true != cls)).sum())
            fn = int(((pred != cls) & (true == cls)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            rec[nm] = r
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        out[dim] = {"n": len(true), "recNONE": rec["NONE"],
                    "recSPLIT": rec["SPLIT"], "recREST": rec["REST"],
                    "macroF1": float(np.mean(f1s))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="/workspace/results/models")
    ap.add_argument("--a", default="student_h9a")
    ap.add_argument("--b", default="student_h9a_otimo")
    ap.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    ap.add_argument("--seqs", nargs="+",
                    default=["HoneyBee", "FlowerPan", "Lips",
                             "Jockey", "RaceNight", "RiverBank"])
    ap.add_argument("--limit", type=int, default=30000,
                    help="cap global de superblocos (0 = todos). Viés de "
                         "amostragem afeta A e B igual -> comparação relativa OK.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, held = datamod.split_entries(entries, args.seqs, train_seqs=[])
    sbs = sp.collect_superblocks(held, feature_set="h9a",
                                 limit=(args.limit or None))
    print("superblocos: {}".format(len(sbs)), flush=True)

    res = {}
    for tag, name in [("A", args.a), ("B", args.b)]:
        bundle = torch.load(os.path.join(args.models_dir, name, "students.pt"),
                            map_location=device, weights_only=False)
        sp.score_with_student(sbs, bundle, device)
        res[name] = per_dim_stats(sbs)
        for sb in sbs:
            for nd in sb["nodes"].values():
                nd.pop("prob", None)

    print("\n{:<8} {:>4} | {:>10} {:>10} {:>10} {:>9}".format(
        "modelo", "dim", "recNONE", "recSPLIT", "recREST", "macroF1"))
    for dim, _ in MODEL_LEVELS:
        for name in (args.a, args.b):
            s = res[name].get(dim)
            if s:
                print("{:<8} {:>4} | {:>10.4f} {:>10.4f} {:>10.4f} {:>9.4f}".format(
                    name.replace("student_", ""), dim, s["recNONE"],
                    s["recSPLIT"], s["recREST"], s["macroF1"]))
        print("-" * 56)


if __name__ == "__main__":
    main()
