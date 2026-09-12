#!/usr/bin/env python3
"""BC contra BC+NC recortado por ponto de quantizacao e por sequencia de teste.

A Tabela I do artigo agrega os quatro cq-level e as tres sequencias de teste numa
unica fronteira. Todos os bracos dividem a MESMA normalizacao, entao a comparacao
e justa -- mas agregada ela nao responde duas perguntas que um revisor faz:

  1. o ganho de BC+NC sobre BC sobrevive em cada cq-level isolado, ou vem de um
     regime de quantizacao so?
  2. o ganho aparece nos tres conteudos independentes, ou e dominado por uma
     sequencia?

Nada precisa ser retreinado: os pacotes do `rpp_ladder` ja existem. O que se
refaz e o REPLAY, agora fatiado. Isso e valido porque o replay da arvore podada
e independente entre superblocos -- podar um no de um superbloco nao muda nada
em outro --, logo particionar a lista de superblocos e somar o denominador
restrito a cada particao da exatamente a fronteira daquele recorte.

Uma unica leitura do dataset serve os sete recortes (4 cq + 3 sequencias), em vez
de sete execucoes do `oracle_regret.py`.

Uso (conteiner):
    build/venv-ml/bin/python src/scripts/partition_model/slice_frontier.py \
        --out results/models/oracle_regret_rpp_test3/slices.csv
"""
import argparse
import collections
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import oracle_regret as orc  # noqa: E402

TAUS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
READ = [10.0, 15.0, 20.0, 25.0, 30.0]
RUNGS = [("A", "BC"), ("A_Bshuf", "BC+shuffled NC"), ("A_B", "BC+NC")]


def slice_key(src, mode):
    """('Jockey', 'cq32') a partir de Jockey_3840x2160_..._cq32.pkl."""
    seq = src.split("_")[0]
    m = re.search(r"_cq(\d+)", src)
    cq = "cq" + m.group(1) if m else "cq?"
    return seq if mode == "seq" else cq


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    ap.add_argument("--seqs", nargs="+",
                    default=["Jockey", "RaceNight", "RiverBank"])
    ap.add_argument("--models-dir", default="/workspace/results/models")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="/workspace/results/models/"
                                     "oracle_regret_rpp_test3/slices.csv")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, held = datamod.split_entries(entries, args.seqs, train_seqs=[])
    datamod.assert_real_luma(held)
    print("pkls: {} ({})".format(len(held), args.seqs), flush=True)

    # Coleta enxuta: os bracos aqui leem so nd["feat"].
    sbs, total_rd = orc.collect(held, need_luma=False, need_extra=False)
    print("superblocos: {}, RD total: {:.3g}".format(len(sbs), total_rd),
          flush=True)

    # Um indice de superblocos por recorte, mais o denominador de cada um.
    groups = {}
    for mode in ("cq", "seq"):
        for i, sb in enumerate(sbs):
            groups.setdefault((mode, slice_key(sb["src"], mode)), []).append(i)
    groups[("all", "all")] = list(range(len(sbs)))

    rows = []
    for rung, label in RUNGS:
        for seed in args.seeds:
            tag = "{}_s{}".format(rung, seed)
            bundle = torch.load(
                os.path.join(args.models_dir, "rpp_ladder", tag, "students.pt"),
                map_location=device, weights_only=False)
            orc.score_student(sbs, bundle, device, "feat", 0,
                              cols=featmod.RPP_SUBSETS[rung],
                              shuffle_cols=featmod.RPP_SHUFFLE_COLS.get(rung),
                              shuffle_seed=773000 + seed * 1000)
            for (mode, key), idx in sorted(groups.items()):
                sub = [sbs[i] for i in idx]
                den = sum(sbs[i]["none_rd"] for i in idx)
                curve = orc.sweep(sub, TAUS, den)
                for x in READ:
                    rows.append({
                        "mode": mode, "slice": key, "rung": rung,
                        "label": label, "seed": seed, "cost_red": x,
                        "ppm": orc.interp_at(curve, x, "reg_frac") * 1e4})
            orc.clear_probs(sbs)
            print("  [ok] {}".format(tag), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "slice", "rung", "label",
                                          "seed", "cost_red", "ppm"])
        w.writeheader()
        for r in rows:
            r["ppm"] = round(r["ppm"], 4)
            w.writerow(r)
    print("Saved -> {}".format(args.out))

    # ---- leitura: BC contra BC+NC em cada recorte, no ponto ancora de 25% ----
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["mode"], r["slice"], r["rung"], r["cost_red"])].append(r["ppm"])
    for mode, title in (("cq", "POR PONTO DE QUANTIZACAO"),
                        ("seq", "POR SEQUENCIA DE TESTE")):
        print("\n=== {} — em 25% de reducao casada (10^-4 %) ===".format(title))
        print("  {:<12}{:>9}{:>9}{:>9}{:>14}{:>12}".format(
            "recorte", "BC", "shufNC", "BC+NC", "BC+NC s/ BC", "s/ controle"))
        keys = sorted({k[1] for k in agg if k[0] == mode})
        for key in keys:
            def m(rung):
                v = agg.get((mode, key, rung, 25.0), [])
                return float(np.mean(v)) if v else float("nan")
            bc, sh, nc = m("A"), m("A_Bshuf"), m("A_B")
            print("  {:<12}{:9.2f}{:9.2f}{:9.2f}{:13.1f}%{:11.1f}%".format(
                key, bc, sh, nc, 100 * (bc - nc) / bc, 100 * (sh - nc) / sh))


if __name__ == "__main__":
    main(sys.argv[1:])
