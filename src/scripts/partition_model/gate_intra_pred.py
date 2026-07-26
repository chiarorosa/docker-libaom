#!/usr/bin/env python3
"""Portao offline do bloco D' -- predizibilidade intra a partir dos vizinhos.

PERGUNTA. O bloco D implementado (`features.py::block_satd`, o "H9b" do Gate 2)
mede o SATD do bloco-FONTE, nao o residuo de uma predicao a partir dos vizinhos
-- divergindo do que `PLANO_H9_contribuicao_tese.md:326` especificou. A hipotese
original nunca foi testada. Este portao a testa, offline e sem codificar nada.

DESENHO. Mesma bancada do Gate 2 (`gate2_signal.py`): mesmo dataset, mesmo split
por sequencia, mesmo MLP por nivel, mesma politica NONE-commit + rect-off, mesma
grade de tau, mesmos limites de risco casado. A UNICA diferenca entre os bracos e
o conjunto de colunas -- o veredito e sobre a INFORMACAO, nao sobre capacidade.

  H9a      A+B+C (36)                     -- o podador implantado
  H9b      A+B+C+D (38)                   -- referencia: SATD da fonte (o D atual)
  H9a+D'   A+B+C + D' (39)                -- a hipotese do plano, enfim testada

TRES LEITURAS, em ordem decrescente de poder estatistico:

  1. DISPONIBILIDADE. Quantos nos tem os dois vizinhos dentro do superbloco.
     Em 64px e sempre ZERO -- este portao nada diz sobre aquele nivel.
  2. SINAL POR NIVEL, RESTRITO AOS NOS COM VIZINHO (alto poder). Entropia
     cruzada e AUC de P(NONE) so nos nos onde o atributo pode agir. Se D' nao
     move nada aqui, nao movera nada agregado -- e a via fecha.
  3. cost_red EM RISCO CASADO (baixo poder, mas comparavel). A tabela do Gate 2.
     Diluida porque 64px nao usa o atributo; leia como LIMITE INFERIOR.

Ver `docs/RESULTADOS_auditoria_dominio_pixels.md` para as duas tendencias de
sinal oposto embutidas na aproximacao (vizinho-fonte em vez de reconstruido).
"""

import argparse
import copy
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
import features_intrapred as ipmod  # noqa: E402
import student as studentmod  # noqa: E402
from simulate_pruning import simulate, metrics  # noqa: E402

NUM_H9 = featmod.NUM_FEATURES_H9              # 41
IP0 = NUM_H9                                  # D' ocupa 41,42,43
SUBSETS = {
    "H9a": list(range(36)),
    "H9b": list(range(38)),
    "H9a+D'": list(range(36)) + [IP0, IP0 + 1, IP0 + 2],
}


def collect(entries, limit=None, per_pkl=None):
    """Superblocos com o vetor H9 de 41 atributos + os 3 do bloco D'.

    D' so e calculado nos niveis modelados (16/32/64); os nos de 8px entram na
    arvore para a simulacao, mas nenhum MLP os consome, e o Hadamard por no e a
    parte cara desta coleta."""
    modeled = {d for d, _ in MODEL_LEVELS}
    zeros_ip = np.zeros(ipmod.NUM_FEATURES_INTRAPRED, dtype=np.float32)
    sbs = []
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD; re-extraia com a "
                                 "instrumentacao H9.".format(e["path"]))
            nodes = {}
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                ctx = dict(sb["ctx"][k])
                ctx["bsize_enum"] = -1
                f9 = featmod.node_features_h9(sb["luma"], dim, r, c,
                                              sb["qindex"], ctx)
                fip = (ipmod.node_features_intrapred(sb["luma"], dim, r, c)
                       if dim in modeled else zeros_ip)
                nodes[(dim, r, c)] = {
                    "truth": label,
                    "feat": np.concatenate([f9, fip]),
                }
            sbs.append({"nodes": nodes})
            took += 1
            if per_pkl and took >= per_pkl:
                break
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
    """Treina um MLP por tamanho de bloco em `cols`; anexa `prob` na validacao.
    Identico ao `gate2_signal.train_and_score` (media de `seeds` fits)."""
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


def _auc(scores, positives):
    """AUC pelo posto de Mann-Whitney (com correcao de empates)."""
    s = np.asarray(scores, dtype=np.float64)
    yb = np.asarray(positives, dtype=bool)
    npos, nneg = int(yb.sum()), int((~yb).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[yb].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def per_level_signal(val_sbs, avail_only=True):
    """CE e AUC de P(NONE) por nivel; opcionalmente so nos nos com vizinho."""
    out = {}
    for dim, _ in MODEL_LEVELS:
        p, yv = [], []
        for sb in val_sbs:
            for (d, r, c), nd in sb["nodes"].items():
                if d != dim or "prob" not in nd:
                    continue
                if avail_only and nd["feat"][IP0] < 0.5:
                    continue
                p.append(nd["prob"])
                yv.append(studentmod.collapse_label(nd["truth"]))
        if not p:
            out[dim] = (0, float("nan"), float("nan"))
            continue
        p = np.clip(np.array(p), 1e-9, 1.0)
        yv = np.array(yv)
        ce = float(-np.log(p[np.arange(len(yv)), yv]).mean())
        out[dim] = (len(yv), ce, _auc(p[:, 0], yv == 0))
    return out


def sweep_curve(sbs, taus_none, tau_rest):
    pts = []
    for tn in taus_none:
        for tr in tau_rest:
            m = metrics(simulate(sbs, tn, 2.0, tr))
            pts.append((tn, tr, m["cost_red"], m["none_wrong"],
                        m["split_lost"], m["rect_off_wrong"]))
    return pts


def cost_at_risk(pts, max_split_lost, max_rect_off_wrong=5.0):
    ok = [c for _, _, c, _nw, sl, rw in pts
          if sl <= max_split_lost and rw <= max_rect_off_wrong]
    return max(ok) if ok else 0.0


def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--val-seqs", nargs="+",
                   default=["HoneyBee", "FlowerPan", "Lips"])
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--subsets", nargs="+", default=["H9a", "H9b", "H9a+D'"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--per-pkl", type=int, default=None)
    p.add_argument("--out-csv",
                   default="/workspace/results/models/gate_intra_pred.csv")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs,
                                           args.train_seqs)
    datamod.assert_real_luma(train_e)
    print("train pkls: {}, val pkls: {} (val {})".format(
        len(train_e), len(val_e), args.val_seqs), flush=True)
    train_sbs = collect(train_e, limit=args.limit, per_pkl=args.per_pkl)
    val_base = collect(val_e, limit=args.limit, per_pkl=args.per_pkl)
    print("train superblocos: {}, val: {}".format(len(train_sbs),
                                                  len(val_base)), flush=True)

    # ---- Leitura 1: disponibilidade do atributo ---------------------------
    print("\n[1] disponibilidade do bloco D' (vizinhos dentro do superbloco):",
          flush=True)
    print("{:>6}  {:>10}  {:>10}  {:>8}".format("nivel", "nos", "com vizinho",
                                                "%"), flush=True)
    avail_rows = []
    for dim, _ in MODEL_LEVELS:
        tot = av = 0
        for sb in val_base:
            for (d, _r, _c), nd in sb["nodes"].items():
                if d == dim:
                    tot += 1
                    av += int(nd["feat"][IP0] >= 0.5)
        pct = 100.0 * av / tot if tot else 0.0
        print("{:>6}  {:>10}  {:>10}  {:>7.1f}%".format(dim, tot, av, pct),
              flush=True)
        avail_rows.append([dim, tot, av, round(pct, 2)])

    # ---- Bracos ------------------------------------------------------------
    curves, signal = {}, {}
    for name in args.subsets:
        cols = SUBSETS[name]
        val_sbs = copy.deepcopy(val_base)
        train_and_score(train_sbs, val_sbs, cols, device, seeds=args.seeds)
        signal[name] = per_level_signal(val_sbs, avail_only=True)
        curves[name] = sweep_curve(val_sbs, [0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                                             0.95], [-1.0, 0.1, 0.2, 0.3])
        print("  braco {} treinado ({} colunas)".format(name, len(cols)),
              flush=True)

    # ---- Leitura 2: sinal por nivel, so nos nos com vizinho ---------------
    print("\n[2] sinal por nivel, RESTRITO aos nos com vizinho "
          "(menor CE e melhor; maior AUC e melhor):", flush=True)
    print("{:>6}  {:>8}".format("nivel", "nos") +
          "".join("  {:>16}".format(n) for n in args.subsets), flush=True)
    sig_rows = []
    for dim, _ in MODEL_LEVELS:
        n_nodes = signal[args.subsets[0]][dim][0]
        line = "{:>6}  {:>8}".format(dim, n_nodes)
        row = [dim, n_nodes]
        for name in args.subsets:
            _n, ce, auc = signal[name][dim]
            line += "  CE{:6.4f}/A{:5.3f}".format(ce, auc)
            row += [round(ce, 5), round(auc, 5)]
        print(line, flush=True)
        sig_rows.append(row)

    # ---- Leitura 3: cost_red em risco casado (tabela do Gate 2) -----------
    caps = [0.5, 1.0, 2.0, 3.0]
    print("\n[3] cost_red% em risco casado (limite de SPLIT-lost; "
          "rect_off_wrong<=5%) -- DILUIDO, 64px nao usa D':", flush=True)
    print("subset       " + "".join("  <=SL{:.1f}%".format(c) for c in caps),
          flush=True)
    cost_rows = []
    for name in args.subsets:
        cells = [cost_at_risk(curves[name], c, 5.0) for c in caps]
        print("{:<11}".format(name) +
              "".join("  {:8.2f}".format(v) for v in cells), flush=True)
        cost_rows.append([name] + [round(v, 3) for v in cells])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# leitura 1: disponibilidade"])
        w.writerow(["dim", "nodes", "avail", "avail_pct"])
        w.writerows(avail_rows)
        w.writerow([])
        w.writerow(["# leitura 2: CE/AUC por nivel, so nos com vizinho"])
        hdr = ["dim", "nodes"]
        for n in args.subsets:
            hdr += ["ce_" + n, "auc_" + n]
        w.writerow(hdr)
        w.writerows(sig_rows)
        w.writerow([])
        w.writerow(["# leitura 3: cost_red em risco casado"])
        w.writerow(["subset"] + ["cost_red_SL{}".format(c) for c in caps])
        w.writerows(cost_rows)
    with open(args.out_csv.replace(".csv", "_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "tau_none", "tau_rest", "cost_red", "none_wrong",
                    "split_lost", "rect_off_wrong"])
        for name in args.subsets:
            for row in curves[name]:
                w.writerow([name] + [round(x, 3) for x in row])
    print("\nescreveu", args.out_csv, flush=True)
    print("\nPORTAO: se [2] nao mostrar CE menor / AUC maior para H9a+D' em "
          "16 e 32px, o atributo nao carrega sinal alem do bloco A -- a via de "
          "pixels fecha sem custo de re-extracao.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
