#!/usr/bin/env python3
"""B2 -- tau adaptativo por qindex: calibração offline (sem re-treino).

Hipótese (A8): a grade tau fixa deixa ganho na mesa porque a mistura de rótulos
varia fortemente com o CQ (SPLIT em 16x16 vai de 7,6% em cq20 a 0,14% em cq55).
Um tau global é conservador num regime e agressivo noutro.

Teste, na mesma vara do crivo (`oracle_regret`) e com o MESMO risco ponderado por
*regret*: compara duas políticas de NONE-commit para o estudante implantado H9a,
a `cost_red` casado --
  * GLOBAL: um tau único aplicado a todos os qindex (a política atual);
  * ESTRATIFICADA-ÓTIMA: cada qindex escolhe seu tau. Enumeram-se TODAS as
    combinações (grade^4, exato) e toma-se a fronteira de Pareto (cost_red x regret).
Se a estratificada dominar a global além do ruído, o B2 ajuda. O lado C já lê tau
do ambiente (sem recompilar); a confirmação em BD x tempo exigiria encodes.

Uso (contêiner):
  python b2_tau_per_qindex.py --out-dir results/models/b2_tau_qindex
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import oracle_regret as orq  # noqa: E402
import simulate_pruning as sp  # noqa: E402
import data as datamod  # noqa: E402

# qindex -> rótulo CQ (base_qindex = 4*cq; medido em A8).
CQ_OF_Q = {80: "cq20", 128: "cq32", 172: "cq43", 220: "cq55"}


def simulate_raw(sbs, tau_none):
    """Soma ABSOLUTA (base_cost, cost, reg_abs) para um subconjunto de sbs a um
    tau_none escalar (NONE-commit), para poder agregar entre buckets de qindex."""
    tn = {d: tau_none for d, _ in sp.MODEL_LEVELS}
    s = {"base_cost": 0, "cost": 0, "reg_abs": 0.0}

    def visit(nodes, key):
        nd = nodes.get(key)
        if nd is None:
            return
        dim = key[0]
        present = [(dim // 2, 2 * key[1] + dr, 2 * key[2] + dc)
                   for dr, dc in sp.CHILD_OFFSETS] if dim > 8 else []
        present = [c for c in present if c in nodes]
        prob = nd.get("prob")
        if prob is None:
            s["cost"] += sp.node_cost(dim)
            for c in present:
                visit(nodes, c)
            return
        if float(prob[0]) > tn[dim]:
            s["cost"] += dim * dim
            if nd["reg_abs"] is not None:
                s["reg_abs"] += nd["reg_abs"]
            return
        s["cost"] += sp.node_cost(dim)
        for c in present:
            visit(nodes, c)

    def baseline(nodes, key):
        if key not in nodes:
            return
        s["base_cost"] += sp.node_cost(key[0])
        if key[0] > 8:
            for dr, dc in sp.CHILD_OFFSETS:
                baseline(nodes, (key[0] // 2, 2 * key[1] + dr, 2 * key[2] + dc))

    for sb in sbs:
        baseline(sb["nodes"], (64, 0, 0))
        visit(sb["nodes"], (64, 0, 0))
    return s


def pareto(points):
    """Fronteira de Pareto: max cost_red, min reg_frac. points=[(cost,reg,meta)]."""
    pts = sorted(points, key=lambda p: (-p[0], p[1]))
    out, best_reg = [], float("inf")
    for cr, rf, meta in pts:            # cost decrescente
        if rf < best_reg - 1e-12:
            out.append((cr, rf, meta))
            best_reg = rf
    return sorted(out, key=lambda p: p[0])


def interp(frontier, x):
    xs = np.array([p[0] for p in frontier])
    ys = np.array([p[1] for p in frontier])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if x < xs.min() or x > xs.max():
        return float("nan")
    return float(np.interp(x, xs, ys))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    ap.add_argument("--models-dir", default="/workspace/results/models")
    ap.add_argument("--seqs", nargs="+",
                    default=["HoneyBee", "FlowerPan", "Lips",
                             "Jockey", "RaceNight", "RiverBank"])
    ap.add_argument("--per-pkl", type=int, default=2000)
    ap.add_argument("--out-dir", default="/workspace/results/models/b2_tau_qindex")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, held = datamod.split_entries(entries, args.seqs, train_seqs=[])
    datamod.assert_real_luma(held)
    sbs, total_rd = orq.collect(held, per_pkl=(args.per_pkl or None))
    bundle = torch.load(os.path.join(args.models_dir, "student_h9a/students.pt"),
                        map_location=device, weights_only=False)
    orq.score_student(sbs, bundle, device, "feat", 36)
    print("superblocos: {}, RD total: {:.3g}".format(len(sbs), total_rd),
          flush=True)

    # bucketiza por qindex
    buckets = {}
    for sb in sbs:
        buckets.setdefault(sb["qindex"], []).append(sb)
    qs = sorted(buckets)
    print("qindex buckets:", {CQ_OF_Q.get(q, q): len(buckets[q]) for q in qs})

    taus = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]

    # sim por (qindex, tau): base/cost/reg absolutos
    raw = {q: {} for q in qs}
    base_total = 0.0
    for q in qs:
        b = None
        for t in taus:
            s = simulate_raw(buckets[q], t)
            raw[q][t] = s
            b = s["base_cost"]
        base_total += b   # base independe de tau

    def agg(cost_sum, reg_sum):
        cost_red = 100.0 * (base_total - cost_sum) / max(base_total, 1)
        reg_frac = 100.0 * reg_sum / max(total_rd, 1.0)
        return cost_red, reg_frac

    # frontier GLOBAL: mesmo tau em todos os buckets
    global_front = []
    for t in taus:
        cost = sum(raw[q][t]["cost"] for q in qs)
        reg = sum(raw[q][t]["reg_abs"] for q in qs)
        cr, rf = agg(cost, reg)
        global_front.append((cr, rf, {"tau": t}))

    # frontier ESTRATIFICADA: todas as combinações de tau por qindex -> Pareto
    strat_points = []
    for combo in itertools.product(taus, repeat=len(qs)):
        cost = sum(raw[q][combo[i]]["cost"] for i, q in enumerate(qs))
        reg = sum(raw[q][combo[i]]["reg_abs"] for i, q in enumerate(qs))
        cr, rf = agg(cost, reg)
        strat_points.append((cr, rf, {CQ_OF_Q.get(q, q): combo[i]
                                      for i, q in enumerate(qs)}))
    strat_front = pareto(strat_points)
    global_pareto = pareto([(cr, rf, m) for cr, rf, m in global_front])

    # --- relatório ---
    targets = [15.0, 20.0, 25.0, 30.0, 35.0]
    md = ["# B2 — τ adaptativo por qindex (calibração offline, H9a)",
          "", "**Data:** 2026-07-20  ",
          "**Split:** validação + teste held-out — {} superblocos.".format(len(sbs)),
          "**Reprodução:** `python src/scripts/partition_model/b2_tau_per_qindex.py`",
          "",
          "Compara, no risco ponderado por *regret* do crivo, o τ GLOBAL (um só "
          "para todos os CQ, política atual) contra o τ ESTRATIFICADO-ÓTIMO (cada "
          "qindex escolhe o seu; Pareto sobre todas as combinações).", "",
          "## 1. `reg_frac %` a `cost_red` casado — global vs estratificado", "",
          "| cost_red alvo | τ global reg_frac | τ estratificado reg_frac | ganho (pp) |",
          "|--:|--:|--:|--:|"]
    rows = []
    for x in targets:
        g = interp(global_pareto, x)
        s = interp(strat_front, x)
        gain = (g - s) if (g == g and s == s) else float("nan")
        md.append("| {:.0f}% | {} | {} | {} |".format(
            x, "{:.4f}".format(g) if g == g else "—",
            "{:.4f}".format(s) if s == s else "—",
            "{:+.4f}".format(gain) if gain == gain else "—"))
        rows.append({"cost_red": x, "global_reg_frac": g, "strat_reg_frac": s,
                     "gain_pp": gain})
    md.append("")

    # a que τ por qindex o ótimo escolhe, num ponto representativo
    md += ["## 2. τ por qindex que a política ótima escolhe (evidência do B2)", ""]
    for x in (20.0, 30.0):
        near = min([p for p in strat_front], key=lambda p: abs(p[0] - x),
                   default=None)
        if near:
            md.append("- **cost_red ≈ {:.0f}%** (ponto {:.1f}%): τ = {}".format(
                x, near[0], near[2]))
    md += ["", "## 3. Fronteira por qindex isolado (τ para atingir cada cost_red)",
           "",
           "| qindex | " + " | ".join("cr {:.0f}%".format(x) for x in targets) + " |",
           "|---|" + "--:|" * len(targets)]
    for q in qs:
        qfront = [agg_q(raw[q][t], base_total_q(raw[q]), total_rd) + ({"tau": t},)
                  for t in taus]
        qpar = pareto(qfront)
        cells = []
        for x in targets:
            # tau que atinge cost_red x nesse qindex (interp inversa aproximada)
            crs = [p[0] for p in sorted(qfront, key=lambda p: p[0])]
            tvs = [p[2]["tau"] for p in sorted(qfront, key=lambda p: p[0])]
            if x < min(crs) or x > max(crs):
                cells.append("—")
            else:
                cells.append("{:.2f}".format(float(np.interp(x, crs, tvs))))
        md.append("| {} | ".format(CQ_OF_Q.get(q, q)) + " | ".join(cells) + " |")
    md.append("")

    md += ["## 4. Veredito", ""]
    gains = [r["gain_pp"] for r in rows if r["gain_pp"] == r["gain_pp"]]
    max_gain = max(gains) if gains else float("nan")
    md.append("Ganho máximo do τ estratificado sobre o global: **{:.4f} pp** de "
              "reg_frac.".format(max_gain) if max_gain == max_gain else "sem overlap")
    md.append("")
    md.append("(reg_frac é % de sobrecarga RD; a família aprendida opera na casa de "
              "0,00X, então interpretar o ganho em termos RELATIVOS e lembrar que a "
              "confirmação real exige encodes — o lado C já lê τ do ambiente.)")

    with open(os.path.join(args.out_dir, "b2_frontier.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cost_red", "global_reg_frac",
                                          "strat_reg_frac", "gain_pp"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nWrote report.md + b2_frontier.csv to " + args.out_dir)


def base_total_q(raw_q):
    return next(iter(raw_q.values()))["base_cost"]


def agg_q(s, base, total_rd):
    cost_red = 100.0 * (base - s["cost"]) / max(base, 1)
    reg_frac = 100.0 * s["reg_abs"] / max(total_rd, 1.0)
    return (cost_red, reg_frac)


if __name__ == "__main__":
    main(sys.argv[1:])
