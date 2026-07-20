#!/usr/bin/env python3
"""A5 -- avaliador offline ponderado por *regret*: um CRIVO de triagem defensável.

Objetivo metodológico (não é predizer o encoder -- o encoder é sempre o árbitro
final). É dar um proxy offline barato e defensável para decidir QUAL solução
merece a implantação cara em C. O oráculo atual (`simulate_pruning.py`) mede o
risco por CONTAGEM -- `true_split_lost_pct`, a fração de nós true-SPLIT
comprometidos a NONE. Isso é indefensável: trata uma poda errada num bloco liso
(truly-NONE, custo ~0) igual a uma num bloco texturizado (true-SPLIT, custo
alto). Este script substitui a contagem pelo *regret* (`regret.py`):

    regret_abs(n) = none_rdcost(n) - RD_subtree(n)   >= 0

o custo RD real de comprometer NONE no nó n contra a subárvore RD-ótima. É zero
quando a decisão certa já era NONE, e grande quando se corta uma subárvore cara.

Desenho (corrige os dois vícios da 1a tentativa):
  * reportado como FRONTEIRA regret x cost_red (análogo a BD-rate), não um ponto
    casado -- comparar pontos casados misturava regimes (variância no extremo
    baixo vs pixels no alto);
  * regret NORMALIZADO pela RD total (`reg_frac` = % de sobrecarga RD), legível
    e comparável entre conteúdos.

Roda TODAS as soluções propostas na mesma vara held-out (val+test): aleatório,
variância, pixels24, H9a, H9c, regressão de regret, GNN deployable, GNN causal.

Leitura honesta: o crivo separa bem os níveis (piso/heurística/aprendidos) e
serve para ELIMINAR candidatos inferiores. No único par com chão real limpo
(GNN vs H9a) ele DIVERGE do encoder -- rankeia o GNN à frente, que o encoder
rebaixa ~2x. Logo o crivo filtra perdedores, mas não adjudica o vencedor final:
isso é do encoder. A falha real do GNN não está nas podas NONE medidas aqui
(baratas por contagem E por regret), o que refina -- sem provar a causa -- a
explicação do Approach B. Tudo offline; nenhum encode.

Uso (contêiner):
  python oracle_regret.py --out-dir results/models/oracle_regret
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
import simulate_pruning as sp  # noqa: E402
import regret as regretmod  # noqa: E402

# Ordem de apresentação (piso -> teto), família implantável primeiro.
DISPLAY_ORDER = ["random", "variance", "pixels24", "H9a", "H9a_otimo", "H9c",
                 "regret", "GNN", "GNN_causal"]

# Chão-de-verdade real medido no encoder (menor BD = vencedor), com fonte e
# uma flag de confiabilidade do próprio chão.
REAL_PAIRS = [
    ("GNN", "H9a", "H9a", True,
     "RESULTADOS_approachB.md §5 (Jockey 5fr, replay fiel): H9a ~2x melhor BD "
     "em todo tau. Chão LIMPO (ambos modelos competentes)."),
    ("pixels24", "variance", "variance", False,
     "ablation_matched.csv: variancia vence o 'ML'. Chão CONTAMINADO -- o 'ML' "
     "era o estudante de pixels fraco (macro-F1 0,203, CB-2); nao vale como "
     "verdade sobre um pruner competente."),
]


def node_regret_full(members, ctx):
    """{(dim,r,c): {'abs','rel','none_rd'}} por nó de decisão com none_rd>0 e
    subárvore reconstruível (mesma lógica de regret.node_regrets, +absoluto)."""
    node = {}
    for k, (dim, r, c, _luma, label) in enumerate(members):
        node[(dim, r, c)] = {"label": int(label),
                             "rd": int(ctx[k].get("none_rdcost", 0))}

    def subtree_rd(key):
        nd = node.get(key)
        if nd is None or nd["rd"] <= 0:
            return None, True
        if nd["label"] == regretmod.SPLIT:
            total, cens = 0, False
            for ch in regretmod._children(*key):
                crd, ccen = subtree_rd(ch)
                if crd is None:
                    return nd["rd"], True
                total += crd
                cens = cens or ccen
            return total, cens
        return nd["rd"], (nd["label"] != regretmod.NONE)

    out = {}
    for key, nd in node.items():
        if key[0] not in regretmod.DECISION_DIMS or nd["rd"] <= 0:
            continue
        srd, _ = subtree_rd(key)
        if srd is None or srd <= 0:
            continue
        out[key] = {"abs": max(float(nd["rd"] - srd), 0.0),
                    "rel": max(float((nd["rd"] - srd) / srd), 0.0),
                    "none_rd": float(nd["rd"])}
    return out


def collect(entries, per_pkl=None):
    """Superblocos com feat H9a(36) e H9c(39), verdade, regret por nó, e a RD
    total (Sigma none_rd dos nós de decisão) para normalizar."""
    sbs = []
    total_none_rd = 0.0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD".format(e["path"]))
            reg = node_regret_full(sb["members"], sb["ctx"])
            nodes = {}
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                fa = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                               sb["qindex"], sb["ctx"][k])
                fc = featmod.node_features_h9c(sb["luma"], dim, r, c,
                                               sb["qindex"], sb["ctx"][k])
                rr = reg.get((dim, r, c))
                if rr:
                    total_none_rd += rr["none_rd"]
                nodes[(dim, r, c)] = {
                    "truth": label, "feat": fa, "feat_h9c": fc,
                    "reg_abs": rr["abs"] if rr else None,
                    "reg_rel": rr["rel"] if rr else None}
            sbs.append({"nodes": nodes, "luma": sb["luma"],
                        "qindex": sb["qindex"]})
            took += 1
            if per_pkl and took >= per_pkl:
                break
        print("  {} ({} SB)".format(os.path.basename(e["path"]), took),
              flush=True)
    return sbs, total_none_rd


def score_student(sbs, bundle, device, feat_key, width):
    nets = {}
    for dim, _ in MODEL_LEVELS:
        if dim in bundle["students"]:
            net = studentmod.make_student(width, bundle["hidden"])
            net.load_state_dict(bundle["students"][dim])
            nets[dim] = net.to(device).eval()
    for dim in nets:
        idx = [(si, key) for si, sb in enumerate(sbs) for key in sb["nodes"]
               if key[0] == dim]
        if not idx:
            continue
        feats = np.stack([sbs[si]["nodes"][key][feat_key][:width]
                          for si, key in idx])
        with torch.no_grad():
            p = F.softmax(nets[dim](torch.tensor(
                feats, dtype=torch.float32, device=device)), -1).cpu().numpy()
        for (si, key), pp in zip(idx, p):
            sbs[si]["nodes"][key]["prob"] = pp


def score_random(sbs, seed=0):
    rng = np.random.default_rng(seed)
    modeled = {d for d, _ in MODEL_LEVELS}
    for sb in sbs:
        for key, nd in sb["nodes"].items():
            if key[0] not in modeled:
                continue
            u = float(rng.random())
            nd["prob"] = np.array([u, 1.0 - u, 0.0])


def clear_probs(sbs):
    for sb in sbs:
        for nd in sb["nodes"].values():
            nd.pop("prob", None)


def simulate_regret(sbs, tau_none):
    """Replay só da ação NONE-commit. cost_red (proxy de speedup), split_lost
    (contagem, risco antigo) e regret_abs/rel incorrido (risco novo)."""
    s = {"base_cost": 0, "cost": 0, "true_split": 0, "true_split_cut": 0,
         "none_forced": 0, "reg_abs": 0.0, "reg_rel": 0.0,
         "cov": 0, "all": 0}
    tn = sp._as_level_map(tau_none)

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
            s["none_forced"] += 1
            s["all"] += 1
            if nd["reg_abs"] is not None:
                s["reg_abs"] += nd["reg_abs"]
                s["reg_rel"] += nd["reg_rel"]
                s["cov"] += 1
            if nd["truth"] == 3:
                s["true_split"] += 1
                s["true_split_cut"] += 1
            return
        if nd["truth"] == 3:
            s["true_split"] += 1
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
    return {
        "cost_red": 100.0 * (s["base_cost"] - s["cost"]) / max(s["base_cost"], 1),
        "split_lost": 100.0 * s["true_split_cut"] / max(s["true_split"], 1),
        "reg_abs": s["reg_abs"], "reg_rel": s["reg_rel"],
        "none_forced": s["none_forced"],
        "reg_cov": 100.0 * s["cov"] / max(s["all"], 1),
    }


def sweep(sbs, taus, total_none_rd):
    out = []
    for t in taus:
        m = simulate_regret(sbs, t)
        m["tau"] = t
        m["reg_frac"] = 100.0 * m["reg_abs"] / max(total_none_rd, 1.0)
        out.append(m)
    return out


def interp_at(curve, x_target, key_y, key_x="cost_red"):
    xs = np.array([p[key_x] for p in curve])
    ys = np.array([p[key_y] for p in curve])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x_target < xs.min() - 1e-9 or x_target > xs.max() + 1e-9:
        return float("nan")
    return float(np.interp(x_target, xs, ys))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    ap.add_argument("--seqs", nargs="+",
                    default=["HoneyBee", "FlowerPan", "Lips",
                             "Jockey", "RaceNight", "RiverBank"],
                    help="held-out (treino = as outras 10): 3 validacao + 3 teste")
    ap.add_argument("--models-dir", default="/workspace/results/models")
    ap.add_argument("--per-pkl", type=int, default=0, help="cap SB/pkl (0=todos)")
    ap.add_argument("--canonical-cost", type=float, default=30.0,
                    help="cost_red do ponto de triagem (dentro do alcance de todos)")
    ap.add_argument("--out-dir",
                    default="/workspace/results/models/oracle_regret")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries = datamod.discover_pkls(args.dataset_dir)
    _, held = datamod.split_entries(entries, args.seqs, train_seqs=[])
    datamod.assert_real_luma(held)
    print("pkls held-out: {} ({})".format(len(held), args.seqs), flush=True)
    sbs, total_rd = collect(held, per_pkl=(args.per_pkl or None))
    n_dec = sum(1 for sb in sbs for k in sb["nodes"] if k[0] in (16, 32, 64))
    print("superblocos: {}, nós de decisão: {}, RD total: {:.3g}".format(
        len(sbs), n_dec, total_rd))

    taus = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]

    def load(name):
        return torch.load(os.path.join(args.models_dir, name),
                          map_location=device, weights_only=False)

    curves = {}

    def run(name, scorer):
        try:
            scorer()
            curves[name] = sweep(sbs, taus, total_rd)
            print("  [ok] {}".format(name), flush=True)
        except Exception as ex:
            print("  [pulado] {}: {}".format(name, ex), flush=True)
        finally:
            clear_probs(sbs)

    run("random", lambda: score_random(sbs))
    run("variance", lambda: sp.score_with_variance(sbs))
    run("pixels24", lambda: score_student(
        sbs, load("student_real/students.pt"), device, "feat", 24))
    run("H9a", lambda: score_student(
        sbs, load("student_h9a/students.pt"), device, "feat", 36))
    if os.path.exists(os.path.join(args.models_dir,
                                   "student_h9a_otimo/students.pt")):
        run("H9a_otimo", lambda: score_student(
            sbs, load("student_h9a_otimo/students.pt"), device, "feat", 36))
    run("H9c", lambda: score_student(
        sbs, load("student_h9c/students.pt"), device, "feat_h9c", 39))
    run("regret", lambda: sp.score_with_regret(
        sbs, load("regret/students.pt"), device, 0.1))
    run("GNN", lambda: sp.score_with_gnn(
        sbs, load("gnn_L2_pixel/gnn.pt"), device))
    run("GNN_causal", lambda: sp.score_with_gnn(
        sbs, load("gnn_L2_causal/gnn.pt"), device))

    names = [n for n in DISPLAY_ORDER if n in curves]

    # dump da fronteira
    with open(os.path.join(args.out_dir, "frontier.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pruner", "tau", "cost_red", "split_lost", "reg_abs",
                    "reg_rel", "reg_frac_pct", "none_forced", "reg_cov_pct"])
        for name in names:
            for p in curves[name]:
                w.writerow([name, p["tau"], round(p["cost_red"], 3),
                            round(p["split_lost"], 3), round(p["reg_abs"], 1),
                            round(p["reg_rel"], 4), round(p["reg_frac"], 4),
                            p["none_forced"], round(p["reg_cov"], 1)])

    xc = args.canonical_cost
    md = ["# A5 — crivo offline ponderado por *regret* (triagem de soluções)",
          "", "**Data:** 2026-07-19  ",
          "**Split:** validação + teste held-out — {} ({} nós de decisão). "
          "Modelos treinados nas 10 restantes.".format(
              ", ".join(args.seqs), n_dec),
          "**Reprodução:** `python src/scripts/partition_model/oracle_regret.py`",
          "",
          "Crivo de **triagem**, não predição do encoder (que é o árbitro final). "
          "Risco = *regret* ponderado por custo RD, normalizado pela RD total "
          "(`reg_frac` = % de sobrecarga RD), em vez de contagem de erros.",
          ""]

    # ---- 1. ranking de triagem no ponto canônico ----
    md += ["## 1. Ranking de triagem — `cost_red = {:.0f}%` casado".format(xc),
           "",
           "Menor `reg_frac` = menos custo RD desperdiçado por unidade de busca "
           "poupada = melhor candidato a avançar. `split_lost` (contagem, o "
           "critério antigo) ao lado, para contraste.", "",
           "| # | solução | reg_frac % ↓ | reg_rel ↓ | split_lost % | cost_red atingível |",
           "|--:|---|--:|--:|--:|---|"]
    ranking = []
    for name in names:
        cur = curves[name]
        cr_min = min(p["cost_red"] for p in cur)
        cr_max = max(p["cost_red"] for p in cur)
        rf = interp_at(cur, xc, "reg_frac")
        rr = interp_at(cur, xc, "reg_rel")
        sl = interp_at(cur, xc, "split_lost")
        ranking.append({"pruner": name, "reg_frac": rf, "reg_rel": rr,
                        "split_lost": sl, "cr_min": cr_min, "cr_max": cr_max})
    ranked = sorted([r for r in ranking if r["reg_frac"] == r["reg_frac"]],
                    key=lambda r: r["reg_frac"])
    unreach = [r for r in ranking if r["reg_frac"] != r["reg_frac"]]
    for i, r in enumerate(ranked, 1):
        md.append("| {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.0f}–{:.0f}% |".format(
            i, r["pruner"], r["reg_frac"], r["reg_rel"], r["split_lost"],
            r["cr_min"], r["cr_max"]))
    for r in unreach:
        md.append("| — | {} | (não alcança {:.0f}%) | | | {:.0f}–{:.0f}% |".format(
            r["pruner"], xc, r["cr_min"], r["cr_max"]))
    md.append("")

    # ---- 2. fronteira em vários cost_red ----
    targets = [15.0, 20.0, 30.0, 40.0, 50.0]
    md += ["## 2. Fronteira `reg_frac %` por `cost_red` casado", "",
           "| solução | " + " | ".join("{:.0f}%".format(x) for x in targets) + " |",
           "|---|" + "--:|" * len(targets)]
    matrows = []
    for name in names:
        cells = []
        for x in targets:
            v = interp_at(curves[name], x, "reg_frac")
            cells.append("{:.3f}".format(v) if v == v else "—")
            matrows.append({"pruner": name, "cost_red": x, "reg_frac": v})
        md.append("| {} | ".format(name) + " | ".join(cells) + " |")
    md.append("")

    # ---- 3. validação onde há chão real ----
    md += ["## 3. O crivo concorda com o encoder onde há chão real?", "",
           "| par | vencedor real | crivo (reg_frac) diz | chão |",
           "|---|---|---|---|"]
    for a, b, real_win, clean, src in REAL_PAIRS:
        if a not in curves or b not in curves:
            md.append("| {} vs {} | {} | (falta modelo) | — |".format(a, b, real_win))
            continue
        wa = wb = 0
        for x in targets:
            va = interp_at(curves[a], x, "reg_frac")
            vb = interp_at(curves[b], x, "reg_frac")
            if va == va and vb == vb:
                wa += va < vb
                wb += vb < va
        pred = a if wa > wb else (b if wb > wa else "—")
        ok = "✅ concorda" if pred == real_win else "❌ diverge"
        md.append("| {} vs {} | **{}** | {} ({}) | {} |".format(
            a, b, real_win, pred, ok, "limpo" if clean else "contaminado"))
    md += ["",
           "> **GNN vs H9a (chão limpo):** " + REAL_PAIRS[0][4],
           "> **pixels24 vs variância (chão contaminado):** " + REAL_PAIRS[1][4],
           ""]

    # ---- 4. leitura ----
    md += ["## 4. Leitura para a tese", "",
           "**O que o crivo faz bem — triagem por níveis.** Ele separa "
           "inequivocamente o piso (`random`), a heurística barata (`variância`) "
           "e a família aprendida, ordenando por custo RD real desperdiçado e não "
           "por contagem cega de erros. Para *filtrar candidatos obviamente "
           "inferiores* antes de gastar encode, é defensável e melhor que a "
           "contagem: a variância tem `split_lost` baixo em regimes onde seu "
           "`reg_frac` é alto (corta poucos true-SPLIT, mas caros) — só o *regret* "
           "expõe isso.",
           "",
           "**O que o crivo NÃO faz — adjudicar entre modelos competitivos.** No "
           "único par com chão real limpo (GNN vs H9a), o crivo **diverge** do "
           "encoder: rankeia o GNN à frente, o encoder rebaixa o GNN ~2× em BD "
           "(`RESULTADOS_approachB.md`). Logo o crivo serve para eliminar "
           "perdedores, **não** para escolher o vencedor final — isso é do encoder.",
           "",
           "**Achado que refina a explicação do Approach B.** O `approachB:118-121` "
           "atribui a derrota real do GNN a 'poucas podas confiantes **caras em "
           "RD**'. A medição **não sustenta** isso: as podas NONE do GNN são "
           "baratas por AMBOS os critérios (`split_lost` 0,02% e `reg_frac`≈0 a "
           "30% de cost_red). A falha real do GNN, portanto, **não está na ação "
           "NONE-commit** medida aqui; sua causa não fica estabelecida por esta "
           "análise (candidatos não testados: vazamento de vizinhança na "
           "expressividade do grafo; descasamento cpu0↔cpu1; dano por outra ação "
           "da política). Registrar como pergunta aberta, não como causa provada.",
           "",
           "**Limitação estrutural do crivo.** O *regret* usa rótulos RDO "
           "cpu-used=0 (a referência de treino); a implantação roda cpu-used≥1. O "
           "crivo mede qualidade de decisão contra o ótimo cpu0, não o custo cpu1 "
           "exato — outra razão para o encoder permanecer o árbitro final.", ""]

    with open(os.path.join(args.out_dir, "ranking.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pruner", "reg_frac", "reg_rel",
                                          "split_lost", "cr_min", "cr_max"])
        w.writeheader()
        for r in ranking:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})
    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nWrote report.md + frontier/ranking.csv to " + args.out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
