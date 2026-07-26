#!/usr/bin/env python3
"""Bloco 7 — relatorio de E3 (joelho da curva de tau), decomposicao de 3 pernas
e E2 (sigma medido do tempo de parede).

E3.  A sensibilidade a tau do H9c era suave e monotonica, com joelho pronunciado
entre tau30 e tau60 (6,1 pp de TS por 0,48 pp de BD, contra 0,09 pp em toda a
faixa tau60->95). A faixa (30,60) estava inexplorada: o E3 mede tau45. A curva
completa so existe no subconjunto casado de 3 seqs (Neon1224/PierSeaSide/
TimeLapse); tau45 foi medido nas 8.

DEC.  O E4 mediu "quanto do TS do h9c_tau90 some quando o H9a e neutralizado".
Com h9adef (H9a@default sozinho, H9c off) nas mesmas sequencias, a decomposicao
fecha em tres pernas e a INTERACAO passa a ser observavel:
    TS(H9a) + TS(H9c) + interacao = TS(H9a+H9c)
Interacao negativa = os dois podam os mesmos nos (redundancia).

E2.  Cinco repeticoes intercaladas de anchor/ml_balanced/native_cpu1 numa
sequencia. Ate aqui o piso de ruido era INFERIDO por violacoes de monotonicidade
(~1-2% por encode); aqui e medido. O numero que importa nao e o sigma do tempo
bruto, e o sigma do TS pareado -- que e a estatistica que a tese reporta.

Reproducao:
    /workspace/build/venv-ml/bin/python src/scripts/fase6/report_e3_dec_e2.py
"""

import argparse
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "benchmark"))
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
from bd_rate import bd_rate  # noqa: E402

CQS = (20, 32, 43, 55)
TAU_ORDER = ["h9c_tau30", "h9c_tau45", "h9c_tau60", "h9c_tau70",
             "h9c_tau90", "h9c_tau95"]


def load(path):
    d = {}
    for r in csv.DictReader(open(path, newline="")):
        fps = int(r["fps_num"]) / int(r["fps_den"])
        kbps = int(r["bytes"]) * 8 * fps / int(r["frames"]) / 1000.0
        d.setdefault(r["seq"], {}).setdefault(r["config"], {})[int(r["cq"])] = {
            "rate": kbps, "psnr": float(r["psnr_y"]), "time": float(r["time_s"])}
    return d


def full(cfgs, name):
    pts = cfgs.get(name)
    return pts if pts and all(c in pts for c in CQS) else None


def metrics(anchor, pts):
    bd = bd_rate([anchor[c]["rate"] for c in CQS],
                 [anchor[c]["psnr"] for c in CQS],
                 [pts[c]["rate"] for c in CQS],
                 [pts[c]["psnr"] for c in CQS])
    ts = st.mean((anchor[c]["time"] - pts[c]["time"]) / anchor[c]["time"] * 100.0
                 for c in CQS)
    return bd, ts


# --------------------------------------------------------------------------
def report_e3(d):
    print("=" * 74)
    print("E3 - curva de tau do H9c: onde esta o joelho")
    print("=" * 74)
    matched = [s for s in sorted(d)
               if full(d[s], "anchor") and all(full(d[s], t) for t in TAU_ORDER)]
    print("subconjunto casado (todos os taus): {}".format(
        ", ".join(matched) or "NENHUM"))
    if not matched:
        return
    print("\n{:<12}{:>11}{:>9}{:>14}{:>14}".format(
        "tau", "BD-rate(%)", "TS(%)", "d BD/d TS", "vs anterior"))
    prev = None
    for t in TAU_ORDER:
        bds, tss = [], []
        for s in matched:
            bd, ts = metrics(d[s]["anchor"], d[s][t])
            bds.append(bd); tss.append(ts)
        bd, ts = st.mean(bds), st.mean(tss)
        if prev is None:
            print("{:<12}{:>+11.3f}{:>9.2f}{:>14}{:>14}".format(
                t, bd, ts, "--", "--"))
        else:
            dbd, dts = prev[0] - bd, prev[1] - ts   # tau menor = mais agressivo
            slope = dbd / dts if abs(dts) > 1e-9 else float("nan")
            print("{:<12}{:>+11.3f}{:>9.2f}{:>14.4f}{:>14}".format(
                t, bd, ts, slope, "{:+.2f} pp TS".format(-dts)))
        prev = (bd, ts)
    print("\n(d BD/d TS = pp de BD-rate pagos por pp de TS ao afrouxar tau um "
          "degrau;\n valores maiores = trecho caro da curva.)")

    print("\n--- tau45 nas 8 sequencias (vs anchor) ---")
    rows = []
    for s in sorted(d):
        a, t45 = full(d[s], "anchor"), full(d[s], "h9c_tau45")
        if a and t45:
            bd, ts = metrics(a, t45)
            rows.append((s, bd, ts))
            print("{:<15}{:>+9.3f}%{:>9.1f}%".format(s, bd, ts))
    if rows:
        print("{:<15}{:>+9.3f}%{:>9.1f}%   (media, n={})".format(
            "MEDIA", st.mean(r[1] for r in rows), st.mean(r[2] for r in rows),
            len(rows)))


def report_decomposition(d):
    print("\n" + "=" * 74)
    print("DEC - decomposicao de 3 pernas: H9a puro / H9c puro / interacao")
    print("=" * 74)
    print("{:<15}{:>10}{:>10}{:>10}{:>12}{:>11}".format(
        "Sequencia", "H9a so", "H9c so", "H9a+H9c", "soma", "interacao"))
    rows = []
    for s in sorted(d):
        a = full(d[s], "anchor")
        h9a = full(d[s], "h9adef")
        h9c = full(d[s], "h9ciso_tau90")
        both = full(d[s], "h9c_tau90")
        if not (a and h9a and h9c and both):
            continue
        _, ts_a = metrics(a, h9a)
        _, ts_c = metrics(a, h9c)
        _, ts_b = metrics(a, both)
        inter = ts_b - (ts_a + ts_c)
        rows.append((ts_a, ts_c, ts_b, inter))
        print("{:<15}{:>9.1f}%{:>9.1f}%{:>9.1f}%{:>11.1f}%{:>10.1f} pp".format(
            s, ts_a, ts_c, ts_b, ts_a + ts_c, inter))
    if rows:
        print("-" * 74)
        print("{:<15}{:>9.1f}%{:>9.1f}%{:>9.1f}%{:>11.1f}%{:>10.1f} pp".format(
            "MEDIA", st.mean(r[0] for r in rows), st.mean(r[1] for r in rows),
            st.mean(r[2] for r in rows),
            st.mean(r[0] + r[1] for r in rows), st.mean(r[3] for r in rows)))
        print("\nInteracao negativa = os dois podadores disputam os mesmos nos "
              "(redundancia):\no empilhamento entrega menos que a soma das "
              "partes.")


def report_e2(path, reps):
    print("\n" + "=" * 74)
    print("E2 - sigma MEDIDO do tempo de parede ({} repeticoes)".format(reps))
    print("=" * 74)
    if not os.path.exists(path):
        print("ausente:", path)
        return
    raw = {}
    for r in csv.DictReader(open(path, newline="")):
        cfg, rep = r["config"].rsplit("_r", 1)
        raw.setdefault(cfg, {}).setdefault(int(r["cq"]), {})[int(rep)] = \
            float(r["time_s"])

    print("\n--- (a) dispersao do tempo bruto, por configuracao e CQ ---")
    print("{:<14}{:>5}{:>10}{:>9}{:>8}".format("config", "cq", "media(s)",
                                               "sd(s)", "CV%"))
    cvs = []
    for cfg in sorted(raw):
        for cq in sorted(raw[cfg]):
            v = list(raw[cfg][cq].values())
            if len(v) < 2:
                continue
            m, sd = st.mean(v), st.stdev(v)
            cvs.append(sd / m * 100.0)
            print("{:<14}{:>5}{:>10.1f}{:>9.2f}{:>8.2f}".format(
                cfg, cq, m, sd, sd / m * 100.0))
    if cvs:
        print("\nCV por encode: mediana {:.2f}%, maximo {:.2f}%".format(
            st.median(cvs), max(cvs)))

    print("\n--- (b) o que importa: sigma do TS pareado vs anchor ---")
    anchor = raw.get("anchor")
    if not anchor:
        print("sem anchor repetido")
        return
    for cfg in sorted(k for k in raw if k != "anchor"):
        per_rep = []
        reps_common = sorted(set.intersection(
            *[set(raw[cfg][cq]) & set(anchor[cq]) for cq in sorted(raw[cfg])]))
        for rp in reps_common:
            ts = st.mean((anchor[cq][rp] - raw[cfg][cq][rp]) / anchor[cq][rp]
                         * 100.0 for cq in sorted(raw[cfg]))
            per_rep.append(ts)
        if len(per_rep) < 2:
            continue
        m, sd = st.mean(per_rep), st.stdev(per_rep)
        se = sd / len(per_rep) ** 0.5
        print("{:<14} TS = {:.2f}% +- {:.2f} pp (sd de {} repeticoes; "
              "se {:.2f} pp)".format(cfg, m, sd, len(per_rep), se))
        print("{:<14}   -> resolucao: diferencas abaixo de ~{:.2f} pp "
              "(2 sd) nao sao distinguiveis num encode unico".format(
                  "", 2 * sd))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",
                   default="/workspace/results/benchmark/fase6/raw_results.csv")
    p.add_argument("--repeat-csv",
                   default="/workspace/results/benchmark/fase6_repeat/raw_results.csv")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    d = load(args.csv)
    report_e3(d)
    report_decomposition(d)
    report_e2(args.repeat_csv, args.reps)


if __name__ == "__main__":
    main()
