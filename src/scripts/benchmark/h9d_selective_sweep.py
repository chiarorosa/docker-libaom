#!/usr/bin/env python3
"""H9d Etapa 3, Fase 1 -- o seletivo domina a curva de tau do H9a?

Sweep do tau do H9d (AV1_STUDENT_H9D_TAU) sobre a base H9a P_ref, no MESMO plano
de 5 quadros / mesmo anchor nativo das etapas anteriores. Sobrepoe:
  - curva de tau do H9a (h9d_tau/raw.csv: P_ref, A1, A2, A3);
  - blanket H9a+extoff (h9d_marg/raw.csv, arm h9a_extoff);
  - H9a sozinho (P_ref) e o novo H9a+H9d seletivo.
Pergunta: os pontos H9a+H9d_seletivo ficam ACIMA/A ESQUERDA da curva de tau
(mais speedup ao mesmo BD, ou menos BD ao mesmo speedup)? Se sim, o seletivo e
uma contribuicao real -- 2a solucao positiva. Confirmacao final (>=10 quadros)
e a Fase 2.

Binario libaom_extoff_ml (PARTITION_ML_STUDENT=1 + H9d). Anchor nativo pristino:
h9d_ub/raw.csv (arm native, build sem estudante).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bd_rate import bd_rate                       # noqa: E402
from h9d_upper_bound import SEQS, encode, H9A_PREF  # noqa: E402

ENC = "/workspace/build/libaom_extoff_ml/aomenc"
FRAMES = 5
CQS = [20, 32, 43, 55]
SEQ_LIST = ["Jockey", "RaceNight", "RiverBank"]
WORK = "/workspace/results/benchmark/h9d_selective"
NATIVE_CSV = "/workspace/results/benchmark/h9d_ub/raw.csv"
TAU_CURVE_CSV = "/workspace/results/benchmark/h9d_tau/raw.csv"  # curva tau H9a
MARG_CSV = "/workspace/results/benchmark/h9d_marg/raw.csv"      # blanket

H9D_TAUS = [0.05, 0.10, 0.20, 0.30, 0.45]


def load(path, key, want_col=None, want_val=None):
    d = {}
    for r in csv.DictReader(open(path)):
        if want_col and r.get(want_col) != want_val:
            continue
        k = (r[key], r["seq"]) if key != "seq" else r["seq"]
        e = d.setdefault(k, {"bits": [], "psnr": [], "t": 0.0})
        e["bits"].append(float(r["bits"]))
        e["psnr"].append(float(r["psnr_y"]))
        e["t"] += float(r["wall_s"])
    return d


def bd_sp(nat, c):
    return (bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"]),
            nat["t"] / c["t"])


def main():
    os.makedirs(WORK, exist_ok=True)
    native = load(NATIVE_CSV, "seq", "arm", "native")

    # roda o sweep do H9d seletivo
    res = {t: {} for t in H9D_TAUS}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("h9d_tau,seq,cq,bits,psnr_y,wall_s\n")
        for t in H9D_TAUS:
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                env = dict(H9A_PREF)
                env["AV1_STUDENT_H9D_ENABLE"] = "1"
                env["AV1_STUDENT_H9D_TAU"] = str(t)
                e = res[t].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "h9d%.2f_%s_cq%d.obu" % (t, seq, cq))
                    bits, psnr, dt = encode(ENC, sf, fps, cq, FRAMES, out, 0, env)
                    cf.write("%.2f,%s,%d,%d,%.4f,%.2f\n" % (t, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  h9d_tau%.2f %-10s cq%-2d %6.1fs" % (t, seq, cq, dt), flush=True)

    # referencias existentes (mesmo plano 5fr)
    tau_curve = load(TAU_CURVE_CSV, "point")          # H9a: P_ref/A1/A2/A3
    blanket = load(MARG_CSV, "seq", "arm", "h9a_extoff")
    h9a_base = load(MARG_CSV, "seq", "arm", "h9a")

    print("\n=== Etapa 3 F1: H9a+H9d seletivo vs curva de tau do H9a (5fr) ===")
    for seq in SEQ_LIST:
        nat = native[seq]
        print("\n--- {} ---  config                 BD-rate%  speedup".format(seq))
        # H9a base
        bd, sp = bd_sp(nat, h9a_base[seq])
        print("   H9a(P_ref)                 %+8.3f  %6.3fx" % (bd, sp))
        # curva de tau H9a
        for pt in ["A1_none80", "A2_none70", "A3_none60_rest40"]:
            if (pt, seq) in tau_curve:
                bd, sp = bd_sp(nat, tau_curve[(pt, seq)])
                print("   H9a tau %-18s %+8.3f  %6.3fx" % (pt, bd, sp))
        # blanket
        bd, sp = bd_sp(nat, blanket[seq])
        print("   H9a+extoff (blanket)       %+8.3f  %6.3fx" % (bd, sp))
        # H9d seletivo
        for t in H9D_TAUS:
            bd, sp = bd_sp(nat, res[t][seq])
            print("   H9a+H9d tau=%.2f            %+8.3f  %6.3fx  <--" % (t, bd, sp))


if __name__ == "__main__":
    main()
