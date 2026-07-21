#!/usr/bin/env python3
"""Compara, no MESMO setup de 5 quadros e MESMO anchor nativo, duas formas de
ganhar mais speedup a partir do H9a implantado:

  (a) subir o tau do H9a (curva P_ref -> A1 -> A2 -> A3) -- knob de graca;
  (b) empilhar poda de AB/4-way pos-NONE (H9a+extoff) -- o H9d marginal (blanket).

Se o ponto empilhado (b) fica ABAIXO/na curva de tau (a), o H9d blanket nao supera
o knob gratis. Se um H9d SELETIVO puder ficar ACIMA da curva, ai vale construir.
Este script mede a curva de tau; os pontos empilhados vem de h9d_marg/raw.csv.

Anchor nativo pristino: results/benchmark/h9d_ub/raw.csv (arm 'native').
Usa o binario libaom_extoff_ml (PARTITION_ML_STUDENT=1 + gate ext-off).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bd_rate import bd_rate            # noqa: E402
from h9d_upper_bound import SEQS, encode  # noqa: E402

ENC = "/workspace/build/libaom_extoff_ml/aomenc"
FRAMES = 5
CQS = [20, 32, 43, 55]
SEQ_LIST = ["Jockey", "RaceNight", "RiverBank"]
WORK = "/workspace/results/benchmark/h9d_tau"

# Pontos de operacao do H9a (tau), de h7h8_bench.py. P_ref = referencia
# implantada (per-level); A1..A3 = curva agressiva (global).
POINTS = {
    "P_ref": {"AV1_STUDENT_TAU_NONE_16": "0.85", "AV1_STUDENT_TAU_NONE_32": "0.80",
              "AV1_STUDENT_TAU_NONE_64": "0.80", "AV1_STUDENT_TAU_SPLIT": "0.90",
              "AV1_STUDENT_TAU_REST_16": "0.20", "AV1_STUDENT_TAU_REST_32": "0.20",
              "AV1_STUDENT_TAU_REST_64": "0.20"},
    "A1_none80": {"AV1_STUDENT_TAU_NONE": "0.80", "AV1_STUDENT_TAU_SPLIT": "0.90",
                  "AV1_STUDENT_TAU_REST": "0.20"},
    "A2_none70": {"AV1_STUDENT_TAU_NONE": "0.70", "AV1_STUDENT_TAU_SPLIT": "0.90",
                  "AV1_STUDENT_TAU_REST": "0.30"},
    "A3_none60_rest40": {"AV1_STUDENT_TAU_NONE": "0.60", "AV1_STUDENT_TAU_SPLIT": "0.85",
                         "AV1_STUDENT_TAU_REST": "0.40"},
}


def load_arm(path, arm):
    d = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] != arm:
            continue
        e = d.setdefault(r["seq"], {"bits": [], "psnr": [], "t": 0.0})
        e["bits"].append(float(r["bits"]))
        e["psnr"].append(float(r["psnr_y"]))
        e["t"] += float(r["wall_s"])
    return d


def main():
    os.makedirs(WORK, exist_ok=True)
    native = load_arm("/workspace/results/benchmark/h9d_ub/raw.csv", "native")
    marg = load_arm("/workspace/results/benchmark/h9d_marg/raw.csv", "h9a_extoff")

    # roda a curva de tau (sem ext-off)
    curve = {p: {} for p in POINTS}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("point,seq,cq,bits,psnr_y,wall_s\n")
        for pname, env in POINTS.items():
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                e = curve[pname].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "%s_%s_cq%d.obu" % (pname, seq, cq))
                    bits, psnr, dt = encode(ENC, sf, fps, cq, FRAMES, out, 0, env)
                    cf.write("%s,%s,%d,%d,%.4f,%.2f\n" % (pname, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  %-16s %-10s cq%-2d %6.1fs" % (pname, seq, cq, dt), flush=True)

    print("\n=== BD-rate x speedup vs MESMO anchor nativo (5fr) ===")
    print("%-11s %-16s %9s %9s" % ("seq", "config", "BD-rate%", "speedup"))
    for seq in SEQ_LIST:
        nat = native[seq]; tnat = nat["t"]
        for pname in POINTS:
            c = curve[pname][seq]
            bd = bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"])
            print("%-11s %-16s %+9.3f %8.3fx" % (seq, pname + " (tau)", bd, tnat / c["t"]))
        # ponto empilhado (H9a P_ref + blanket extoff)
        m = marg[seq]
        bd = bd_rate(nat["bits"], nat["psnr"], m["bits"], m["psnr"])
        print("%-11s %-16s %+9.3f %8.3fx  <-- empilhado" %
              (seq, "H9a+extoff", bd, tnat / m["t"]))
        print("-" * 50)


if __name__ == "__main__":
    main()
