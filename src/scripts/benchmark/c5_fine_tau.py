#!/usr/bin/env python3
"""C5 — probe de falsificacao: a fronteira BD-rate x speedup do limiar NONE
global do H9a ja e densa/continua num grid FINO de tau?

Sweep da env var global AV1_STUDENT_TAU_NONE num grid fino, com
AV1_STUDENT_TAU_SPLIT e AV1_STUDENT_TAU_REST FIXOS e SEM overrides por nivel
(_16/_32/_64) -- o tau global se aplica uniformemente a todos os niveis. Se o
sweep discreto de tau ja produz uma fronteira (BD-rate, speedup) densa e sem
saltos, um C5 de modulacao suave por P(NONE) dificilmente adiciona valor: o
limiar rigido (hard-threshold) ja varre o espaco de operacao continuamente.
Se houver lacunas (saltos grandes de BD-rate/speedup entre taus vizinhos), ha
espaco para a modulacao suave preencher a fronteira.

Regime identico aos scripts irmaos (cpu-used=0, All-Intra, end-usage=q),
mesmo binario libaom_extoff_ml (PARTITION_ML_STUDENT=1), mesmo anchor nativo
pristino do h9d_ub (5fr).
"""
import argparse
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
WORK = "/workspace/results/benchmark/c5_finetau"
NATIVE_CSV = "/workspace/results/benchmark/h9d_ub/raw.csv"  # anchor pristino 5fr

TAU_GRID = [0.55, 0.62, 0.68, 0.74, 0.80, 0.86, 0.92, 0.96]


def base_env_for(tau):
    """Global uniforme: mesmo tau_none em todos os niveis, sem overrides _16/_32/_64."""
    return {"AV1_STUDENT_TAU_NONE": str(tau), "AV1_STUDENT_TAU_SPLIT": "0.90",
             "AV1_STUDENT_TAU_REST": "0.20"}


def load_native(path):
    d = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] != "native":
            continue
        e = d.setdefault(r["seq"], {"bits": [], "psnr": [], "t": 0.0})
        e["bits"].append(float(r["bits"]))
        e["psnr"].append(float(r["psnr_y"]))
        e["t"] += float(r["wall_s"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    native = load_native(NATIVE_CSV)
    res = {tau: {} for tau in TAU_GRID}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("tau,seq,cq,bits,psnr_y,wall_s\n")
        for tau in TAU_GRID:
            env = base_env_for(tau)
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                e = res[tau].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "tau%.2f_%s_cq%d.obu" % (tau, seq, cq))
                    bits, psnr, dt = encode(ENC, sf, fps, cq, args.frames, out, 0, env)
                    cf.write("%.2f,%s,%d,%d,%.4f,%.2f\n" % (tau, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  tau%.2f %-10s cq%-2d %6.1fs" % (tau, seq, cq, dt), flush=True)

    print("\n=== C5: fronteira BD-rate x speedup por tau_none global (vs nativo, 5fr) ===")
    print("%-11s %-6s %9s %9s" % ("seq", "tau", "BD-rate%", "speedup"))
    for seq in SEQ_LIST:
        nat = native[seq]; tnat = nat["t"]
        for tau in TAU_GRID:
            c = res[tau][seq]
            bd = bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"])
            sp = tnat / c["t"]
            print("%-11s %-6.2f %+9.3f %8.3fx" % (seq, tau, bd, sp))
        print("-" * 46)


if __name__ == "__main__":
    main()
