#!/usr/bin/env python3
"""H9d Etapa 3, Fase 2 -- confirmacao a >=10 quadros do vencedor da Fase 1.

Compara, a 10 quadros (rigor de resultado E, nao sonda), no MESMO plano vs
nativo, quatro configs por seq/CQ:
  - native            : encoder pristino (anchor), binario libaom_extoff (sem estudante)
  - H9a(P_ref)        : pruner implantado, base
  - H9a A2_none70     : subir o tau do H9a ate ~mesmo speedup do H9d(0.30) -- o
                        competidor da curva de tau
  - H9a+H9d(tau=0.30) : o seletivo no ponto de operacao intermediario

Pergunta: ao ~mesmo speedup, H9a+H9d custa MENOS BD-rate que H9a A2 (subir tau)?
Se sim, a vitoria da Fase 1 (RaceNight/Jockey) e sinal, nao ruido de 5 quadros.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bd_rate import bd_rate                       # noqa: E402
from h9d_upper_bound import SEQS, encode, H9A_PREF  # noqa: E402

ENC_NATIVE = "/workspace/build/libaom_extoff/aomenc"     # estudante OFF
ENC_ML = "/workspace/build/libaom_extoff_ml/aomenc"      # H9a + H9d
FRAMES = 10
CQS = [20, 32, 43, 55]
SEQ_LIST = ["Jockey", "RaceNight", "RiverBank"]
WORK = "/workspace/results/benchmark/h9d_phase2"

A2 = {"AV1_STUDENT_TAU_NONE": "0.70", "AV1_STUDENT_TAU_SPLIT": "0.90",
      "AV1_STUDENT_TAU_REST": "0.30"}
H9D = dict(H9A_PREF); H9D["AV1_STUDENT_H9D_ENABLE"] = "1"
H9D["AV1_STUDENT_H9D_TAU"] = "0.30"

# nome -> (binario, env)
CONFIGS = [
    ("native", ENC_NATIVE, {}),
    ("h9a", ENC_ML, dict(H9A_PREF)),
    ("h9a_a2", ENC_ML, A2),
    ("h9a_h9d", ENC_ML, H9D),
]


def main():
    os.makedirs(WORK, exist_ok=True)
    res = {name: {} for name, _, _ in CONFIGS}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("config,seq,cq,bits,psnr_y,wall_s\n")
        for name, enc, env in CONFIGS:
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                e = res[name].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "%s_%s_cq%d.obu" % (name, seq, cq))
                    bits, psnr, dt = encode(enc, sf, fps, cq, FRAMES, out, 0, env)
                    cf.write("%s,%s,%d,%d,%.4f,%.2f\n" % (name, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  %-9s %-10s cq%-2d %6.1fs" % (name, seq, cq, dt), flush=True)

    print("\n=== Etapa 3 F2 (10fr): H9a+H9d vs subir tau (A2), vs nativo ===")
    print("%-11s %-9s %9s %9s" % ("seq", "config", "BD-rate%", "speedup"))
    for seq in SEQ_LIST:
        nat = res["native"][seq]; tn = nat["t"]
        for name in ["h9a", "h9a_a2", "h9a_h9d"]:
            c = res[name][seq]
            bd = bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"])
            tag = "  <-- H9d" if name == "h9a_h9d" else ""
            print("%-11s %-9s %+9.3f %8.3fx%s" % (seq, name, bd, tn / c["t"], tag))
        print("-" * 44)


if __name__ == "__main__":
    main()
