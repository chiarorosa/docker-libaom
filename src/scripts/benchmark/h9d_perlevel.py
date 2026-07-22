#!/usr/bin/env python3
"""H9d Etapa 3, Fase 2b -- tau por NIVEL (16/32/64) supera o tau global?

A Fase 2 usou tau global (0.30). Mas os limiares theta que perdem a MESMA fracao
de vencedores diferem muito por nivel (Etapa 1, 39-feat): a winners_lost~10%,
theta_64=0.014, theta_32=0.103, theta_16=0.091. Um tau global e agressivo demais
no 32px -- onde EXT mais vence (base 12%), logo onde mora o custo de BD (e a leve
perda do RiverBank na Fase 2). Este sweep testa configs CALIBRADAS por nivel
(perda de vencedores consistente por nivel, protegendo o 32px sensivel).

Tudo vs o mesmo anchor nativo pristino (h9d_ub, 5fr), binario libaom_extoff_ml.
Referencias no mesmo plano: curva de tau H9a (h9d_tau) e tau global do H9d
(h9d_selective).
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
WORK = "/workspace/results/benchmark/h9d_perlevel"
NATIVE_CSV = "/workspace/results/benchmark/h9d_ub/raw.csv"

# theta por nivel (16/32/64), do modelo 39-feat da Etapa 1.
CONFIGS = {
    # calibrado winners_lost~10% (perda consistente por nivel)
    "PL10": {"16": "0.091", "32": "0.103", "64": "0.014"},
    # calibrado winners_lost~20% (mais agressivo, perda consistente)
    "PL20": {"16": "0.178", "32": "0.163", "64": "0.032"},
    # agressivo onde a predicao e confiavel/barata (16, AUC 0.92), protege o
    # 32px (EXT comum) e mantem o 64 conservador.
    "PLmix": {"16": "0.20", "32": "0.08", "64": "0.03"},
}


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
    os.makedirs(WORK, exist_ok=True)
    native = load_native(NATIVE_CSV)
    res = {name: {} for name in CONFIGS}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("config,seq,cq,bits,psnr_y,wall_s\n")
        for name, taus in CONFIGS.items():
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                env = dict(H9A_PREF)
                env["AV1_STUDENT_H9D_ENABLE"] = "1"
                for lvl, t in taus.items():
                    env["AV1_STUDENT_H9D_TAU_" + lvl] = t
                e = res[name].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "%s_%s_cq%d.obu" % (name, seq, cq))
                    bits, psnr, dt = encode(ENC, sf, fps, cq, FRAMES, out, 0, env)
                    cf.write("%s,%s,%d,%d,%.4f,%.2f\n" % (name, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  %-6s %-10s cq%-2d %6.1fs" % (name, seq, cq, dt), flush=True)

    print("\n=== Fase 2b: H9d tau por nivel (vs nativo, 5fr) ===")
    print("%-11s %-7s %9s %9s" % ("seq", "config", "BD-rate%", "speedup"))
    for seq in SEQ_LIST:
        nat = native[seq]; tn = nat["t"]
        for name in CONFIGS:
            c = res[name][seq]
            bd = bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"])
            print("%-11s %-7s %+9.3f %8.3fx" % (seq, name, bd, tn / c["t"]))
        print("-" * 42)


if __name__ == "__main__":
    main()
