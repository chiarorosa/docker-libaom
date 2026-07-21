#!/usr/bin/env python3
"""C2 — de onde vem o ganho do H9a, por NIVEL de bloco (16/32/64)?

Sweep de variaveis de ambiente (custo zero, sem recompilar). O H9a poda em tres
niveis; desligar um nivel = setar os tres taus daquele nivel para nao disparar
nenhuma acao. Semantica (partition_strategy.c:2142-2148, cadeia if/elif):
    NONE-commit : probs[0] > tau_none   -> desligar com tau_none = 2.0 (probs<=1)
    SPLIT-force : probs[1] > tau_split  -> desligar com tau_split = 2.0
    REST-off    : probs[2] < tau_rest   -> desligar com tau_rest = -1 (nao 2.0!)

Leave-one-out a partir do P_ref (ponto de referencia implantado): P_ref menos cada
nivel isola a CONTRIBUICAO MARGINAL daquele nivel (speedup e BD-rate perdidos ao
desliga-lo). Tudo vs o MESMO anchor nativo pristino (h9d_ub, 5fr), mesmo binario
libaom_extoff_ml (PARTITION_ML_STUDENT=1).
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
WORK = "/workspace/results/benchmark/c2_levels"
NATIVE_CSV = "/workspace/results/benchmark/h9d_ub/raw.csv"  # anchor pristino 5fr

P_REF = {"AV1_STUDENT_TAU_NONE_16": "0.85", "AV1_STUDENT_TAU_NONE_32": "0.80",
         "AV1_STUDENT_TAU_NONE_64": "0.80", "AV1_STUDENT_TAU_SPLIT": "0.90",
         "AV1_STUDENT_TAU_REST_16": "0.20", "AV1_STUDENT_TAU_REST_32": "0.20",
         "AV1_STUDENT_TAU_REST_64": "0.20"}


def off_level(lvl):
    """P_ref com o nivel lvl (16/32/64) desligado."""
    e = dict(P_REF)
    e["AV1_STUDENT_TAU_NONE_%d" % lvl] = "2.0"
    e["AV1_STUDENT_TAU_SPLIT_%d" % lvl] = "2.0"
    e["AV1_STUDENT_TAU_REST_%d" % lvl] = "-1"
    return e


VARIANTS = {
    "P_ref": P_REF,
    "off16": off_level(16),
    "off32": off_level(32),
    "off64": off_level(64),
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
    res = {v: {} for v in VARIANTS}
    with open(os.path.join(WORK, "raw.csv"), "w") as cf:
        cf.write("variant,seq,cq,bits,psnr_y,wall_s\n")
        for vname, env in VARIANTS.items():
            for seq in SEQ_LIST:
                sf, fps = SEQS[seq]
                e = res[vname].setdefault(seq, {"bits": [], "psnr": [], "t": 0.0})
                for cq in CQS:
                    out = os.path.join(WORK, "%s_%s_cq%d.obu" % (vname, seq, cq))
                    bits, psnr, dt = encode(ENC, sf, fps, cq, FRAMES, out, 0, env)
                    cf.write("%s,%s,%d,%d,%.4f,%.2f\n" % (vname, seq, cq, bits, psnr, dt))
                    cf.flush()
                    e["bits"].append(bits); e["psnr"].append(psnr); e["t"] += dt
                    print("  %-7s %-10s cq%-2d %6.1fs" % (vname, seq, cq, dt), flush=True)

    print("\n=== C2: contribuicao por nivel (vs nativo, 5fr) ===")
    print("%-11s %-8s %9s %9s" % ("seq", "variant", "BD-rate%", "speedup"))
    for seq in SEQ_LIST:
        nat = native[seq]; tnat = nat["t"]
        vals = {}
        for v in VARIANTS:
            c = res[v][seq]
            bd = bd_rate(nat["bits"], nat["psnr"], c["bits"], c["psnr"])
            sp = tnat / c["t"]
            vals[v] = (bd, sp)
            tag = "  <-- todos" if v == "P_ref" else ""
            print("%-11s %-8s %+9.3f %8.3fx%s" % (seq, v, bd, sp, tag))
        # contribuicao marginal = P_ref menos leave-one-out
        bp, sp_ref = vals["P_ref"]
        print("   contribuicao marginal de cada nivel (P_ref - offL):")
        for lvl, v in [(16, "off16"), (32, "off32"), (64, "off64")]:
            bd_l, sp_l = vals[v]
            print("     nivel %2d: speedup %+.3fx  BD %+.3f pp" %
                  (lvl, sp_ref - sp_l, bp - bd_l))
        print("-" * 46)


if __name__ == "__main__":
    main()
