#!/usr/bin/env python3
"""Cota superior do ganho do H9d (poda de particao estendida AB+4-way).

Blanket-disable de AB/4-way (env AV1_EXT_PART_OFF=1) vs nativo (env unset), no
MESMO binario `libaom_extoff` -> isola o efeito de remover a busca estendida.
Mede, por sequencia de teste, o par (BD-rate perdido, tempo economizado):

  - BD-rate(extoff vs nativo) > 0  => custo de qualidade de remover AB/4-way
  - speedup = t_nativo / t_extoff   => teto de aceleracao (blanket, sem selecao)

Um H9d REAL (seletivo) fica DENTRO deste envelope: menos speedup que o blanket,
mas menos BD-rate. A cota superior de speedup e o blanket; o custo de BD-rate do
blanket e o pior caso que um H9d bom deve evitar. Se o BD-rate do blanket ja for
pequeno, o H9d e facil (quase de graca); se for grande, exige selecao fina.

Regime identico ao encode_ctc.py (cpu-used=0, All-Intra, end-usage=q).
Rate para BD-rate = bits totais do fluxo (tamanho do arquivo). PSNR-Y da linha
--psnr. Ambos os bracos: mesmos N quadros, mesma fps -> comparacao valida.
"""
import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bd_rate import bd_rate  # noqa: E402

SAMPLES = "/workspace/src/samples"
# seq -> (arquivo, fps)  (todas 3840x2160 8bit 420)
SEQS = {
    "Jockey": ("Jockey_3840x2160_120fps_420_8bit_YUV_RAW.yuv", "120/1"),
    "RaceNight": ("RaceNight_3840x2160_50fps_420_8bit_YUV_RAW.yuv", "50/1"),
    "RiverBank": ("RiverBank_3840x2160_50fps_420_8bit_YUV_RAW.yuv", "50/1"),
}
PSNR_RE = re.compile(r"PSNR \(Overall/Avg/Y/U/V\)\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([\d.]+)")

# Ponto de operacao de referencia do H9a implantado (P_ref em h7h8_bench.py).
# Aplicado a AMBOS os bracos quando --stack h9a, para medir o ganho MARGINAL
# de desligar AB/4-way EM CIMA do H9a operante (H9d como estagio pos-NONE).
H9A_PREF = {
    "AV1_STUDENT_TAU_NONE_16": "0.85", "AV1_STUDENT_TAU_NONE_32": "0.80",
    "AV1_STUDENT_TAU_NONE_64": "0.80", "AV1_STUDENT_TAU_SPLIT": "0.90",
    "AV1_STUDENT_TAU_REST_16": "0.20", "AV1_STUDENT_TAU_REST_32": "0.20",
    "AV1_STUDENT_TAU_REST_64": "0.20",
}


def encode(enc, seq_file, fps, cq, frames, out, ext_off, base_env=None):
    cmd = [
        enc, "-w", "3840", "-h", "2160", "--fps=" + fps, "--input-bit-depth=8",
        "--cpu-used=0", "--passes=1", "--end-usage=q", "--cq-level=%d" % cq,
        "--kf-min-dist=0", "--kf-max-dist=0", "--deltaq-mode=0",
        "--enable-tpl-model=0", "--enable-keyframe-filtering=0",
        "--tile-columns=1", "--tile-rows=0", "--threads=2", "--row-mt=0",
        "--bit-depth=8", "--limit=%d" % frames, "--psnr", "--obu",
        "-o", out, os.path.join(SAMPLES, seq_file),
    ]
    env = dict(os.environ)
    if base_env:
        env.update(base_env)
    if ext_off:
        env["AV1_EXT_PART_OFF"] = "1"
    else:
        env.pop("AV1_EXT_PART_OFF", None)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-600:] + "\n")
        raise SystemExit("encode failed: cq%d ext_off=%d" % (cq, ext_off))
    m = PSNR_RE.search(r.stderr.decode(errors="replace"))
    if not m:
        raise SystemExit("no PSNR line for cq%d ext_off=%d" % (cq, ext_off))
    psnr_y = float(m.group(3))
    bits = os.path.getsize(out) * 8
    return bits, psnr_y, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="/workspace/build/libaom_extoff/aomenc")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    ap.add_argument("--seqs", nargs="+", default=list(SEQS))
    ap.add_argument("--work", default="/workspace/results/benchmark/h9d_ub")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--stack", choices=["none", "h9a"], default="none",
                    help="none: nativo vs extoff (teto isolado). "
                         "h9a: H9a(P_ref) vs H9a+extoff (marginal pos-NONE).")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    csv_path = args.csv or os.path.join(args.work, "raw.csv")

    base_env = H9A_PREF if args.stack == "h9a" else {}
    base_name = "h9a" if args.stack == "h9a" else "native"
    off_name = "h9a_extoff" if args.stack == "h9a" else "extoff"

    rows = []
    with open(csv_path, "w") as cf:
        cf.write("seq,cq,arm,bits,psnr_y,wall_s\n")
        for seq in args.seqs:
            seq_file, fps = SEQS[seq]
            for cq in args.cqs:
                for ext_off in (0, 1):
                    arm = off_name if ext_off else base_name
                    out = os.path.join(args.work, "%s_cq%d_%s.obu" %
                                       (seq, cq, arm))
                    bits, psnr_y, dt = encode(args.enc, seq_file, fps, cq,
                                              args.frames, out, ext_off,
                                              base_env)
                    cf.write("%s,%d,%s,%d,%.4f,%.2f\n" %
                             (seq, cq, arm, bits, psnr_y, dt))
                    cf.flush()
                    rows.append((seq, cq, arm, bits, psnr_y, dt))
                    print("  %-10s cq%-2d %-6s  %8.1f kb  PSNR-Y %.3f  %6.1fs" %
                          (seq, cq, arm, bits / 8 / 1000, psnr_y, dt),
                          flush=True)

    # --- agregacao ---
    title = ("marginal H9d pos-NONE (H9a+extoff vs H9a)" if args.stack == "h9a"
             else "cota superior H9d (blanket AB/4-way off vs nativo)")
    print("\n=== %s ===" % title)
    print("%-11s %10s %10s %12s" % ("seq", "BD-rate%", "speedup",
                                    "t_base/t_off"))
    tot_b = tot_o = 0.0
    bdrates = []
    for seq in args.seqs:
        pts = {arm: ([], []) for arm in (base_name, off_name)}
        tb = to = 0.0
        for (s, cq, arm, bits, psnr_y, dt) in rows:
            if s != seq:
                continue
            pts[arm][0].append(bits)
            pts[arm][1].append(psnr_y)
            if arm == base_name:
                tb += dt
            else:
                to += dt
        bd = bd_rate(pts[base_name][0], pts[base_name][1],
                     pts[off_name][0], pts[off_name][1])
        bdrates.append(bd)
        tot_b += tb
        tot_o += to
        print("%-11s %+10.3f %9.3fx %12s" %
              (seq, bd, tb / to if to else float("nan"),
               "%.1f/%.1f" % (tb, to)))
    import statistics
    print("-" * 46)
    print("%-11s %+10.3f %9.3fx" %
          ("MEDIA", statistics.mean(bdrates),
           tot_b / tot_o if tot_o else float("nan")))
    if args.stack == "h9a":
        print("\nLeitura: BD-rate/speedup MARGINAL de desligar AB/4-way em cima "
              "do H9a operante (P_ref). Se marginal ~ ruido, H9d pos-NONE morre; "
              "se material a baixo BD, um H9d seletivo estilo H9c vale construir.")
    else:
        print("\nLeitura: BD-rate>0 = custo de qualidade de remover AB/4-way; "
              "speedup = teto (blanket). H9d seletivo fica dentro do envelope.")


if __name__ == "__main__":
    main()
