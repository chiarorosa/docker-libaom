#!/usr/bin/env python3
"""Fase 6 extension -- H9a(conservative) + H9c COMBINED swap (frontier check).

The analysis (ANDAMENTO S8 / global Pareto) showed: H9c-swap alone matches the
native CNN at cpu1/2, H9a-swap alone is worse (higher BD), and stacking H9c on
H9a adds ~0 TS but slightly lowers BD (Neon1224 hint). This tests the one
untested lever: can a CONSERVATIVE H9a + H9c, combined as the native-CNN
substitute, push past the native frontier?

At fixed cpu-used=N: native CNN off, and BOTH students active --
  - H9a pre-search, conservative (prunes only very-high-confidence blocks, so
    BD stays low): tau_none/split high, tau_rest=-1 (no rect-off).
  - H9c post-NONE at 0.90.
Two H9a conservatism levels map a short trajectory:
  h9acomb98_cpuN : H9a tau_none=0.98 tau_split=0.95   (most conservative)
  h9acomb95_cpuN : H9a tau_none=0.95 tau_split=0.92   (moderate)

Compared against native_cpuN, h9c_tau9x_cpuN and h9a_*_cpuN already measured for
the same sequence. native_cpuN and the cpu0 anchor are reused from Fase 6.
Expected outcome (prior): does NOT beat native -- levers exploit correlated
signal (information ceiling). This is the empirical confirmation.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from encode_ctc import parse_y4m, encode, load_done, append_row  # noqa: E402

CNN_OFF = {"AV1_DISABLE_NATIVE_CNN": "1"}
H9C_ON = {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.90"}
# (tag, H9a tau_none, H9a tau_split) -- tau_rest=-1 (rect-off off) throughout.
H9A_LEVELS = [("98", "0.98", "0.95"), ("95", "0.95", "0.92")]


def configs_combo(levels):
    cfgs = []
    for n in levels:
        for tag, tn, tsp in H9A_LEVELS:
            env = dict(CNN_OFF, **H9C_ON)
            env["AV1_STUDENT_TAU_NONE"] = tn
            env["AV1_STUDENT_TAU_SPLIT"] = tsp
            env["AV1_STUDENT_TAU_REST"] = "-1"
            cfgs.append(("h9acomb{}_cpu{}".format(tag, n), env, n))
    return cfgs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_swap_combo")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seqs", nargs="+", default=["Tango"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no matching .y4m sequences in " + args.seq_dir)
    cfgs = configs_combo(args.levels)

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("Fase 6 SWAP COMBO (H9a-cons + H9c): {} seqs x {} configs x {} cqs "
          "= {} encodes".format(len(seqs), len(cfgs), len(args.cqs), total),
          flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, env, cpu in cfgs:
                if (name, cname, cq) in done:
                    print("  cq{:>2} {:<18} [skip, done]".format(cq, cname),
                          flush=True)
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cname, cq))
                dt, psnr_y = encode(args.ml_enc, seq, cq, args.frames, bd, cpu,
                                    env, out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                append_row(csv_path, {
                    "seq": name, "config": cname, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, cname, cq))
                print("  cq{:>2} {:<18} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nFASE6_SWAP_COMBO_ENCODE_DONE (csv: {})".format(csv_path),
          flush=True)


if __name__ == "__main__":
    main()
