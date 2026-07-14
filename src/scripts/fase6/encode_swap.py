#!/usr/bin/env python3
"""Fase 6 extension -- H9a-as-SOTA-substitute swap experiment.

Tests deploying the distilled H9a student as a drop-in replacement for libaom's
native intra CNN partition pruner (intra_cnn_based_part_prune), inside the real
fast presets cpu-used=1/2/3 where partition pruners actually operate. At a fixed
cpu-used=N every other speed feature is identical; the two configs differ ONLY
in the partition pruner, so the (BD-rate, time) gap is attributable purely to the
pruner -- the clean ML-vs-ML comparison the thesis needs.

At cpu-used>=1 libaom turns the native CNN on (speed feature, level 2). We run:

    h9a_bal_cpuN    perf + AV1_DISABLE_NATIVE_CNN=1     CNN off, student balanced
    h9a_aggr_cpuN   perf + AV1_DISABLE_NATIVE_CNN=1     CNN off, student aggressive

for N in {1,2,3}. The native_cpuN reference (native CNN on) is NOT re-run: it is
reused from the Fase 6 run, whose bitstream is deterministic and whose wall-clock
is comparable because the host/container is dedicated to this task. report_swap.py
reads native_cpuN and the cpu-used=0 anchor from the Fase 6 CSV
(results/benchmark/fase6/raw_results.csv).

Same AOM-CTC All-Intra encode command as encode_ctc.py; reuses its helpers.
Resumable via raw_results.csv. Prepared but NOT auto-launched.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from encode_ctc import (  # noqa: E402  reuse the vetted Fase 6 primitives
    TAU_BALANCED, TAU_AGGRESSIVE, parse_y4m, encode, load_done, append_row,
)

CNN_OFF = {"AV1_DISABLE_NATIVE_CNN": "1"}


def configs_swap(anchor_enc, ml_enc, levels):
    """(name, encoder, env, cpu_used) for the swap experiment.

    Only the H9a swap configs run here; the native_cpuN reference is reused from
    the Fase 6 run (deterministic bitstream; the dedicated host/container keeps
    wall-clock stable across runs, so its timing is comparable). report_swap.py
    pulls native_cpuN and the cpu0 anchor from results/benchmark/fase6.
    """
    cfgs = []
    for n in levels:
        cfgs.append(("h9a_bal_cpu{}".format(n), ml_enc,
                     dict(CNN_OFF, **TAU_BALANCED), n))
        cfgs.append(("h9a_aggr_cpu{}".format(n), ml_enc,
                     dict(CNN_OFF, **TAU_AGGRESSIVE), n))
    return cfgs


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_swap")
    p.add_argument("--anchor-enc",
                   default="/workspace/build/libaom_perf_anchor/aomenc")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3],
                   help="cpu-used levels to test (default 1 2 3)")
    p.add_argument("--seqs", nargs="+", default=None,
                   help="only sequences whose name contains these substrings")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    if args.seqs:
        seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no .y4m sequences in " + args.seq_dir)
    cfgs = configs_swap(args.anchor_enc, args.ml_enc, args.levels)

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("Fase 6 SWAP: {} seqs x {} configs x {} cqs = {} encodes".format(
        len(seqs), len(cfgs), len(args.cqs), total), flush=True)
    print("already done: {}/{}".format(len(done), total), flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, enc, env, cpu in cfgs:
                if (name, cname, cq) in done:
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cname, cq))
                dt, psnr_y = encode(enc, seq, cq, args.frames, bd, cpu, env,
                                    out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                append_row(csv_path, {
                    "seq": name, "config": cname, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, cname, cq))
                print("  cq{:>2} {:<14} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nFASE6_SWAP_ENCODE_DONE ({} rows in {})".format(len(done),
          csv_path), flush=True)


if __name__ == "__main__":
    main()
