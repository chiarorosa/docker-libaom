#!/usr/bin/env python3
"""Fase 6 extension -- H9c-as-SOTA-substitute swap experiment.

The H9c analog of encode_swap.py. Deploys the post-NONE H9c student as a
drop-in replacement for libaom's native intra CNN partition pruner
(intra_cnn_based_part_prune), inside the fast presets cpu-used=1/2/3. At a
fixed cpu-used=N every other speed feature is identical; the two configs
differ ONLY in the partition pruner, so the (BD-rate, time) gap is
attributable purely to the pruner -- the clean ML-vs-ML comparison.

CRITICAL isolation detail: libaom_perf runs the H9a pre-search student BY
DEFAULT (tau_none/split default to 0.9). To make H9c the SOLE student pruner
(the honest analog of the H9a swap, where H9a was the sole substitute), H9a
is neutralized here: tau_none/tau_split=2.0 (softmax probs are in [0,1] so no
H9a branch ever fires) and tau_rest=-1 (rect-off already inert). Only H9c
then prunes.

We run, for N in {1,2,3} and the two best-efficiency H9c thresholds:

    h9c_tau90_cpuN   perf + CNN off + H9a neutralized + H9c tau=0.90
    h9c_tau95_cpuN   perf + CNN off + H9a neutralized + H9c tau=0.95

The native_cpuN reference (native CNN on) and the cpu-used=0 anchor are NOT
re-run: they are reused from the Fase 6 run (results/benchmark/fase6/
raw_results.csv), whose bitstream is deterministic and whose wall-clock is
comparable (dedicated host/container). The H9a swap for comparison lives in
results/benchmark/fase6_swap/.

Same AOM-CTC All-Intra encode command as encode_ctc.py; reuses its helpers.
Resumable via raw_results.csv.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from encode_ctc import (  # noqa: E402  reuse the vetted Fase 6 primitives
    parse_y4m, encode, load_done, append_row,
)

# Native intra CNN off (so the student is the only intra partition pruner).
CNN_OFF = {"AV1_DISABLE_NATIVE_CNN": "1"}
# H9a pre-search student neutralized: tau_none/split=2 never fire (softmax
# probs <= 1), tau_rest=-1 never fires -> H9a is fully inert, H9c is sole pruner.
H9A_OFF = {"AV1_STUDENT_TAU_NONE": "2", "AV1_STUDENT_TAU_SPLIT": "2",
           "AV1_STUDENT_TAU_REST": "-1"}
# The two best-efficiency H9c operating points from the cq20 sweep.
H9C_TAUS = [("90", "0.90"), ("95", "0.95")]


def configs_swap(ml_enc, levels):
    """(name, encoder, env, cpu_used) for the H9c swap experiment."""
    cfgs = []
    for n in levels:
        for tag, tau in H9C_TAUS:
            env = dict(CNN_OFF, **H9A_OFF)
            env["AV1_STUDENT_H9C_ENABLE"] = "1"
            env["AV1_STUDENT_H9C_TAU"] = tau
            cfgs.append(("h9c_tau{}_cpu{}".format(tag, n), ml_enc, env, n))
    return cfgs


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_swap_h9c")
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
    cfgs = configs_swap(args.ml_enc, args.levels)

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("Fase 6 SWAP H9c: {} seqs x {} configs x {} cqs = {} encodes".format(
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
                    print("  cq{:>2} {:<16} [skip, done]".format(cq, cname),
                          flush=True)
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
                print("  cq{:>2} {:<16} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nFASE6_SWAP_H9C_ENCODE_DONE ({} rows in {})".format(
        len(done), csv_path), flush=True)


if __name__ == "__main__":
    main()
