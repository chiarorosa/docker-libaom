#!/usr/bin/env python3
"""H9c ISOLATED head-to-head at cpu-used=0 (H9a neutralized).

Confound check for the earlier encode_h9c_cq20.py runs. Those set ONLY the
H9c env (AV1_STUDENT_H9C_ENABLE / _TAU) and left the H9a pre-search student
at its compiled-in defaults (tau_none/split=0.9), so libaom_perf pruned with
H9a@0.9/0.9 + H9c stacked -- NOT H9c alone.

This script re-runs the same sequence/cqs/cpu0 with the H9a student
NEUTRALIZED (tau_none/split=2 never fire; tau_rest=-1 already inert), so ONLY
H9c prunes. The single changed variable vs the earlier runs is the H9a
neutralization; the anchor, binary, cpu-used, cqs, and frame count are
identical. The delta between the earlier h9c_tau{N} rows (H9a active) and
these h9ciso_tau{N} rows (H9a inert) is exactly the H9a@default contribution
that was wrongly attributed to H9c.

Rows are appended to the SAME results/benchmark/fase6/raw_results.csv (shared
anchor for BD-rate), under distinct config names h9ciso_tau{95,90,60}. No
AV1_DISABLE_NATIVE_CNN: the native intra CNN is a cpu>=1 speed feature, off at
cpu-used=0, so it never runs here (matching the earlier runs exactly).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import encode_ctc as base  # noqa: E402

# H9a pre-search student neutralized (softmax probs <=1 so tau=2 never fires;
# tau_rest=-1 never fires) -> only H9c prunes.
H9A_OFF = {"AV1_STUDENT_TAU_NONE": "2", "AV1_STUDENT_TAU_SPLIT": "2",
           "AV1_STUDENT_TAU_REST": "-1"}
# The three thresholds the earlier confounded runs used for Neon1224.
H9C_TAUS = [("95", "0.95"), ("90", "0.90"), ("60", "0.60")]


def configs_iso(taus=None):
    cfgs = []
    for tag, tau in H9C_TAUS:
        if taus and tag not in taus:
            continue
        env = dict(H9A_OFF)
        env["AV1_STUDENT_H9C_ENABLE"] = "1"
        env["AV1_STUDENT_H9C_TAU"] = tau
        cfgs.append(("h9ciso_tau{}".format(tag), env))
    return cfgs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seqs", nargs="+", default=["Neon1224"],
                   help="sequences whose filename contains these substrings")
    p.add_argument("--taus", nargs="+", default=None,
                   help="only these H9c thresholds, by tag (e.g. 90); "
                        "default: all of " + " ".join(t for t, _ in H9C_TAUS))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = base.load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no matching .y4m sequences in " + args.seq_dir)
    cfgs = configs_iso(args.taus)
    if not cfgs:
        raise SystemExit("no H9c threshold matches " + " ".join(args.taus))

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("H9c ISOLATED cpu0 (H9a neutralized): {} seqs x {} configs x {} cqs "
          "= {} encodes".format(len(seqs), len(cfgs), len(args.cqs), total),
          flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = base.parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, env in cfgs:
                if (name, cname, cq) in done:
                    print("  cq{:>2} {:<14} [skip, done]".format(cq, cname),
                          flush=True)
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cname, cq))
                dt, psnr_y = base.encode(args.ml_enc, seq, cq, args.frames,
                                         bd, 0, env, out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                base.append_row(csv_path, {
                    "seq": name, "config": cname, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, cname, cq))
                print("  cq{:>2} {:<14} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nH9C_ISO_ENCODE_DONE  (csv: {})".format(csv_path), flush=True)


if __name__ == "__main__":
    main()
