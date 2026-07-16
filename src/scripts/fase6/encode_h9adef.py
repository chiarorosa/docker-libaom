#!/usr/bin/env python3
"""H9a@default ALONE at cpu-used=0 -- the missing decomposition baseline.

The earlier h9c_tau* runs measured H9a@0.9/0.9 + H9c stacked (see ANDAMENTO
S8.1). To cleanly attribute 'H9a alone vs H9a+H9c vs H9c alone vs anchor', we
need H9a@default running ALONE (H9c off). That is exactly libaom_perf with an
EMPTY env: the H9a pre-search student runs unconditionally at its compiled-in
defaults (tau_none/split=0.9), and H9c stays off (AV1_STUDENT_H9C_ENABLE unset).

Appends h9adef rows to results/benchmark/fase6/raw_results.csv (shared anchor).
cpu-used=0, so the native intra CNN never runs (cpu>=1 speed feature) -- the
only pruner active is H9a@default, matching what ran beneath the h9c_tau* rows.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import encode_ctc as base  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seqs", nargs="+", default=["Neon1224"])
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

    print("H9a@default ALONE cpu0: {} seqs x 1 config x {} cqs = {} encodes"
          .format(len(seqs), len(args.cqs), len(seqs) * len(args.cqs)),
          flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = base.parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            if (name, "h9adef", cq) in done:
                print("  cq{:>2} h9adef  [skip, done]".format(cq), flush=True)
                continue
            out_obu = os.path.join(work, "{}_h9adef_{}.obu".format(name, cq))
            # EMPTY env -> H9a@default only, H9c off.
            dt, psnr_y = base.encode(args.ml_enc, seq, cq, args.frames, bd, 0,
                                     {}, out_obu)
            nbytes = os.path.getsize(out_obu)
            os.remove(out_obu)
            base.append_row(csv_path, {
                "seq": name, "config": "h9adef", "cq": cq,
                "fps_num": fps_num, "fps_den": fps_den,
                "frames": args.frames, "bytes": nbytes,
                "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
            })
            done.add((name, "h9adef", cq))
            print("  cq{:>2} h9adef  time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                  .format(cq, dt, nbytes, psnr_y), flush=True)

    print("\nH9ADEF_ENCODE_DONE  (csv: {})".format(csv_path), flush=True)


if __name__ == "__main__":
    main()
