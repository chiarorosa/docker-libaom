#!/usr/bin/env python3
"""H9c CTC head-to-head at cq=20 only.

Appends h9c_tau90/h9c_tau95 rows to the SAME results/benchmark/fase6/
raw_results.csv the Fase 6 CTC run already populated (anchor, ml_balanced,
ml_aggr, native_cpu1/2/3), so report_ctc.py-style analysis can compare
anchor vs H9a (2 profiles) vs H9c side by side without re-encoding anything
already measured. Reuses encode_ctc.py's encode/CSV plumbing verbatim.

Only cq=20 is run (the user's ask): isolate the highest-quality point,
where the H9a-vs-H9c pilot gap was narrowest, before ruling out H9c.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import encode_ctc as base  # noqa: E402

H9C_TAUS = {
    "h9c_tau30": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.30"},
    # E3: sonda do joelho, na faixa inexplorada tau em (30, 60).
    "h9c_tau45": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.45"},
    "h9c_tau60": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.60"},
    "h9c_tau70": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.70"},
    "h9c_tau90": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.90"},
    "h9c_tau95": {"AV1_STUDENT_H9C_ENABLE": "1", "AV1_STUDENT_H9C_TAU": "0.95"},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--ml-enc", default="/workspace/build/libaom_perf/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seqs", nargs="+", default=None,
                   help="only sequences whose filename contains one of these "
                        "substrings (default: all .y4m in --seq-dir)")
    p.add_argument("--taus", nargs="+", default=None,
                   help="only these H9c thresholds, by tag (e.g. 90 95); "
                        "default: all of " + " ".join(
                            k[len("h9c_tau"):] for k in H9C_TAUS))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    done = base.load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    if args.seqs:
        seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no .y4m sequences in " + args.seq_dir)
    cfgs = list(H9C_TAUS.items())
    if args.taus:
        want = {"h9c_tau" + t for t in args.taus}
        cfgs = [c for c in cfgs if c[0] in want]
        if not cfgs:
            raise SystemExit("no H9c threshold matches " + " ".join(args.taus))

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("H9c CTC cq20 head-to-head: {} seqs x {} configs x {} cqs = {} encodes"
          .format(len(seqs), len(cfgs), len(args.cqs), total), flush=True)
    print("already done: {}/{}".format(
        sum(1 for s in seqs for cn, _ in cfgs for cq in args.cqs
            if (s.split("_")[0], cn, cq) in done), total), flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = base.parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, env in cfgs:
                if (name, cname, cq) in done:
                    print("  cq{:>2} {:<12} [skip, already done]".format(
                        cq, cname), flush=True)
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
                print("  cq{:>2} {:<12} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                      .format(cq, cname, dt, nbytes, psnr_y), flush=True)

    print("\nH9C_CQ20_ENCODE_DONE  (csv: {})".format(csv_path), flush=True)


if __name__ == "__main__":
    main()
