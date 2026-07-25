#!/usr/bin/env python3
"""Fase 6 — CTC final-results encoder for the H9d lever (results chapter).

Runs the H9d post-NONE extended-partition pruner in its deployed configuration
over the CTC A1 test set, under the exact AOM-CTC All-Intra protocol already
used for the H9a points (encode_ctc.py). Rows are appended to the same
raw_results.csv so report_ctc.py compares everything against the same anchor.

    ml_bal_h9d   H9a P_rect (TAU_BALANCED) + H9d PL10 defaults   cpu-used=0

H9d is a *complement* to the deployed H9a pruner, never a substitute: the
balanced H9a policy is held fixed and H9d is stacked on top, so the delta
against the existing ml_balanced rows is the marginal contribution of H9d.

The H9d binary (build/libaom_perf_h9d) is built with the same flags as
build/libaom_perf (Release, -DPARTITION_ML_STUDENT=1, INTERNAL_STATS=0) so wall
times are comparable across configs. --integrity re-encodes one reference point
with H9d switched off and checks it reproduces the recorded ml_balanced row
byte-for-byte, proving the H9a base is unchanged in the new binary.

Reproduction:
    /workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d.py --integrity
    /workspace/build/venv-ml/bin/python src/scripts/fase6/ctc_h9d.py
"""

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from encode_ctc import (  # noqa: E402
    TAU_BALANCED, parse_y4m, encode, load_done, append_row,
)

# H9d on, thresholds left unset => the PL10 per-level defaults baked into
# student_h9d_get_tau (tau_16=0.091, tau_32=0.103, tau_64=0.014).
H9D_ENV = dict(TAU_BALANCED, AV1_STUDENT_H9D_ENABLE="1")

CONFIG_NAME = "ml_bal_h9d"

# Reference row for the integrity check: same seq/cq under ml_balanced.
INTEGRITY_SEQ, INTEGRITY_CQ = "BoxingPractice", 32


def read_row(csv_path, seq, config, cq):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["seq"], row["config"], int(row["cq"])) == (seq, config, cq):
                return row
    return None


def integrity_check(args, csv_path):
    """Encode the reference point with H9d OFF; must match the ml_balanced row."""
    ref = read_row(csv_path, INTEGRITY_SEQ, "ml_balanced", INTEGRITY_CQ)
    if ref is None:
        raise SystemExit("no ml_balanced reference row in " + csv_path)
    seqs = [f for f in sorted(os.listdir(args.seq_dir))
            if f.endswith(".y4m") and f.startswith(INTEGRITY_SEQ)]
    if not seqs:
        raise SystemExit("reference sequence not found in " + args.seq_dir)
    seq = os.path.join(args.seq_dir, seqs[0])
    _, _, _, _, bd = parse_y4m(seq)
    work = args.work or os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    out_obu = os.path.join(work, "_integrity.obu")

    print("integrity: {} cq{} on {} with H9d OFF".format(
        INTEGRITY_SEQ, INTEGRITY_CQ, args.h9d_enc), flush=True)
    dt, psnr_y = encode(args.h9d_enc, seq, INTEGRITY_CQ, args.frames, bd, 0,
                        dict(TAU_BALANCED), out_obu)
    nbytes = os.path.getsize(out_obu)
    os.remove(out_obu)

    ok_bytes = nbytes == int(ref["bytes"])
    ok_psnr = abs(psnr_y - float(ref["psnr_y"])) < 5e-4
    print("  bytes   got {:>9d}  ref {:>9d}  {}".format(
        nbytes, int(ref["bytes"]), "OK" if ok_bytes else "MISMATCH"), flush=True)
    print("  psnr_y  got {:.4f}    ref {:.4f}    {}".format(
        psnr_y, float(ref["psnr_y"]), "OK" if ok_psnr else "MISMATCH"), flush=True)
    print("  time    got {:.1f}s   ref {}s (informative)".format(
        dt, ref["time_s"]), flush=True)
    if not (ok_bytes and ok_psnr):
        raise SystemExit("INTEGRITY_FAIL: libaom_perf_h9d does not reproduce "
                         "the H9a baseline with H9d off")
    print("INTEGRITY_OK", flush=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir", default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir", default="/workspace/results/benchmark/fase6")
    p.add_argument("--h9d-enc", default="/workspace/build/libaom_perf_h9d/aomenc")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--seqs", nargs="+", default=None)
    p.add_argument("--work", default=None)
    p.add_argument("--integrity", action="store_true",
                   help="only run the H9d-off reproduction check and exit")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "raw_results.csv")

    if args.integrity:
        integrity_check(args, csv_path)
        return

    work = args.work or os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    done = load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    if args.seqs:
        seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no .y4m sequences in " + args.seq_dir)

    total = len(seqs) * len(args.cqs)
    print("CTC H9d ({}): {} seqs x {} cqs = {} encodes".format(
        CONFIG_NAME, len(seqs), len(args.cqs), total), flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            if (name, CONFIG_NAME, cq) in done:
                print("  cq{:>2} {:<12} (already done)".format(cq, CONFIG_NAME),
                      flush=True)
                continue
            out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                name, CONFIG_NAME, cq))
            dt, psnr_y = encode(args.h9d_enc, seq, cq, args.frames, bd, 0,
                                dict(H9D_ENV), out_obu)
            nbytes = os.path.getsize(out_obu)
            os.remove(out_obu)
            append_row(csv_path, {
                "seq": name, "config": CONFIG_NAME, "cq": cq,
                "fps_num": fps_num, "fps_den": fps_den,
                "frames": args.frames, "bytes": nbytes,
                "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
            })
            done.add((name, CONFIG_NAME, cq))
            print("  cq{:>2} {:<12} time={:7.1f}s  {:8d} B  PSNR-Y={:.4f} dB"
                  .format(cq, CONFIG_NAME, dt, nbytes, psnr_y), flush=True)

    print("\nCTC_H9D_ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()
