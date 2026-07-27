#!/usr/bin/env python3
"""Fase 6 — CTC final-results encoder for the H9d lever (results chapter).

Runs the H9d post-NONE extended-partition pruner in its deployed configuration
over the CTC A1 test set, under the exact AOM-CTC All-Intra protocol already
used for the H9a points (encode_ctc.py). Rows are appended to the same
raw_results.csv so report_ctc.py compares everything against the same anchor.

    ml_bal_h9d        H9a P_rect + H9d PL10   (the deployed point)
    ml_bal_h9d_pl20   H9a P_rect + H9d PL20
    ml_aggr_h9d       H9a A3     + H9d PL10
    ml_aggr_h9d_pl20  H9a A3     + H9d PL20

H9d is a *complement* to the deployed H9a pruner, never a substitute: the H9a
policy is held fixed and H9d is stacked on top, so the delta against the
matching ml_balanced/ml_aggr rows is the marginal contribution of H9d.

The four points exist to turn H9d from a single measurement into a CURVE. With
only ml_bal_h9d, H9d is one point while H9a is a whole tau curve, so "H9d beats
the tau knob" rests on one operating point; crossing two H9a bases with two H9d
strengths lets the two mechanisms be compared as surfaces.

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
    TAU_BALANCED, TAU_AGGRESSIVE, parse_y4m, encode, load_done, append_row,
)

# Per-level H9d thresholds, taken from the operating points of the Etapa-1 gate
# (results/models/h9d_predictability_h9c/run.log). PL10/PL20 = the theta that
# costs ~10%/~20% of true-EXT winners at each block size.
#                 16px     32px     64px
PL10 = ("0.0910", "0.1031", "0.0144")  # == the C defaults; kept for documentation
PL20 = ("0.1776", "0.1627", "0.0321")


def _h9d(base, per_level=None):
    """H9a base policy + H9d on. `per_level=None` leaves the thresholds unset,
    so student_h9d_get_tau falls back to its baked-in PL10 defaults (verified
    identical to PL10 explicit, RESULTADOS_H9d_etapa3_encoder.md:112)."""
    env = dict(base, AV1_STUDENT_H9D_ENABLE="1")
    if per_level:
        env["AV1_STUDENT_H9D_TAU_16"] = per_level[0]
        env["AV1_STUDENT_H9D_TAU_32"] = per_level[1]
        env["AV1_STUDENT_H9D_TAU_64"] = per_level[2]
    return env


# The H9d frontier: two H9a operating points (P_rect, A3) x two H9d strengths
# (PL10, PL20). ml_bal_h9d is the already-measured deployed point; the other
# three turn the single H9d point into a curve comparable to the H9a tau knob.
CONFIGS = {
    "ml_bal_h9d":       _h9d(TAU_BALANCED),            # P_rect x PL10 (deployed)
    "ml_bal_h9d_pl20":  _h9d(TAU_BALANCED, PL20),      # P_rect x PL20
    "ml_aggr_h9d":      _h9d(TAU_AGGRESSIVE),          # A3     x PL10
    "ml_aggr_h9d_pl20": _h9d(TAU_AGGRESSIVE, PL20),    # A3     x PL20
}

# Reference points for the integrity check. The balanced one proves the H9a
# base is unchanged in this binary; the aggressive one matters because
# TAU_AGGRESSIVE is exercised in libaom_perf_h9d for the FIRST time here.
INTEGRITY_SEQ, INTEGRITY_CQ = "BoxingPractice", 32
INTEGRITY_REFS = [
    ("ml_balanced", TAU_BALANCED),
    ("ml_aggr", TAU_AGGRESSIVE),
]


def read_row(csv_path, seq, config, cq):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["seq"], row["config"], int(row["cq"])) == (seq, config, cq):
                return row
    return None


def integrity_check(args, csv_path):
    """Encode each reference point with H9d OFF; must reproduce the recorded row
    byte-for-byte. Proves the H9a base is untouched in the H9d binary, for BOTH
    tau presets used by the frontier."""
    seqs = [f for f in sorted(os.listdir(args.seq_dir))
            if f.endswith(".y4m") and f.startswith(INTEGRITY_SEQ)]
    if not seqs:
        raise SystemExit("reference sequence not found in " + args.seq_dir)
    seq = os.path.join(args.seq_dir, seqs[0])
    _, _, _, _, bd = parse_y4m(seq)
    work = args.work or os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    out_obu = os.path.join(work, "_integrity.obu")

    failed = []
    for ref_config, tau in INTEGRITY_REFS:
        ref = read_row(csv_path, INTEGRITY_SEQ, ref_config, INTEGRITY_CQ)
        if ref is None:
            raise SystemExit("no {} reference row in {}".format(
                ref_config, csv_path))
        print("\nintegrity: {} cq{} vs '{}' with H9d OFF".format(
            INTEGRITY_SEQ, INTEGRITY_CQ, ref_config), flush=True)
        dt, psnr_y = encode(args.h9d_enc, seq, INTEGRITY_CQ, args.frames, bd, 0,
                            dict(tau), out_obu)
        nbytes = os.path.getsize(out_obu)
        os.remove(out_obu)

        ok_bytes = nbytes == int(ref["bytes"])
        ok_psnr = abs(psnr_y - float(ref["psnr_y"])) < 5e-4
        print("  bytes   got {:>9d}  ref {:>9d}  {}".format(
            nbytes, int(ref["bytes"]), "OK" if ok_bytes else "MISMATCH"),
            flush=True)
        print("  psnr_y  got {:.4f}    ref {:.4f}    {}".format(
            psnr_y, float(ref["psnr_y"]), "OK" if ok_psnr else "MISMATCH"),
            flush=True)
        print("  time    got {:.1f}s   ref {}s (informative)".format(
            dt, ref["time_s"]), flush=True)
        if not (ok_bytes and ok_psnr):
            failed.append(ref_config)

    if failed:
        raise SystemExit("INTEGRITY_FAIL: libaom_perf_h9d does not reproduce "
                         "the H9a baseline with H9d off for: " +
                         ", ".join(failed))
    print("\nINTEGRITY_OK", flush=True)


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
    p.add_argument("--configs", nargs="+", default=list(CONFIGS),
                   choices=list(CONFIGS),
                   help="which frontier points to encode (default: all four; "
                        "ml_bal_h9d is already measured and will be skipped)")
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

    todo = [(name, cfg, cq)
            for name in [s.split("_")[0] for s in seqs]
            for cq in args.cqs
            for cfg in args.configs
            if (name, cfg, cq) not in done]
    print("CTC H9d: {} seqs x {} cqs x {} configs = {} encodes, {} pending"
          .format(len(seqs), len(args.cqs), len(args.configs),
                  len(seqs) * len(args.cqs) * len(args.configs), len(todo)),
          flush=True)

    n = 0
    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        # Configs are interleaved WITHIN each (seq, cq) so that any thermal or
        # load drift hits every arm alike -- the same reasoning that made the
        # repeatability probe (E2) interleave its repetitions.
        for cq in args.cqs:
            for cfg in args.configs:
                if (name, cfg, cq) in done:
                    print("  cq{:>2} {:<17} (already done)".format(cq, cfg),
                          flush=True)
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cfg, cq))
                dt, psnr_y = encode(args.h9d_enc, seq, cq, args.frames, bd, 0,
                                    dict(CONFIGS[cfg]), out_obu)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                append_row(csv_path, {
                    "seq": name, "config": cfg, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                })
                done.add((name, cfg, cq))
                n += 1
                print("  cq{:>2} {:<17} time={:7.1f}s  {:8d} B  "
                      "PSNR-Y={:.4f} dB  [{}/{}]".format(
                          cq, cfg, dt, nbytes, psnr_y, n, len(todo)), flush=True)

    print("\nCTC_H9D_ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()
