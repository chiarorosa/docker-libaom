#!/usr/bin/env python3
"""Fase 6 extension -- H9d stacked on the H9a native-CNN substitute (D2-lean).

Closes the gap declared in results/thesis/R6_analise_integrada.md §6.1: H9d was
only measured at cpu-used=0. Here the H9a student replaces libaom's native intra
CNN partition pruner at cpu-used=1/2/3 (as in encode_swap.py) and H9d is stacked
on top, so the marginal contribution of H9d is measured inside the real presets.

Both arms of every pair run on the SAME binary (build/libaom_perf_h9d), in the
SAME campaign, interleaved per (seq, cq, cpu): the H9a base is re-measured here
instead of reused from fase6_swap, so the H9d marginal is paired and free of
cross-day drift.

    h9a_bal_cpuN       CNN off + H9a balanced            (base, H9d off)
    h9a_bal_h9d_cpuN   CNN off + H9a balanced + H9d PL10  (deployed H9d point)

The aggressive H9a base is deliberately excluded: over it H9d added only
+0.17 pp at cpu-used=0 (R4_h9d.md §4.7), below the ~0.46 pp time resolution.

Integrity (free, 96 checks): the base rows must reproduce the bytes of the
matching h9a_bal_cpuN rows of results/benchmark/fase6_swap/raw_results.csv,
since H9d off must be inert. --check-integrity reports it after (or during)
the campaign; one point was verified on 2026-10-03 (BoxingPractice cq32 cpu1,
1,572,268 B, PSNR-Y 40.9600).

Pre-registered predictions: results/thesis/IEEE_Conference_Template/ISCAS/CLAUDE.md §6.

Reproduction (container av1_bench):
    /workspace/build/venv-ml/bin/python src/scripts/fase6/encode_swap_h9d.py
    /workspace/build/venv-ml/bin/python src/scripts/fase6/encode_swap_h9d.py --check-integrity
"""

import argparse
import csv
import os
import resource
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from encode_ctc import (  # noqa: E402  reuse the vetted Fase 6 primitives
    TAU_BALANCED, CSV_FIELDS, parse_y4m, encode, load_done,
)

# CTC v9 §5.7 makes the *user time* of /usr/bin/time mandatory for runtime.
# getrusage(RUSAGE_CHILDREN) reads the same wait4() accounting /usr/bin/time
# reports, without touching the shared encode(); wall time (time_s) is kept
# so the rows stay comparable with every earlier Fase 6 campaign.
FIELDS = CSV_FIELDS + ["user_s", "sys_s"]


def append_row(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


CNN_OFF = {"AV1_DISABLE_NATIVE_CNN": "1"}
# H9d thresholds left unset -> baked-in PL10 per-level defaults, the deployed
# point (same convention as ctc_h9d.py, verified identical to explicit PL10).
H9D_ON = {"AV1_STUDENT_H9D_ENABLE": "1"}


def configs(levels):
    """(name, env, cpu_used); base first so each pair runs back to back."""
    cfgs = []
    for n in levels:
        base = dict(CNN_OFF, **TAU_BALANCED)
        cfgs.append(("h9a_bal_cpu{}".format(n), base, n))
        cfgs.append(("h9a_bal_h9d_cpu{}".format(n), dict(base, **H9D_ON), n))
    return cfgs


def check_integrity(csv_path, ref_path):
    """Compare re-measured base rows against fase6_swap byte counts."""
    def rows(path, prefix):
        with open(path, newline="") as f:
            return {(r["seq"], r["config"], r["cq"]): r
                    for r in csv.DictReader(f)
                    if r["config"].startswith(prefix)
                    and "_h9d_" not in r["config"]}
    new = rows(csv_path, "h9a_bal_cpu")
    ref = rows(ref_path, "h9a_bal_cpu")
    ok = bad = 0
    for k, r in sorted(new.items()):
        if k not in ref:
            print("  no reference for", k)
            continue
        same = (r["bytes"] == ref[k]["bytes"]
                and abs(float(r["psnr_y"]) - float(ref[k]["psnr_y"])) < 1e-4)
        ok += same
        bad += not same
        if not same:
            print("  MISMATCH", k, r["bytes"], ref[k]["bytes"])
    print("integrity: {} identical, {} mismatched, {} checked".format(
        ok, bad, ok + bad))
    return bad == 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-dir",
                   default="/workspace/src/samples/aomctc_test_set")
    p.add_argument("--out-dir",
                   default="/workspace/results/benchmark/fase6_swap_h9d")
    p.add_argument("--enc", default="/workspace/build/libaom_perf_h9d/aomenc")
    p.add_argument("--ref-csv",
                   default="/workspace/results/benchmark/fase6_swap/raw_results.csv")
    p.add_argument("--cqs", type=int, nargs="+", default=[20, 32, 43, 55])
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--seqs", nargs="+", default=None,
                   help="only sequences whose name contains these substrings")
    p.add_argument("--check-integrity", action="store_true",
                   help="only compare recorded base rows against --ref-csv")
    args = p.parse_args()

    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    if args.check_integrity:
        sys.exit(0 if check_integrity(csv_path, args.ref_csv) else 1)

    os.makedirs(args.out_dir, exist_ok=True)
    work = os.path.join(args.out_dir, "_work")
    os.makedirs(work, exist_ok=True)
    done = load_done(csv_path)

    seqs = sorted(f for f in os.listdir(args.seq_dir) if f.endswith(".y4m"))
    if args.seqs:
        seqs = [f for f in seqs if any(s in f for s in args.seqs)]
    if not seqs:
        raise SystemExit("no .y4m sequences in " + args.seq_dir)
    cfgs = configs(args.levels)

    total = len(seqs) * len(cfgs) * len(args.cqs)
    print("Fase 6 SWAP+H9d: {} seqs x {} configs x {} cqs = {} encodes".format(
        len(seqs), len(cfgs), len(args.cqs), total), flush=True)
    print("already done: {}/{}".format(len(done), total), flush=True)

    for sf in seqs:
        seq = os.path.join(args.seq_dir, sf)
        name = sf.split("_")[0]
        w, h, fps_num, fps_den, bd = parse_y4m(seq)
        print("\n########## {}  ({}x{}, {:.3f} fps, {}-bit) ##########".format(
            name, w, h, fps_num / fps_den, bd), flush=True)
        for cq in args.cqs:
            for cname, env, cpu in cfgs:
                if (name, cname, cq) in done:
                    continue
                out_obu = os.path.join(work, "{}_{}_{}.obu".format(
                    name, cname, cq))
                ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
                dt, psnr_y = encode(args.enc, seq, cq, args.frames, bd, cpu,
                                    env, out_obu)
                ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
                nbytes = os.path.getsize(out_obu)
                os.remove(out_obu)
                append_row(csv_path, {
                    "seq": name, "config": cname, "cq": cq,
                    "fps_num": fps_num, "fps_den": fps_den,
                    "frames": args.frames, "bytes": nbytes,
                    "psnr_y": round(psnr_y, 4), "time_s": round(dt, 3),
                    "user_s": round(ru1.ru_utime - ru0.ru_utime, 3),
                    "sys_s": round(ru1.ru_stime - ru0.ru_stime, 3),
                })
                done.add((name, cname, cq))
                print("  cq{:>2} {:<18} wall={:7.1f}s user={:7.1f}s  {:8d} B  "
                      "PSNR-Y={:.4f} dB".format(cq, cname, dt,
                                                ru1.ru_utime - ru0.ru_utime,
                                                nbytes, psnr_y), flush=True)

    print("\nFASE6_SWAP_H9D_ENCODE_DONE ({} rows in {})".format(
        len(done), csv_path), flush=True)


if __name__ == "__main__":
    main()
