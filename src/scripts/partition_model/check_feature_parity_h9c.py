#!/usr/bin/env python3
"""C <-> Python parity check for the H9c (post-NONE) student feature extractor.

Forked from check_feature_parity.py (the H9a pre-search harness). Encodes one
frame with the PARTITION_ML_STUDENT build while AV1_STUDENT_H9C_ENABLE=1
AV1_STUDENT_H9C_DUMP=<path> makes student_h9c_decide (partition_strategy.c)
append one 208-byte record per scored node:
  head[10] (int32 n, mi_row, mi_col, qindex, neigh_avail, above_bsize,
  left_bsize, dc_q, frame_w, frame_h)   -- 40 bytes
  feats[39] (float32, A+B+C+E)          -- 156 bytes
  probs[3]  (float32, P(NONE/SPLIT/REST)) -- 12 bytes

SCOPE / KNOWN LIMITATION (read before trusting a green run):
Only features 0..35 (blocks A, B, C -- identical to H9a) are independently
verified here, by rebuilding the RD context from `head` and recomputing the
vector with features.node_features_h9c(...)[:36] from the same source pixels.
Features 36..38 (block E: log1p(none_rate), log1p(none_dist),
log1p(none_rdcost)) are NOT independently reconstructable by this harness:
`head` does not carry the raw PARTITION_NONE rate/dist/rdcost as separate
int64 fields, only their already-log1p'd values baked into feats[36:39]. The
C side computes them from part_state->this_rdc (the real RD_STATS the live
encoder just produced for PARTITION_NONE at this node) and there is no
independent Python code path that reproduces that RD search. Features 36-38
are therefore verified BY CONSTRUCTION (same log1p applied to the same
RD_STATS fields the C decision itself consumes), not by cross-implementation
recomputation. This is a known, accepted harness limitation -- not a gap to
work around -- so the final report line says "PARITY OK (0..35)", never
claiming full 0..38 parity.

The optional PyTorch probs check reuses this same convention: it feeds the
net a hybrid vector (Python-recomputed pf[:36] concatenated with the C dump's
own cf[36:39]) so it can still test net-arithmetic parity (Python vs the
encoder's av1_nn_predict/softmax) without pretending to have independently
verified the E-block inputs.

Run inside the container, e.g.:
  AV1_STUDENT_H9C_ENABLE=1 AV1_STUDENT_H9C_DUMP=/tmp/h9c_parity.bin \
      build/libaom_ml_check/aomenc ...
  venv-ml/bin/python check_feature_parity_h9c.py \
      --dump /tmp/h9c_parity.bin --students results/models/student_h9c/students.pt
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402

# head[10] ++ feats[39] ++ probs[3] = 208 bytes/record.
REC = struct.Struct("<10i{}f".format(featmod.NUM_FEATURES_H9C + 3))
NCMP = 36  # only blocks A+B+C (0..35) are independently reconstructable here


def make_yuv(path, w, h, seed=0):
    """Textured synthetic 4:2:0 frame: smooth gradients + noise + hard edges,
    so every feature (variance, gradients, band structure) is exercised."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    y = (96 + 64 * np.sin(xx / 17.0) * np.cos(yy / 23.0) +
         rng.normal(0, 24, size=(h, w)))
    y[:, w // 3:w // 3 + 8] = 235  # vertical bar (VERT cue)
    y[h // 2:h // 2 + 6, :] = 16   # horizontal bar (HORZ cue)
    y = np.clip(y, 0, 255).astype(np.uint8)
    uv = np.full((h // 2, w // 2), 128, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(y.tobytes())
        f.write(uv.tobytes())
        f.write(uv.tobytes())
    return y


def read_y(path, w, h):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(w * h), dtype=np.uint8).reshape(h, w)


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aomenc", default="/workspace/build/libaom_ml_check/aomenc")
    p.add_argument("--yuv", default="/tmp/parity_448x256.yuv",
                   help="source; synthesized if missing (WxH parsed from name)")
    p.add_argument("--dump", default="/tmp/parity_h9c_feats.bin")
    p.add_argument("--cq", type=int, default=32)
    p.add_argument("--cpu-used", type=int, default=0)
    p.add_argument("--atol", type=float, default=2e-6,
                   help="max |C - Python| accepted (float32 ulp headroom)")
    p.add_argument("--students", default="/workspace/results/models/"
                                        "student_h9c/students.pt",
                   help="if set, also verify C probs vs this PyTorch student")
    args = p.parse_args(argv)

    import re
    m = re.search(r"(\d+)x(\d+)", os.path.basename(args.yuv))
    if not m:
        raise SystemExit("cannot parse WxH from " + args.yuv)
    w, h = int(m.group(1)), int(m.group(2))
    if os.path.exists(args.yuv):
        y = read_y(args.yuv, w, h)
    else:
        y = make_yuv(args.yuv, w, h)

    if os.path.exists(args.dump):
        os.remove(args.dump)
    env = dict(os.environ, AV1_STUDENT_H9C_ENABLE="1",
              AV1_STUDENT_H9C_DUMP=args.dump)
    cmd = [args.aomenc, "--usage=2", "--passes=1", "--threads=1",
           "--cpu-used={}".format(args.cpu_used), "--end-usage=q",
           "--cq-level={}".format(args.cq), "-w", str(w), "-h", str(h),
           "--limit=1", "-o", "/tmp/parity_h9c.ivf", args.yuv]
    r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="replace")[-500:] + "\n")
        raise SystemExit("encode failed")

    blob = open(args.dump, "rb").read()
    if len(blob) % REC.size:
        raise SystemExit("dump size {} not a multiple of record size {}".format(
            len(blob), REC.size))
    n_rec = len(blob) // REC.size
    if n_rec == 0:
        raise SystemExit("no records: is AV1_STUDENT_H9C_ENABLE=1 and the "
                         "build PARTITION_ML_STUDENT=1?")

    # Optional: PyTorch students to re-score the dumped C features and compare
    # the resulting 3-class probs against the encoder's own probs.
    nets = None
    if args.students and os.path.exists(args.students):
        import torch
        import torch.nn.functional as F
        bundle = torch.load(args.students, map_location="cpu")
        nets = {}
        nfeat = bundle.get("num_features", featmod.NUM_FEATURES_H9C)
        for dim in bundle["students"]:
            net = studentmod.make_student(nfeat, bundle["hidden"])
            net.load_state_dict(bundle["students"][dim])
            nets[dim] = net.eval()

    NF = featmod.NUM_FEATURES_H9C
    worst = np.zeros(NCMP)
    worst_prob = np.zeros(3)
    seen = set()
    checked = 0
    for i in range(n_rec):
        vals = REC.unpack_from(blob, i * REC.size)
        (n, mi_row, mi_col, qindex, neigh_avail, above_bsize, left_bsize,
         dc_q, frame_w, frame_h) = vals[:10]
        cf = np.array(vals[10:10 + NF], dtype=np.float32)
        cprobs = np.array(vals[10 + NF:], dtype=np.float32)
        key = (n, mi_row, mi_col)
        if key in seen:
            continue  # rd_pick_partition may revisit a node; one check suffices
        seen.add(key)
        sb_r, sb_c = (mi_row & ~15) * 4, (mi_col & ~15) * 4
        sb = y[sb_r:sb_r + 64, sb_c:sb_c + 64]
        r_cell = (mi_row & 15) // (n // 4)
        c_cell = (mi_col & 15) // (n // 4)
        ctx = {"neigh_avail": neigh_avail, "above_bsize": above_bsize,
               "left_bsize": left_bsize, "dc_q": dc_q, "mi_row": mi_row,
               "mi_col": mi_col, "frame_w": frame_w, "frame_h": frame_h}
        # Only 0..35 (A+B+C) are independently reconstructable -- see module
        # docstring for why 36..38 (block E) cannot be recomputed here.
        pf = featmod.node_features_h9c(sb, n, r_cell, c_cell, qindex, ctx)[:NCMP]
        worst = np.maximum(worst, np.abs(cf[:NCMP].astype(np.float64) -
                                         pf.astype(np.float64)))
        if nets is not None and n in nets:
            import torch
            import torch.nn.functional as F
            # Hybrid vector: Python-recomputed 0..35 + the C dump's own
            # 36..38 (not independently verified, taken verbatim -- see
            # module docstring). This still exercises net-arithmetic parity.
            hybrid = np.concatenate([pf, cf[NCMP:NF]]).astype(np.float32)
            with torch.no_grad():
                logits = nets[n](torch.tensor(hybrid, dtype=torch.float32)[None])
                pprobs = F.softmax(logits, dim=-1)[0].numpy()
            worst_prob = np.maximum(worst_prob, np.abs(
                cprobs.astype(np.float64) - pprobs.astype(np.float64)))
        checked += 1

    print("checked {} unique nodes ({} records)".format(checked, n_rec))
    bad = 0
    names = featmod.H9_FEATURE_NAMES
    for k in range(NCMP):
        flag = ""
        if worst[k] > args.atol:
            flag = "  <-- MISMATCH"
            bad += 1
        print("  [{:>2}] {:<18} max|dC-dPy| = {:.3e}{}".format(
            k, names[k], worst[k], flag))
    if nets is not None:
        print("--- probs (C encoder vs PyTorch student, hybrid features) ---")
        prob_atol = 1e-3  # nn_predict prec-reduce + float32 headroom
        for k, nm in enumerate(["P(NONE)", "P(SPLIT)", "P(REST)"]):
            flag = "  <-- MISMATCH" if worst_prob[k] > prob_atol else ""
            print("  {:<8} max|dC-dPy| = {:.3e}{}".format(nm, worst_prob[k],
                                                          flag))
            if worst_prob[k] > prob_atol:
                bad += 1
    if bad:
        raise SystemExit("PARITY FAILED: {} channel(s) above tolerance".format(
            bad))
    print("PARITY OK (0..35) (atol={})".format(args.atol))


if __name__ == "__main__":
    main(sys.argv[1:])
