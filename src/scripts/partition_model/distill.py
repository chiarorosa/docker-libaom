#!/usr/bin/env python3
"""Distill the multi-level ConvNeXt surrogate into per-block-size student MLPs.

For every partition node in the data we pair:
  * handcrafted features of that block's own luma (features.py), and
  * the surrogate's per-cell soft prediction at the matching quadtree level,
collapsed to 3 classes [NONE, SPLIT, REST]. Each block-size student is trained
with alpha*CE(hard truth) + (1-alpha)*KD(soft teacher), then exported to a C
header (export_weights.py) and consumed by av1_nn_predict inside libaom.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from partition_defs import LEVELS, NUM_PARTITION_TYPES  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from model import PartitionSurrogate  # noqa: E402


def build_distill_set(entries, surrogate, device, batch=256, limit=None,
                      verbose=True):
    """Per-block-size distillation arrays.
    Returns {dim: {"feat":(N,F), "teacher":(N,3), "truth":(N,)}}."""
    out = {dim: {"feat": [], "teacher": [], "truth": []} for dim, _ in LEVELS}
    buf_inputs, buf_members, buf_qidx = [], [], []

    def flush():
        if not buf_inputs:
            return
        x = torch.from_numpy(np.stack(buf_inputs)).to(device)
        with torch.no_grad():
            logits = surrogate(x)
            probs = {dim: F.softmax(logits[dim].float(), dim=-1).cpu().numpy()
                     for dim, _ in LEVELS}
        for bi, members in enumerate(buf_members):
            for dim, r, c, luma, label in members:
                teacher10 = probs[dim][bi, r, c]
                out[dim]["feat"].append(
                    featmod.block_features(luma, buf_qidx[bi]))
                out[dim]["teacher"].append(studentmod.collapse_probs(teacher10))
                out[dim]["truth"].append(studentmod.collapse_label(label))
        buf_inputs.clear()
        buf_members.clear()
        buf_qidx.clear()

    n_sb = 0
    for e in entries:
        for sb in datamod.iter_superblock_members(e["path"]):
            luma = sb["luma"].astype(np.float32) / 255.0
            q = sb["qindex"] / 255.0
            qplane = np.full((64, 64), q, dtype=np.float32)
            buf_inputs.append(np.stack([luma, qplane]))
            buf_members.append(sb["members"])
            buf_qidx.append(sb["qindex"])
            n_sb += 1
            if len(buf_inputs) >= batch:
                flush()
            if limit and n_sb >= limit:
                break
        if verbose:
            print("[distill] {}: {} superblocks so far".format(
                os.path.basename(e["path"]), n_sb))
        if limit and n_sb >= limit:
            break
    flush()

    for dim, _ in LEVELS:
        out[dim] = {k: np.asarray(v) for k, v in out[dim].items()}
    return out


def class_weights_3(truth, device):
    c = np.bincount(truth, minlength=studentmod.NUM_STUDENT_CLASSES).astype(
        np.float64)
    pos = c > 0
    w = np.ones(studentmod.NUM_STUDENT_CLASSES)
    w[pos] = c[pos].sum() / (c[pos] * pos.sum())
    w = np.clip(w, 0.1, 10.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_student(rec, hidden, device, epochs, lr, alpha, temp, wd=1e-4,
                  batch=4096, use_class_weight=True):
    feat = torch.tensor(rec["feat"], dtype=torch.float32)
    teach = torch.tensor(rec["teacher"], dtype=torch.float32)
    truth = torch.tensor(rec["truth"], dtype=torch.long)
    # Standardize features (folded into the first layer at the end, so C feeds
    # raw features). Floor std at 0.1: our features are log-scaled or bounded, so
    # real std >> 0.1; a tiny floor would explode W/std for a near-constant
    # feature (e.g. q_norm within a single QP) and wreck float32 precision.
    mean = feat.mean(0)
    std = feat.std(0).clamp_min(1e-1)
    feat = (feat - mean) / std

    net = studentmod.make_student(featmod.NUM_FEATURES, hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # Class weighting counters imbalance but suppresses confident P(NONE); for a
    # confidence-thresholded pruner, unweighted CE gives sharper, better-ranked
    # probabilities (the threshold tau, not the loss, handles the tradeoff).
    cw = class_weights_3(rec["truth"], device) if use_class_weight else None
    n = len(feat)
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            fb = feat[idx].to(device)
            tb = teach[idx].to(device)
            yb = truth[idx].to(device)
            logits = net(fb)
            ce = F.cross_entropy(logits, yb, weight=cw)
            log_p = F.log_softmax(logits / temp, dim=-1)
            teacher_pT = F.softmax(torch.log(tb.clamp_min(1e-8)) / temp, dim=-1)
            kd = F.kl_div(log_p, teacher_pT, reduction="batchmean") * (temp ** 2)
            loss = alpha * ce + (1 - alpha) * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
    # Fold standardization into the first linear layer so C sees raw features:
    #   (W (x-mean)/std + b) = (W/std) x + (b - W (mean/std))
    with torch.no_grad():
        first = next(m for m in net.modules() if isinstance(m, torch.nn.Linear))
        w = first.weight
        first.bias -= (w @ (mean.to(device) / std.to(device)))
        first.weight.copy_(w / std.to(device))
    return net, {"mean": mean.numpy(), "std": std.numpy()}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset")
    p.add_argument("--surrogate", default="/workspace/results/models/surrogate/"
                                          "surrogate_best.pt")
    p.add_argument("--val-seqs", nargs="+", default=["Jockey"])
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--out-dir", default="/workspace/results/models/student")
    p.add_argument("--hidden", type=int, nargs="+", default=[24, 16])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hard-CE weight; (1-alpha) is KD weight")
    p.add_argument("--temp", type=float, default=3.0)
    p.add_argument("--no-class-weight", action="store_true",
                   help="drop student class weighting for sharper P(NONE) "
                        "(better for a confidence-thresholded pruner)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.surrogate, map_location=device)
    sargs = ckpt.get("args", {})
    surrogate = PartitionSurrogate(
        sargs.get("variant", "tiny"), sargs.get("fusion_dim", 128)).to(device)
    surrogate.load_state_dict(ckpt["model"])
    surrogate.eval()
    print("loaded surrogate:", args.surrogate,
          "(headline F1 {:.4f})".format(ckpt.get("headline_macro_f1", -1)))

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, _ = datamod.split_entries(entries, args.val_seqs, args.train_seqs)
    print("distilling from {} train pkl(s)".format(len(train_e)))
    data = build_distill_set(train_e, surrogate, device, limit=args.limit)

    os.makedirs(args.out_dir, exist_ok=True)
    bundle = {"hidden": args.hidden, "students": {}, "norm": {}}
    for dim, _ in LEVELS:
        rec = data[dim]
        if len(rec["truth"]) == 0:
            print("[dim {}] no samples, skipping".format(dim))
            continue
        counts = np.bincount(rec["truth"], minlength=3).tolist()
        net, norm = train_student(rec, args.hidden, device, args.epochs,
                                  args.lr, args.alpha, args.temp,
                                  use_class_weight=not args.no_class_weight)
        # Quick self-check: student vs teacher argmax agreement.
        with torch.no_grad():
            fb = torch.tensor(rec["feat"], dtype=torch.float32, device=device)
            pred = net(fb).argmax(-1).cpu().numpy()
        teach_arg = rec["teacher"].argmax(-1)
        agree = float((pred == teach_arg).mean())
        acc = float((pred == rec["truth"]).mean())
        print("[dim {:>2}] n={} counts(N/S/R)={} teacher-agree={:.3f} "
              "truth-acc={:.3f}".format(dim, len(pred), counts, agree, acc))
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm

    torch.save(bundle, os.path.join(args.out_dir, "students.pt"))
    print("Saved students ->", os.path.join(args.out_dir, "students.pt"))


if __name__ == "__main__":
    main(sys.argv[1:])
