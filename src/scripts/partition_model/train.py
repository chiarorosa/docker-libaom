#!/usr/bin/env python3
"""Train the multi-level ConvNeXt-AV1 partition surrogate.

Loss is a masked, cost-sensitive cross-entropy per quadtree level:
  * masked  -- only quadtree cells an RD node actually visited are supervised;
  * cost-sensitive -- targets are geometric soft labels (partition_defs) so a
    confusion between similar partitions is penalized less than an opposite one;
  * class-weighted -- per-level inverse-frequency weights counter the ~91% NONE
    prior so the model cannot win by always predicting NONE.

Reported per level: top-1, macro-F1, per-class recall, confusion matrix. The
headline gate is macro-F1 / SPLIT-recall (never raw accuracy). Runs on the
container GPU; use --limit for a fast CPU/GPU smoke test on the reduced dataset.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from partition_defs import (  # noqa: E402
    MODEL_LEVELS, NUM_PARTITION_TYPES, PARTITION_NAMES, legality_mask,
    geometric_cost_matrix, geometric_soft_targets)
import data as datamod  # noqa: E402
from model import PartitionSurrogate  # noqa: E402


# --------------------------------------------------------------------------
# Loss helpers
# --------------------------------------------------------------------------
def build_level_targets(temp, eps, device):
    """Per-level (10,10) soft-target matrices, renormalized over legal classes."""
    cost = geometric_cost_matrix()
    soft = geometric_soft_targets(cost, temp=temp, eps=eps)  # (10,10)
    per_level = {}
    for dim, _ in MODEL_LEVELS:
        legal = legality_mask(dim)
        t = soft.copy()
        t[:, ~legal] = 0.0
        t = t / t.sum(axis=1, keepdims=True)
        per_level[dim] = torch.tensor(t, dtype=torch.float32, device=device)
    return per_level


def build_class_weights(counts, scheme, device, clip=10.0):
    """Per-level class weights from training label counts."""
    weights = {}
    for dim, _ in MODEL_LEVELS:
        c = counts[dim].astype(np.float64)
        legal = legality_mask(dim)
        w = np.ones(NUM_PARTITION_TYPES, dtype=np.float64)
        pos = c > 0
        if scheme == "none":
            pass
        elif scheme == "inv":
            w[pos] = c[pos].sum() / (c[pos] * pos.sum())
        elif scheme == "sqrt-inv":
            inv = c[pos].sum() / (c[pos] * pos.sum())
            w[pos] = np.sqrt(inv)
        w = np.clip(w, 1.0 / clip, clip)
        w[~legal] = 0.0
        w[~pos] = 0.0  # never weight unseen classes
        weights[dim] = torch.tensor(w, dtype=torch.float32, device=device)
    return weights


def masked_level_loss(logits, labels, target_mat, class_w):
    """Cost-sensitive weighted CE for one level.
    logits (B,H,W,10), labels (B,H,W) with -1 ignore. Returns (sum_wce, sum_w)."""
    lab = labels.reshape(-1)
    sel = lab >= 0
    if sel.sum() == 0:
        z = logits.sum() * 0.0
        return z, z + 0.0
    lg = logits.reshape(-1, NUM_PARTITION_TYPES)[sel]
    tr = lab[sel]
    logp = F.log_softmax(lg, dim=-1)
    tgt = target_mat[tr]                       # (M,10)
    ce = -(tgt * logp).sum(dim=-1)             # (M,)
    w = class_w[tr]                            # (M,)
    return (w * ce).sum(), w.sum()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def confusion_from_batch(conf, logits, labels):
    lab = labels.reshape(-1)
    sel = lab >= 0
    if sel.sum() == 0:
        return
    pred = logits.reshape(-1, NUM_PARTITION_TYPES)[sel].argmax(-1)
    tr = lab[sel]
    idx = (tr * NUM_PARTITION_TYPES + pred).cpu().numpy()
    conf += np.bincount(
        idx, minlength=NUM_PARTITION_TYPES ** 2
    ).reshape(NUM_PARTITION_TYPES, NUM_PARTITION_TYPES)


def summarize_confusion(conf):
    total = conf.sum()
    acc = np.trace(conf) / total if total else 0.0
    present = conf.sum(axis=1) > 0
    recall = np.divide(np.diag(conf), conf.sum(axis=1),
                       out=np.zeros(NUM_PARTITION_TYPES), where=present)
    pred_tot = conf.sum(axis=0)
    prec = np.divide(np.diag(conf), pred_tot,
                     out=np.zeros(NUM_PARTITION_TYPES), where=pred_tot > 0)
    denom = prec + recall
    f1 = np.divide(2 * prec * recall, denom,
                   out=np.zeros(NUM_PARTITION_TYPES), where=denom > 0)
    macro_f1 = f1[present].mean() if present.any() else 0.0
    return {"acc": float(acc), "macro_f1": float(macro_f1),
            "recall": recall, "support": conf.sum(axis=1).astype(int)}


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, targets, weights, device, optim=None, scaler=None):
    train = optim is not None
    model.train(train)
    conf = {dim: np.zeros((NUM_PARTITION_TYPES, NUM_PARTITION_TYPES),
                          dtype=np.int64) for dim, _ in MODEL_LEVELS}
    per_level = {dim: 0.0 for dim, _ in MODEL_LEVELS}
    tot_loss, n_batches = 0.0, 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = {d: labels[d].to(device, non_blocking=True) for d, _ in MODEL_LEVELS}
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, enabled=scaler is not None):
            out = model(x)
            # Average the per-level mean losses EQUALLY across levels, so the
            # more populous levels (16 >> 32 >> 64 in node count) don't dominate
            # the gradient and starve the coarse levels.
            level_loss = {}
            for dim, _ in MODEL_LEVELS:
                a, b = masked_level_loss(out[dim], labels[dim],
                                         targets[dim], weights[dim])
                level_loss[dim] = a / b.clamp_min(1.0)
            loss = sum(level_loss.values()) / len(MODEL_LEVELS)
        if train:
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
        else:
            for dim, _ in MODEL_LEVELS:
                confusion_from_batch(conf[dim], out[dim].float(), labels[dim])
        tot_loss += float(loss.detach())
        for dim, _ in MODEL_LEVELS:
            per_level[dim] += float(level_loss[dim].detach())
        n_batches += 1
    nb = max(n_batches, 1)
    return tot_loss / nb, conf, {d: per_level[d] / nb for d, _ in MODEL_LEVELS}


def main(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset")
    p.add_argument("--val-seqs", nargs="+", default=["Jockey"])
    p.add_argument("--train-seqs", nargs="+", default=None)
    p.add_argument("--cache-dir", default="/workspace/results/models/cache")
    p.add_argument("--out-dir", default="/workspace/results/models/surrogate")
    p.add_argument("--variant", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--fusion-dim", type=int, default=128)
    p.add_argument("--pretrained", action="store_true",
                   help="ImageNet-pretrained ConvNeXt backbone (stem averaged "
                        "RGB->luma); disambiguates info-ceiling vs weak extractor")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="linear LR warmup; from-scratch ConvNeXt needs it to "
                        "avoid collapsing to the class prior")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--class-weight", default="inv",
                   choices=["inv", "sqrt-inv", "none"])
    p.add_argument("--cost-temp", type=float, default=0.5)
    p.add_argument("--smooth-eps", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="cap superblocks per split (smoke test)")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device,
          "| cuda:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "n/a")

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs,
                                           args.train_seqs)
    print("pkls: {} train, {} val (val seqs {})".format(
        len(train_e), len(val_e), args.val_seqs))
    if not train_e or not val_e:
        raise SystemExit("Empty train or val split; check --val-seqs / dataset.")
    datamod.assert_real_luma(train_e)

    train_d = datamod.assemble_split(train_e, args.cache_dir, tag="train",
                                     rebuild=args.rebuild_cache, limit=args.limit)
    val_d = datamod.assemble_split(val_e, args.cache_dir, tag="val",
                                   rebuild=args.rebuild_cache, limit=args.limit)
    print("superblocks: {} train, {} val".format(
        len(train_d["luma"]), len(val_d["luma"])))

    counts = datamod.level_class_counts(train_d)
    targets = build_level_targets(args.cost_temp, args.smooth_eps, device)
    weights = build_class_weights(counts, args.class_weight, device)

    train_ds = datamod.make_torch_dataset(train_d)
    val_ds = datamod.make_torch_dataset(val_d)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = PartitionSurrogate(args.variant, args.fusion_dim,
                               pretrained=args.pretrained).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    warm = max(0, min(args.warmup_epochs, args.epochs - 1))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs - warm)
    if warm > 0:
        lin = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=0.02, total_iters=warm)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, [lin, cos], milestones=[warm])
    else:
        sched = cos
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp) if use_amp else None

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    fields = (["epoch", "train_loss", "val_loss", "headline_macro_f1"]
              + ["f1_{}".format(d) for d, _ in MODEL_LEVELS]
              + ["acc_{}".format(d) for d, _ in MODEL_LEVELS]
              + ["split_recall_{}".format(d) for d, _ in MODEL_LEVELS])
    mf = open(metrics_csv, "w", newline="")
    writer = csv.DictWriter(mf, fieldnames=fields)
    writer.writeheader()

    best = -1.0
    for epoch in range(args.epochs):
        tr_loss, _, tr_per = run_epoch(model, train_ld, targets, weights, device,
                                       optim=optim, scaler=scaler)
        va_loss, conf, _ = run_epoch(model, val_ld, targets, weights, device)
        sched.step()

        summ = {dim: summarize_confusion(conf[dim]) for dim, _ in MODEL_LEVELS}
        # Headline: cell-support-weighted mean of per-level macro-F1.
        sup = np.array([summ[d]["support"].sum() for d, _ in MODEL_LEVELS], float)
        f1s = np.array([summ[d]["macro_f1"] for d, _ in MODEL_LEVELS])
        headline = float((f1s * sup).sum() / sup.sum()) if sup.sum() else 0.0

        row = {"epoch": epoch, "train_loss": round(tr_loss, 4),
               "val_loss": round(va_loss, 4),
               "headline_macro_f1": round(headline, 4)}
        for dim, _ in MODEL_LEVELS:
            row["f1_{}".format(dim)] = round(summ[dim]["macro_f1"], 4)
            row["acc_{}".format(dim)] = round(summ[dim]["acc"], 4)
            # SPLIT (class 3) recall -- the safety-critical one for pruning.
            row["split_recall_{}".format(dim)] = round(
                float(summ[dim]["recall"][3]), 4)
        writer.writerow(row)
        mf.flush()

        print("epoch {:>2} | train {:.4f} val {:.4f} | headline F1 {:.4f}".format(
            epoch, tr_loss, va_loss, headline))
        print("   train loss/level: " + "  ".join(
            "{}px {:.4f}".format(d, tr_per[d]) for d, _ in MODEL_LEVELS))
        for dim, _ in MODEL_LEVELS:
            print("   {:>2}px: acc {:.3f} macroF1 {:.3f} SPLIT-rec {:.3f}".format(
                dim, summ[dim]["acc"], summ[dim]["macro_f1"],
                summ[dim]["recall"][3]))

        if headline > best:
            best = headline
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "headline_macro_f1": headline,
                        "val_confusion": {d: conf[d] for d, _ in MODEL_LEVELS}},
                       os.path.join(args.out_dir, "surrogate_best.pt"))
            print("   -> saved best (headline F1 {:.4f})".format(headline))

    mf.close()
    print("Done. Best headline macro-F1: {:.4f}".format(best))
    print("Metrics:", metrics_csv)


if __name__ == "__main__":
    main(sys.argv[1:])
