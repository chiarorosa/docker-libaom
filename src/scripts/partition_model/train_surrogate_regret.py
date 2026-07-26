#!/usr/bin/env python3
"""Retreina o ConvNeXt substituto com objetivo de REGRET (experimento autocontido).

Por que existe
--------------
A afirmacao "os pixels saturam na variancia" -- tratada no corpo da tese como
espinha dorsal -- foi estabelecida medindo o teto do dominio de pixels com um
ConvNeXt que tem dois defeitos metodologicos:

1. **Selecao do checkpoint pelo criterio errado.** `train.py:299-304` salva o
   melhor por macro-F1. Em `results/models/surrogate_real/metrics.csv` o minimo
   da perda de validacao esta na epoca 0 (2,0379) e o macro-F1 so cresce ate a
   epoca 27 (0,1866), quando a val_loss ja subiu para 2,2992 -- 15% acima do
   minimo. O `surrogate_best.pt` implantado como "teto" foi colhido em
   sobreajuste franco, por um criterio que discorda da perda.

2. **Objetivo de treino que a propria tese demonstrou ser mau proxy.** O treino
   usa entropia cruzada sobre rotulos duros, isto e, otimiza acuracia por no.
   O Approach B (`RESULTADOS_approachB.md` §6) mediu que **acuracia por-no e mau
   proxy do BD x tempo real de um podador**: a GNN venceu claramente offline e
   ficou ~2x pior no encoder. O alvo certo -- o *regret* do crivo A5 -- existe
   (`train_regret.py`) mas foi aplicado somente as 36 features do H9a, **nunca
   a pixels**.

Este script ataca os dois. Nao ha re-extracao: o `.pkl` guarda a luma sem perdas
(`round(pkl*255)` = quadro-fonte, `maxdiff=0`); e trabalho de GPU.

O que muda, e o que deliberadamente NAO muda
--------------------------------------------
Muda **so o objetivo e a selecao**. A arquitetura (`PartitionSurrogate`), o
formato de saida (logits por nivel sobre PARTITION_TYPE) e, portanto, todo o
caminho a jusante -- `surrogate_replay.py`, o hook `AV1_STUDENT_PROBS_FILE`, a
politica de poda -- ficam **identicos**. Isso e proposital: a comparacao contra o
teto atual precisa ser "mesmo modelo, mesma politica, treino melhor", sem nenhuma
outra variavel solta.

- **Perda:** CE ponderada por no com peso `1 + alpha * regret_rel_norm`. Errar um
  no barato custa pouco; errar um no caro custa muito. `regret_rel` vem de
  `regret.node_regrets` (o mesmo do crivo A5), normalizado por nivel pelo seu
  percentil 95 para que `alpha` tenha significado estavel entre niveis.
- **Selecao:** pelo **mesmo** criterio ponderado, medido na validacao, com parada
  antecipada por paciencia. Treinar por um criterio e selecionar por outro foi
  precisamente o defeito 1.

Nos sem regret exato (`nr["exact"]` falso) entram com peso neutro 1,0 em vez de
serem descartados: descarta-los enviesaria a amostra para os nos faceis.

Portao
------
Este script **nao** decide nada sozinho. O veredito e do crivo A5
(`oracle_regret.py`) aplicado ao modelo resultante, contra a variancia e contra
pixels24/H9a. Se nao superar a variancia la, o negativo da tese se mantem -- mas
passa a repousar em uma medicao honesta em vez de um modelo mal-selecionado
treinado contra o objetivo errado.

Reproducao:
    /workspace/build/venv-ml/bin/python \
        src/scripts/partition_model/train_surrogate_regret.py
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from partition_defs import MODEL_LEVELS, NUM_PARTITION_TYPES  # noqa: E402
import data as datamod  # noqa: E402
import regret as regretmod  # noqa: E402
from model import PartitionSurrogate  # noqa: E402

# Mesmo particionamento do estudante H9a implantado (train_student_h9.py).
TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
VAL_SEQS = ["HoneyBee", "FlowerPan", "Lips"]

DEFAULT_OUT_DIR = "/workspace/results/models/surrogate_regret"
SB = datamod.SB_SIZE_PX
IGNORE = datamod.IGNORE


# --------------------------------------------------------------------------
# Montagem: luma + rotulos + grade de regret, na mesma geometria de data.py
# --------------------------------------------------------------------------
def assemble_with_regret(entries, per_pkl=None, verbose_tag=""):
    """Empilha {luma, qindex, lab{dim}, reg{dim}} para um split.

    `reg{dim}` espelha a grade de `lab{dim}`: regret_rel do no naquela celula,
    ou NaN onde nao ha no (celulas IGNORE). O treino converte NaN em peso
    neutro.
    """
    luma, qidx = [], []
    labs = {dim: [] for dim, _ in MODEL_LEVELS}
    regs = {dim: [] for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD (dataset_h9 requerido)."
                                 .format(e["path"]))
            lab = {dim: np.full((side, side), IGNORE, dtype=np.int64)
                   for dim, side in MODEL_LEVELS}
            reg = {dim: np.full((side, side), np.nan, dtype=np.float32)
                   for dim, side in MODEL_LEVELS}
            for dim, r, c, _l, label in sb["members"]:
                if dim in lab:
                    lab[dim][r, c] = label
            for nr in regretmod.node_regrets(sb["members"], sb["ctx"]):
                dim = nr["dim"]
                if dim not in reg or not nr["exact"]:
                    continue
                _d, r, c, _l, _lab = sb["members"][nr["k"]]
                reg[dim][r, c] = float(nr["regret_rel"])
            luma.append(sb["luma"])
            qidx.append(sb["qindex"])
            for dim, _ in MODEL_LEVELS:
                labs[dim].append(lab[dim])
                regs[dim].append(reg[dim])
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
        if verbose_tag:
            print("[{}] {}: +{} sb ({} total)".format(
                verbose_tag, os.path.basename(e["path"]), took, n_sb),
                flush=True)
    out = {"luma": np.asarray(luma, dtype=np.uint8),
           "qindex": np.asarray(qidx, dtype=np.int64)}
    for dim, _ in MODEL_LEVELS:
        out["lab{}".format(dim)] = np.asarray(labs[dim], dtype=np.int64)
        out["reg{}".format(dim)] = np.asarray(regs[dim], dtype=np.float32)
    return out, n_sb


def _cache_path(cache_dir, entries, per_pkl, tag):
    h = hashlib.sha1()
    for e in sorted(entries, key=lambda x: x["path"]):
        h.update(e["path"].encode())
        h.update(str(os.path.getsize(e["path"])).encode())
    h.update(str(per_pkl).encode())
    return os.path.join(cache_dir, "regret_{}_{}.npz".format(tag, h.hexdigest()[:12]))


def assemble_cached(entries, per_pkl, cache_dir, tag):
    path = _cache_path(cache_dir, entries, per_pkl, tag)
    if os.path.exists(path):
        print("[data] cache {} -> {}".format(tag, path), flush=True)
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d, n = assemble_with_regret(entries, per_pkl=per_pkl, verbose_tag=tag)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **d)
    print("[data] {}: {} superblocos -> {}".format(tag, n, path), flush=True)
    return d


# --------------------------------------------------------------------------
# Normalizacao do regret e Dataset
# --------------------------------------------------------------------------
def regret_scales(assembled, pct=95.0):
    """Percentil `pct` de regret_rel por nivel (>0), usado para normalizar o
    peso. Sem isso, `alpha` significaria coisas diferentes em cada nivel."""
    scales = {}
    for dim, _ in MODEL_LEVELS:
        v = assembled["reg{}".format(dim)].reshape(-1)
        v = v[np.isfinite(v) & (v > 0)]
        scales[dim] = float(np.percentile(v, pct)) if v.size else 1.0
        if scales[dim] <= 0:
            scales[dim] = 1.0
    return scales


class RegretSuperblockDataset(Dataset):
    def __init__(self, d):
        self.luma = d["luma"]
        self.qindex = d["qindex"]
        self.labs = {dim: d["lab{}".format(dim)] for dim, _ in MODEL_LEVELS}
        self.regs = {dim: d["reg{}".format(dim)] for dim, _ in MODEL_LEVELS}

    def __len__(self):
        return len(self.luma)

    def __getitem__(self, i):
        luma = torch.from_numpy(self.luma[i].astype(np.float32) / 255.0)
        q = float(self.qindex[i]) / 255.0
        qplane = torch.full((SB, SB), q, dtype=torch.float32)
        x = torch.stack([luma, qplane], dim=0)
        labels = {dim: torch.from_numpy(self.labs[dim][i].astype(np.int64))
                  for dim, _ in MODEL_LEVELS}
        regs = {dim: torch.from_numpy(self.regs[dim][i].astype(np.float32))
                for dim, _ in MODEL_LEVELS}
        return x, labels, regs


# --------------------------------------------------------------------------
# Perda ponderada por regret
# --------------------------------------------------------------------------
def regret_weighted_loss(logits, labels, regs, scale, alpha):
    """CE por no, com peso 1 + alpha * min(regret_rel/scale, 1).

    Retorna (soma_ponderada, soma_dos_pesos) para que o chamador possa fazer a
    media correta entre lotes. Celulas IGNORE nao contribuem; nos sem regret
    exato (NaN) entram com peso neutro 1,0 -- descarta-los enviesaria a amostra
    para os nos faceis.
    """
    B = logits.shape[0]
    lg = logits.permute(0, 2, 3, 1).reshape(-1, NUM_PARTITION_TYPES)
    lb = labels.reshape(-1)
    rg = regs.reshape(-1)
    valid = lb >= 0
    if not bool(valid.any()):
        z = logits.sum() * 0.0
        return z, z.detach() + 0.0
    lg, lb, rg = lg[valid], lb[valid], rg[valid]
    ce = F.cross_entropy(lg, lb, reduction="none")
    rel = torch.nan_to_num(rg, nan=0.0).clamp_min(0.0) / scale
    w = 1.0 + alpha * rel.clamp(max=1.0)
    del B
    return (ce * w).sum(), w.sum()


def run_epoch(model, loader, scales, alpha, device, optim=None):
    train = optim is not None
    model.train(train)
    tot_w, tot_l = 0.0, 0.0
    with torch.set_grad_enabled(train):
        for x, labels, regs in loader:
            x = x.to(device, non_blocking=True)
            labels = {d: labels[d].to(device, non_blocking=True)
                      for d, _ in MODEL_LEVELS}
            regs = {d: regs[d].to(device, non_blocking=True)
                    for d, _ in MODEL_LEVELS}
            out = model(x)
            num, den = 0.0, 0.0
            for dim, _ in MODEL_LEVELS:
                a, b = regret_weighted_loss(out[dim], labels[dim], regs[dim],
                                            scales[dim], alpha)
                num = num + a
                den = den + b
            loss = num / den.clamp_min(1.0)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            tot_l += float(num.detach())
            tot_w += float(den.detach())
    return tot_l / max(tot_w, 1.0)


# --------------------------------------------------------------------------
def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--val-seqs", nargs="+", default=VAL_SEQS)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--cache-dir", default="/workspace/results/models/_cache_regret")
    p.add_argument("--variant", default="tiny")
    p.add_argument("--fusion-dim", type=int, default=256)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8,
                   help="paradas sem melhora na validacao antes de encerrar")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--alpha", type=float, default=3.0,
                   help="peso maximo extra do regret (peso = 1 + alpha*rel)")
    p.add_argument("--per-pkl-train", type=int, default=3000)
    p.add_argument("--per-pkl-val", type=int, default=2000)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("dispositivo: {} | alpha={} | selecao por perda ponderada de "
          "validacao".format(device, args.alpha), flush=True)

    entries = datamod.discover_pkls(args.dataset_dir)
    train_e, val_e = datamod.split_entries(entries, args.val_seqs,
                                           args.train_seqs)
    if not train_e or not val_e:
        raise SystemExit("split vazio: verifique --dataset-dir e as listas de "
                         "sequencias")
    datamod.assert_real_luma(train_e)
    datamod.assert_real_luma(val_e)

    train_d = assemble_cached(train_e, args.per_pkl_train, args.cache_dir,
                              "train")
    val_d = assemble_cached(val_e, args.per_pkl_val, args.cache_dir, "val")
    scales = regret_scales(train_d)
    print("escalas de regret (p95) por nivel: {}".format(
        {d: round(s, 6) for d, s in scales.items()}), flush=True)

    train_ld = DataLoader(RegretSuperblockDataset(train_d),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_ld = DataLoader(RegretSuperblockDataset(val_d),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = PartitionSurrogate(args.variant, args.fusion_dim,
                               pretrained=args.pretrained).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "metrics.csv")
    best_path = os.path.join(args.out_dir, "surrogate_regret_best.pt")
    with open(metrics_path, "w") as f:
        f.write("epoch,train_wloss,val_wloss,secs\n")

    best, best_ep, since = float("inf"), -1, 0
    for ep in range(args.epochs):
        t0 = time.time()
        tr = run_epoch(model, train_ld, scales, args.alpha, device, optim)
        va = run_epoch(model, val_ld, scales, args.alpha, device, None)
        sched.step()
        dt = time.time() - t0
        with open(metrics_path, "a") as f:
            f.write("{},{:.6f},{:.6f},{:.1f}\n".format(ep, tr, va, dt))
        flag = ""
        if va < best:
            best, best_ep, since = va, ep, 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep, "val_wloss": va, "scales": scales},
                       best_path)
            flag = "  -> melhor, salvo"
        else:
            since += 1
        print("epoca {:>2}  treino {:.5f}  val {:.5f}  ({:.0f}s){}".format(
            ep, tr, va, dt, flag), flush=True)
        if since >= args.patience:
            print("parada antecipada: {} epocas sem melhora".format(since),
                  flush=True)
            break

    print("\nSURROGATE_REGRET_DONE  melhor val_wloss={:.6f} na epoca {} -> {}"
          .format(best, best_ep, best_path), flush=True)
    print("Proximo passo (o portao): aplicar o crivo A5 (oracle_regret.py) a "
          "este modelo contra variancia e pixels24/H9a.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
