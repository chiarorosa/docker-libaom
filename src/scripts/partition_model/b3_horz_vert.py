#!/usr/bin/env python3
"""B3 -- experimento autocontido: separar HORZ de VERT no colapso de rotulos.

O estudante implantado colapsa as 10 PARTITION_TYPE em 3 classes
(NONE/SPLIT/REST). Este script testa a hipotese alternativa de 4 classes
(NONE/SPLIT/HORZ/VERT) e mede a pergunta decisiva: condicionado a "o rotulo
verdadeiro e retangular" (HORZ ou VERT), um MLP consegue prever a DIRECAO
acima do acaso (50%)?

Autocontido: NAO importa nem modifica student.py/distill.py/train_student_h9.py
alem de reusar suas interfaces publicas (features.py, data.py, partition_defs.py).
Nao faz parte do pipeline implantado -- apenas um experimento off-line.

Mapeamento de rotulos (10 -> 4), indices de PARTITION_TYPE
(0=NONE,1=HORZ,2=VERT,3=SPLIT,4=HORZ_A,5=HORZ_B,6=VERT_A,7=VERT_B,8=HORZ_4,9=VERT_4):
  classe 0 (NONE):  {0}
  classe 1 (SPLIT): {3}
  classe 2 (HORZ):  {1,4,5,8}
  classe 3 (VERT):  {2,6,7,9}

Uso (dentro do container av1_bench):
  /workspace/build/venv-ml/bin/python \
      /workspace/src/scripts/partition_model/b3_horz_vert.py \
      --dataset-dir /workspace/results/dataset_h9 \
      --out-dir /workspace/results/models/student_h9a_4cls
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from partition_defs import MODEL_LEVELS  # noqa: E402
import data as datamod  # noqa: E402
import features as featmod  # noqa: E402

# Mesmo particionamento treino/held-out do estudante H9a implantado
# (train_student_h9.py), para que este experimento seja comparavel.
TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
# Held-out = VAL_SEQS do estudante H9a + as 3 sequencias de teste.
HELD_OUT_SEQS = ["HoneyBee", "FlowerPan", "Lips", "Jockey", "RaceNight",
                 "RiverBank"]

# Classes B3.
CLASS_NONE, CLASS_SPLIT, CLASS_HORZ, CLASS_VERT = 0, 1, 2, 3
NUM_B3_CLASSES = 4
CLASS_NAMES = ["NONE", "SPLIT", "HORZ", "VERT"]

_HORZ_SET = {1, 4, 5, 8}
_VERT_SET = {2, 6, 7, 9}


def collapse4(part10):
    """PARTITION_TYPE (0..9) -> classe B3 (NONE/SPLIT/HORZ/VERT)."""
    p = int(part10)
    if p == 0:
        return CLASS_NONE
    if p == 3:
        return CLASS_SPLIT
    if p in _HORZ_SET:
        return CLASS_HORZ
    if p in _VERT_SET:
        return CLASS_VERT
    raise ValueError("PARTITION_TYPE fora do intervalo esperado: {}".format(p))


# --------------------------------------------------------------------------
# Coleta de features/rotulos por nivel (mesmo padrao de train_student_h9.py)
# --------------------------------------------------------------------------
def collect_by_dim(entries, per_pkl=None, limit=None, verbose_tag=""):
    """Per-block-size {'feat':(N,36), 'truth':(N,)} sobre o dataset H9a,
    usando o mapeamento 4-classes deste experimento."""
    acc = {dim: {"feat": [], "truth": []} for dim, _ in MODEL_LEVELS}
    n_sb = 0
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD; reextrair com a "
                                 "instrumentacao H9.".format(e["path"]))
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in acc:
                    continue
                f = featmod.node_features_h9a(sb["luma"], dim, r, c,
                                              sb["qindex"], sb["ctx"][k])
                acc[dim]["feat"].append(f)
                acc[dim]["truth"].append(collapse4(label))
            n_sb += 1
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and n_sb >= limit:
                return _finalize(acc), n_sb
        if verbose_tag:
            print("[{}] {}: +{} sb ({} total)".format(
                verbose_tag, os.path.basename(e["path"]), took, n_sb),
                flush=True)
        if limit and n_sb >= limit:
            break
    return _finalize(acc), n_sb


def _finalize(acc):
    return {dim: {"feat": np.asarray(v["feat"], dtype=np.float32),
                  "truth": np.asarray(v["truth"], dtype=np.int64)}
            for dim, v in acc.items()}


# --------------------------------------------------------------------------
# MLP 4-classes: 36 -> 64 -> 32 -> 4 (ReLU nas ocultas, linear na saida)
# --------------------------------------------------------------------------
def make_b3_student(in_features, hidden):
    layers, prev = [], in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    layers += [nn.Linear(prev, NUM_B3_CLASSES)]
    return nn.Sequential(*layers)


def train_b3_student(rec, hidden, device, epochs, lr, wd=1e-4, batch=4096):
    """CE de rotulo duro, SEM ponderacao de classe (comparabilidade com H9a).
    Padroniza features (mean/std, std clamped >= 0.1) e dobra a padronizacao
    na primeira camada linear ao final (mesma convencao de distill.py)."""
    feat = torch.tensor(rec["feat"], dtype=torch.float32)
    truth = torch.tensor(rec["truth"], dtype=torch.long)

    mean = feat.mean(0)
    std = feat.std(0).clamp_min(1e-1)
    feat_n = (feat - mean) / std

    net = make_b3_student(feat.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = len(feat_n)
    for ep in range(epochs):
        perm = torch.randperm(n)
        net.train()
        tot = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            fb = feat_n[idx].to(device)
            yb = truth[idx].to(device)
            logits = net(fb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print("    epoch {:>2}/{} loss={:.4f}".format(
                ep + 1, epochs, tot / n), flush=True)

    with torch.no_grad():
        first = next(m for m in net.modules() if isinstance(m, nn.Linear))
        w = first.weight
        first.bias -= (w @ (mean.to(device) / std.to(device)))
        first.weight.copy_(w / std.to(device))
    return net, {"mean": mean.numpy(), "std": std.numpy()}


# --------------------------------------------------------------------------
# Avaliacao held-out
# --------------------------------------------------------------------------
def confusion4(truth, pred):
    cm = np.zeros((NUM_B3_CLASSES, NUM_B3_CLASSES), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    return cm


def macro_f1_from_cm(cm):
    f1s = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def evaluate_dim(net, mean_std_noop, feat, truth, device):
    """net ja tem a padronizacao dobrada na 1a camada -> alimenta features cruas."""
    net.eval()
    with torch.no_grad():
        fb = torch.tensor(feat, dtype=torch.float32, device=device)
        logits = net(fb).cpu().numpy()
    probs = _softmax(logits)
    pred4 = logits.argmax(-1)

    cm = confusion4(truth, pred4)
    macro_f1, per_class_f1 = macro_f1_from_cm(cm)

    horz_mask = truth == CLASS_HORZ
    vert_mask = truth == CLASS_VERT
    recall_horz = float((pred4[horz_mask] == CLASS_HORZ).mean()) if horz_mask.any() else float("nan")
    recall_vert = float((pred4[vert_mask] == CLASS_VERT).mean()) if vert_mask.any() else float("nan")

    # Acuracia direcional condicional: apenas nos com rotulo HORZ/VERT,
    # argmax restrito as colunas HORZ/VERT (ignora NONE/SPLIT).
    rect_mask = horz_mask | vert_mask
    n_rect = int(rect_mask.sum())
    if n_rect > 0:
        sub_logits = logits[rect_mask][:, [CLASS_HORZ, CLASS_VERT]]
        dir_pred = np.where(sub_logits[:, 0] >= sub_logits[:, 1],
                            CLASS_HORZ, CLASS_VERT)
        dir_truth = truth[rect_mask]
        dir_acc = float((dir_pred == dir_truth).mean())
        horz_prop = float((dir_truth == CLASS_HORZ).mean())
    else:
        dir_acc = float("nan")
        horz_prop = float("nan")

    return {
        "n": len(truth),
        "n_rect": n_rect,
        "cm": cm,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "recall_horz": recall_horz,
        "recall_vert": recall_vert,
        "dir_acc": dir_acc,
        "horz_prop": horz_prop,
    }


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# Relatorio
# --------------------------------------------------------------------------
def write_report(path, per_dim_eval, agg):
    lines = []
    lines.append("# B3 -- HORZ vs VERT: existe sinal direcional?\n")
    lines.append(
        "Experimento off-line, autocontido (`b3_horz_vert.py`), nao faz parte "
        "do pipeline implantado. Colapso de rotulos 10->4 (NONE/SPLIT/HORZ/VERT); "
        "MLP 36->64->32->4 por nivel, CE de rotulo duro, sem ponderacao de "
        "classe, treinado nas mesmas 10 sequencias de treino do estudante "
        "H9a implantado.\n")
    lines.append(
        "\nPergunta decisiva: condicionado a 'o rotulo verdadeiro e "
        "retangular' (HORZ ou VERT), o modelo prediz a DIRECAO acima do "
        "acaso (50%)? Acuracia direcional condicional = argmax restrito as "
        "colunas HORZ/VERT (softmax completo, ignorando NONE/SPLIT), medida "
        "apenas nos nos cujo rotulo verdadeiro e HORZ ou VERT.\n")

    lines.append("\n## Acuracia direcional por nivel (held-out)\n")
    lines.append("| nivel (px) | n (HORZ|VERT) | prop. HORZ real | acuracia "
                 "direcional | baseline 50% | baseline 'sempre HORZ' |")
    lines.append("|---|---|---|---|---|---|")
    for dim in sorted(per_dim_eval, reverse=True):
        ev = per_dim_eval[dim]
        maj_baseline = max(ev["horz_prop"], 1 - ev["horz_prop"])
        lines.append("| {} | {} | {:.1%} | {:.1%} | 50.0% | {:.1%} |".format(
            dim, ev["n_rect"], ev["horz_prop"], ev["dir_acc"], maj_baseline))
    maj_baseline_agg = max(agg["horz_prop"], 1 - agg["horz_prop"])
    lines.append("| **agregado** | {} | {:.1%} | {:.1%} | 50.0% | {:.1%} |".format(
        agg["n_rect"], agg["horz_prop"], agg["dir_acc"], maj_baseline_agg))

    lines.append("\n## Metricas secundarias por nivel\n")
    for dim in sorted(per_dim_eval, reverse=True):
        ev = per_dim_eval[dim]
        lines.append("\n### nivel {}px\n".format(dim))
        lines.append("n={}, macro-F1(4cls)={:.4f}, recall(HORZ)={:.1%}, "
                     "recall(VERT)={:.1%}\n".format(
                         ev["n"], ev["macro_f1"], ev["recall_horz"],
                         ev["recall_vert"]))
        lines.append("Matriz de confusao (linhas=verdade, colunas=predito; "
                     "ordem {}):\n".format(CLASS_NAMES))
        lines.append("```")
        header = "        " + "".join("{:>10}".format(n) for n in CLASS_NAMES)
        lines.append(header)
        for i, name in enumerate(CLASS_NAMES):
            row = "".join("{:>10}".format(int(v)) for v in ev["cm"][i])
            lines.append("{:<8}{}".format(name, row))
        lines.append("```")
        lines.append("F1 por classe: " + ", ".join(
            "{}={:.4f}".format(n, f) for n, f in zip(CLASS_NAMES, ev["per_class_f1"])))

    lines.append("\n## Veredito\n")
    dirs = [per_dim_eval[d]["dir_acc"] for d in per_dim_eval]
    majs = [max(per_dim_eval[d]["horz_prop"], 1 - per_dim_eval[d]["horz_prop"])
            for d in per_dim_eval]
    all_above = all(da - ma > 0.10 for da, ma in zip(dirs, majs))
    all_near_chance = all(abs(da - 0.5) < 0.05 for da in dirs)
    if all_near_chance:
        veredito = (
            "**NEGATIVO.** A acuracia direcional condicional fica proxima de "
            "50% em todos os niveis -- a direcao HORZ vs VERT nao e "
            "aprendivel a partir das features H9a atuais, mesmo condicionando "
            "ao fato de o rotulo ser retangular. B3 nao se sustenta com este "
            "conjunto de features: nao ha sinal a explorar separando HORZ de "
            "VERT na politica C (`prune_rect_part[HORZ]`/`[VERT]`).")
    elif all_above:
        veredito = (
            "**POSITIVO.** A acuracia direcional condicional supera "
            "claramente tanto o acaso (50%) quanto o baseline de 'sempre "
            "escolher a classe majoritaria' em todos os niveis -- ha sinal "
            "direcional aproveitavel nas features H9a. B3 e viavel e "
            "justificaria separar prune_rect_part[HORZ] de "
            "prune_rect_part[VERT] em partition_strategy.c:1260-1261.")
    else:
        veredito = (
            "**MISTO.** Alguns niveis mostram sinal direcional acima dos "
            "baselines, outros ficam perto do acaso -- ver tabela acima por "
            "nivel antes de decidir sobre B3.")
    lines.append(veredito)
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--held-out-seqs", nargs="+", default=HELD_OUT_SEQS)
    p.add_argument("--out-dir",
                   default="/workspace/results/models/student_h9a_4cls")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--per-pkl-train", type=int, default=3000)
    p.add_argument("--per-pkl-eval", type=int, default=2000)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    entries = datamod.discover_pkls(args.dataset_dir)
    print("descobertos {} pkl(s) em {}".format(len(entries), args.dataset_dir),
         flush=True)

    # split_entries(entries, val_seqs, train_seqs) -> (train, val): val holds
    # everything matching held_out_seqs; train holds the rest, restricted to
    # train_seqs. One call keeps the two sets disjoint by construction.
    train_e, heldout_e = datamod.split_entries(entries, args.held_out_seqs,
                                               args.train_seqs)
    print("treino: {} pkl(s) ({}); held-out: {} pkl(s) ({})".format(
        len(train_e), args.train_seqs, len(heldout_e), args.held_out_seqs),
        flush=True)
    if not train_e:
        raise SystemExit("nenhum pkl de treino encontrado -- verifique "
                         "--dataset-dir/--train-seqs")
    if not heldout_e:
        raise SystemExit("nenhum pkl held-out encontrado -- verifique "
                         "--dataset-dir/--held-out-seqs")

    datamod.assert_real_luma(train_e)
    datamod.assert_real_luma(heldout_e)

    print("\n=== coleta (treino) ===", flush=True)
    train_data, n_sb_train = collect_by_dim(
        train_e, per_pkl=args.per_pkl_train, verbose_tag="train")
    print("total superblocos (treino): {}".format(n_sb_train), flush=True)

    print("\n=== coleta (held-out) ===", flush=True)
    eval_data, n_sb_eval = collect_by_dim(
        heldout_e, per_pkl=args.per_pkl_eval, verbose_tag="eval")
    print("total superblocos (held-out): {}".format(n_sb_eval), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    bundle = {"hidden": args.hidden, "students": {}, "norm": {},
              "num_features": featmod.NUM_FEATURES_H9A, "feature_set": "h9a",
              "classes": NUM_B3_CLASSES}

    per_dim_eval = {}
    all_truth, all_logits_dir = [], []

    for dim, _ in MODEL_LEVELS:
        rec = train_data[dim]
        if len(rec["truth"]) == 0:
            print("[dim {}] sem amostras de treino, pulando".format(dim))
            continue
        print("\n=== treino dim {}px (n={}) ===".format(dim, len(rec["truth"])),
             flush=True)
        counts = np.bincount(rec["truth"], minlength=NUM_B3_CLASSES).tolist()
        print("  counts(NONE/SPLIT/HORZ/VERT)={}".format(counts), flush=True)
        net, norm = train_b3_student(rec, args.hidden, device, args.epochs,
                                     args.lr, batch=args.batch)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm

        erec = eval_data[dim]
        if len(erec["truth"]) == 0:
            print("[dim {}] sem amostras held-out, pulando avaliacao".format(dim))
            continue
        ev = evaluate_dim(net, norm, erec["feat"], erec["truth"], device)
        per_dim_eval[dim] = ev
        print("[dim {:>2}] n={} n_rect={} dir_acc={:.3f} horz_prop={:.3f} "
             "macro_f1={:.4f} recall_horz={:.3f} recall_vert={:.3f}".format(
                 dim, ev["n"], ev["n_rect"], ev["dir_acc"], ev["horz_prop"],
                 ev["macro_f1"], ev["recall_horz"], ev["recall_vert"]),
             flush=True)

        # acumula para o agregado (nos retangulares, truth e pred de direcao)
        with torch.no_grad():
            fb = torch.tensor(erec["feat"], dtype=torch.float32, device=device)
            logits = net(fb).cpu().numpy()
        rect_mask = (erec["truth"] == CLASS_HORZ) | (erec["truth"] == CLASS_VERT)
        all_truth.append(erec["truth"][rect_mask])
        all_logits_dir.append(logits[rect_mask][:, [CLASS_HORZ, CLASS_VERT]])

    students_path = os.path.join(args.out_dir, "students.pt")
    torch.save(bundle, students_path)
    print("\nSalvo ->", students_path, flush=True)

    all_truth = np.concatenate(all_truth) if all_truth else np.array([], dtype=np.int64)
    if all_logits_dir:
        all_logits_dir = np.concatenate(all_logits_dir)
        dir_pred = np.where(all_logits_dir[:, 0] >= all_logits_dir[:, 1],
                            CLASS_HORZ, CLASS_VERT)
        agg_dir_acc = float((dir_pred == all_truth).mean())
        agg_horz_prop = float((all_truth == CLASS_HORZ).mean())
    else:
        agg_dir_acc = float("nan")
        agg_horz_prop = float("nan")
    agg = {"n_rect": len(all_truth), "dir_acc": agg_dir_acc,
          "horz_prop": agg_horz_prop}

    print("\n=== AGREGADO (todos os niveis, held-out) ===", flush=True)
    print("n_rect={} dir_acc={:.3f} horz_prop={:.3f} maj_baseline={:.3f}".format(
        agg["n_rect"], agg["dir_acc"], agg["horz_prop"],
        max(agg["horz_prop"], 1 - agg["horz_prop"])), flush=True)

    report_path = os.path.join(args.out_dir, "b3_report.md")
    write_report(report_path, per_dim_eval, agg)
    print("Relatorio ->", report_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
