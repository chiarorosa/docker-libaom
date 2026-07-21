#!/usr/bin/env python3
"""H9d -- previsibilidade de EXT (AB/4-way): existe sinal para um seletivo?

O estudante implantado (H9a) colapsa as 10 PARTITION_TYPE em 3 classes
(NONE/SPLIT/REST) e nunca distingue os casos "AB" (HORZ_A/HORZ_B/VERT_A/
VERT_B) nem "4-way" (HORZ_4/VERT_4) do resto. Esses seis rotulos --
chamados aqui de EXT (extended partitions) -- sao os mais caros de
pesquisar (a busca AB/4-way so roda depois de HORZ/VERT/SPLIT). Este
experimento testa a pergunta decisiva e diferente de B3: condicionado as
36 features H9a (sem custo de busca), um MLP consegue prever se o no vai
"ganhar" com uma particao EXT, o bastante para sustentar uma poda
seletiva (pular a busca AB/4-way quando a probabilidade prevista for
baixa) sem descartar excesso de vencedores reais?

Rotulo binario por no (indices de PARTITION_TYPE, 0=NONE,1=HORZ,2=VERT,
3=SPLIT,4=HORZ_A,5=HORZ_B,6=VERT_A,7=VERT_B,8=HORZ_4,9=VERT_4):
  y = 1 (EXT)     se partition_label em {4,5,6,7,8,9} (AB ou 4-way)
  y = 0 (NAO_EXT) caso contrario (NONE, HORZ, VERT, SPLIT)

Metrica central: a troca poda x custo. A regra de decisao e "pule a busca
AB/4-way no no se p_ext < theta". Definindo, para um dado theta:
  search_avoided(theta) = fracao de TODOS os nos com p_ext < theta
                           (nos onde a busca AB/4-way e evitada)
  winners_lost(theta)   = fracao dos nos EXT verdadeiros com p_ext < theta
                           (= 1 - recall; vencedores reais descartados,
                           custo em BD-rate)
O baseline "blanket" (nunca pesquisar EXT) e (search_avoided=100%,
winners_lost=100%); o baseline "sempre pesquisar" e (0%, 0%). Este
script varre theta e reporta a tabela nas duas direcoes (fixando
winners_lost e lendo search_avoided, e vice-versa).

Autocontido: NAO importa nem modifica student.py/distill.py/
train_student_h9.py alem de reusar suas interfaces publicas (features.py,
data.py, partition_defs.py). Nao faz parte do pipeline implantado --
apenas um experimento off-line, modelado sobre b3_horz_vert.py.

Uso (dentro do container av1_bench):
  /workspace/build/venv-ml/bin/python \
      /workspace/src/scripts/partition_model/h9d_predictability.py \
      --dataset-dir /workspace/results/dataset_h9 \
      --out-dir /workspace/results/models/h9d_predictability
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

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Mesmo particionamento treino/held-out do estudante H9a implantado
# (train_student_h9.py) e do experimento B3, para comparabilidade.
TRAIN_SEQS = ["Beauty", "Bosphorus", "CityAlley", "FlowerFocus", "FlowerKids",
              "ReadySetGo", "ShakeNDry", "SunBath", "Twilight", "YachtRide"]
# Held-out = VAL_SEQS do estudante H9a + as 3 sequencias de teste.
HELD_OUT_SEQS = ["HoneyBee", "FlowerPan", "Lips", "Jockey", "RaceNight",
                 "RiverBank"]

# Classes H9d (binario).
CLASS_NOT_EXT, CLASS_EXT = 0, 1
NUM_EXT_CLASSES = 2
CLASS_NAMES = ["NAO_EXT", "EXT"]

# AB (HORZ_A/HORZ_B/VERT_A/VERT_B) + 4-way (HORZ_4/VERT_4).
_EXT_SET = {4, 5, 6, 7, 8, 9}


def label_ext(part10):
    """PARTITION_TYPE (0..9) -> rotulo binario EXT (1) / NAO_EXT (0)."""
    return CLASS_EXT if int(part10) in _EXT_SET else CLASS_NOT_EXT


# --------------------------------------------------------------------------
# Coleta de features/rotulos por nivel (mesmo padrao de b3_horz_vert.py)
# --------------------------------------------------------------------------
def collect_by_dim(entries, per_pkl=None, limit=None, verbose_tag=""):
    """Per-block-size {'feat':(N,36), 'truth':(N,)} sobre o dataset H9a,
    usando o rotulo binario EXT deste experimento."""
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
                acc[dim]["truth"].append(label_ext(label))
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
# MLP binario: 36 -> 64 -> 32 -> 2 (ReLU nas ocultas, linear na saida)
# --------------------------------------------------------------------------
def make_ext_student(in_features, hidden):
    layers, prev = [], in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    layers += [nn.Linear(prev, NUM_EXT_CLASSES)]
    return nn.Sequential(*layers)


def train_ext_student(rec, hidden, device, epochs, lr, wd=1e-4, batch=4096):
    """CE de rotulo duro, SEM ponderacao de classe (comparabilidade com H9a
    e B3, apesar do desbalanceamento natural de EXT). Padroniza features
    (mean/std, std clamped >= 0.1) e dobra a padronizacao na primeira
    camada linear ao final (mesma convencao de distill.py)."""
    feat = torch.tensor(rec["feat"], dtype=torch.float32)
    truth = torch.tensor(rec["truth"], dtype=torch.long)

    mean = feat.mean(0)
    std = feat.std(0).clamp_min(1e-1)
    feat_n = (feat - mean) / std

    net = make_ext_student(feat.shape[1], hidden).to(device)
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


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------
# AUC -- sklearn se disponivel, senao fallback manual (rank-based)
# --------------------------------------------------------------------------
def _rankdata_average(a):
    """Ranks (1-based) com media de empates -- equivalente a
    scipy.stats.rankdata(a, method='average'), sem depender de scipy."""
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = ranks_sorted
    return ranks


def manual_roc_auc(y, scores):
    """AUC = P(score(positivo) > score(negativo)), via soma de ranks
    (estatistica de Mann-Whitney U), com tratamento de empates."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_average(scores)
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def manual_pr_auc(y, scores):
    """Average precision (area sob a curva precisao-recall, integracao em
    degraus), mesma convencao de sklearn.metrics.average_precision_score:
    agrupa por score empatado e usa o ponto mais a direita de cada grupo."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    change = np.r_[scores_sorted[1:] != scores_sorted[:-1], True]
    idx = np.nonzero(change)[0]
    precision_th = precision[idx]
    recall_th = recall[idx]
    recall_prev = np.r_[0.0, recall_th[:-1]]
    ap = float(np.sum((recall_th - recall_prev) * precision_th))
    return ap


def compute_aucs(y, scores):
    y = np.asarray(y)
    if _HAS_SKLEARN:
        try:
            roc = float(roc_auc_score(y, scores))
        except ValueError:
            roc = float("nan")
        try:
            pr = float(average_precision_score(y, scores))
        except ValueError:
            pr = float("nan")
        return roc, pr
    return manual_roc_auc(y, scores), manual_pr_auc(y, scores)


# --------------------------------------------------------------------------
# Troca poda x custo (pruning tradeoff)
# --------------------------------------------------------------------------
def tradeoff_table(p_ext, truth,
                    winners_lost_targets=(0.05, 0.10, 0.20),
                    avoided_targets=(0.50, 0.70, 0.90)):
    """Para cada alvo de winners_lost, acha theta = quantil dos p_ext dos
    positivos verdadeiros (winners_lost(theta) = CDF empirica de p_ext
    restrita aos positivos, avaliada em theta) e le search_avoided(theta)
    resultante. Simetricamente para cada alvo de search_avoided, theta =
    quantil dos p_ext de TODOS os nos, e le-se winners_lost(theta)."""
    p_ext = np.asarray(p_ext)
    truth = np.asarray(truth)
    pos_mask = truth == CLASS_EXT
    p_pos = p_ext[pos_mask]
    n_pos = len(p_pos)

    rows_by_wl = []
    for wl in winners_lost_targets:
        if n_pos == 0:
            rows_by_wl.append((wl, float("nan"), float("nan"), float("nan")))
            continue
        theta = float(np.quantile(p_pos, wl))
        actual_wl = float((p_pos < theta).mean())
        avoided = float((p_ext < theta).mean())
        rows_by_wl.append((wl, theta, actual_wl, avoided))

    rows_by_avoided = []
    for av in avoided_targets:
        theta = float(np.quantile(p_ext, av))
        actual_avoided = float((p_ext < theta).mean())
        wl = float((p_pos < theta).mean()) if n_pos > 0 else float("nan")
        rows_by_avoided.append((av, theta, actual_avoided, wl))

    return rows_by_wl, rows_by_avoided


def print_tradeoff(rows_by_wl, rows_by_avoided, indent="    "):
    print("{}[fixando winners_lost -> search_avoided]".format(indent))
    for target_wl, theta, actual_wl, avoided in rows_by_wl:
        print("{}  winners_lost~{:>4.0%}: theta={:.4f} "
              "winners_lost={:>6.1%} search_avoided={:>6.1%}".format(
                  indent, target_wl, theta, actual_wl, avoided), flush=True)
    print("{}[fixando search_avoided -> winners_lost]".format(indent))
    for target_av, theta, actual_avoided, wl in rows_by_avoided:
        print("{}  search_avoided~{:>4.0%}: theta={:.4f} "
              "search_avoided={:>6.1%} winners_lost={:>6.1%}".format(
                  indent, target_av, theta, actual_avoided, wl), flush=True)


def evaluate_dim(net, feat, truth, device):
    """net ja tem a padronizacao dobrada na 1a camada -> alimenta features
    cruas. Retorna p_ext, metricas de separabilidade e a tabela de troca."""
    net.eval()
    with torch.no_grad():
        fb = torch.tensor(feat, dtype=torch.float32, device=device)
        logits = net(fb).cpu().numpy()
    probs = _softmax(logits)
    p_ext = probs[:, CLASS_EXT]

    n = len(truth)
    base_rate = float((truth == CLASS_EXT).mean()) if n > 0 else float("nan")
    roc_auc, pr_auc = compute_aucs(truth, p_ext)
    rows_by_wl, rows_by_avoided = tradeoff_table(p_ext, truth)

    return {
        "n": n,
        "base_rate": base_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "rows_by_wl": rows_by_wl,
        "rows_by_avoided": rows_by_avoided,
        "p_ext": p_ext,
        "truth": truth,
    }


# --------------------------------------------------------------------------
def main(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="/workspace/results/dataset_h9")
    p.add_argument("--train-seqs", nargs="+", default=TRAIN_SEQS)
    p.add_argument("--held-out-seqs", nargs="+", default=HELD_OUT_SEQS)
    p.add_argument("--out-dir",
                   default="/workspace/results/models/h9d_predictability")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--per-pkl-train", type=int, default=3000)
    p.add_argument("--per-pkl-eval", type=int, default=2000)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    print("sklearn disponivel:" , _HAS_SKLEARN, flush=True)

    entries = datamod.discover_pkls(args.dataset_dir)
    print("descobertos {} pkl(s) em {}".format(len(entries), args.dataset_dir),
         flush=True)

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
              "classes": NUM_EXT_CLASSES}

    per_dim_eval = {}
    all_p_ext, all_truth = [], []

    for dim, _ in MODEL_LEVELS:
        rec = train_data[dim]
        if len(rec["truth"]) == 0:
            print("[dim {}] sem amostras de treino, pulando".format(dim))
            continue
        print("\n=== treino dim {}px (n={}) ===".format(dim, len(rec["truth"])),
             flush=True)
        counts = np.bincount(rec["truth"], minlength=NUM_EXT_CLASSES).tolist()
        print("  counts(NAO_EXT/EXT)={}".format(counts), flush=True)
        net, norm = train_ext_student(rec, args.hidden, device, args.epochs,
                                      args.lr, batch=args.batch)
        bundle["students"][dim] = net.state_dict()
        bundle["norm"][dim] = norm

        erec = eval_data[dim]
        if len(erec["truth"]) == 0:
            print("[dim {}] sem amostras held-out, pulando avaliacao".format(dim))
            continue
        ev = evaluate_dim(net, erec["feat"], erec["truth"], device)
        per_dim_eval[dim] = ev
        print("\n--- nivel {}px (held-out) ---".format(dim), flush=True)
        print("  n={} base_rate(EXT)={:.1%} roc_auc={:.4f} pr_auc={:.4f}".format(
            ev["n"], ev["base_rate"], ev["roc_auc"], ev["pr_auc"]), flush=True)
        print_tradeoff(ev["rows_by_wl"], ev["rows_by_avoided"])

        all_p_ext.append(ev["p_ext"])
        all_truth.append(ev["truth"])

    students_path = os.path.join(args.out_dir, "students.pt")
    torch.save(bundle, students_path)
    print("\nSalvo ->", students_path, flush=True)

    if all_p_ext:
        agg_p_ext = np.concatenate(all_p_ext)
        agg_truth = np.concatenate(all_truth)
    else:
        agg_p_ext = np.array([], dtype=np.float32)
        agg_truth = np.array([], dtype=np.int64)

    agg_n = len(agg_truth)
    agg_base_rate = float((agg_truth == CLASS_EXT).mean()) if agg_n > 0 else float("nan")
    agg_roc_auc, agg_pr_auc = compute_aucs(agg_truth, agg_p_ext)
    agg_rows_by_wl, agg_rows_by_avoided = tradeoff_table(agg_p_ext, agg_truth)

    print("\n=== AGREGADO (todos os niveis, held-out) ===", flush=True)
    print("n={} base_rate(EXT)={:.1%} roc_auc={:.4f} pr_auc={:.4f}".format(
        agg_n, agg_base_rate, agg_roc_auc, agg_pr_auc), flush=True)
    print_tradeoff(agg_rows_by_wl, agg_rows_by_avoided)

    # Veredito: no alvo winners_lost~10% agregado, o search_avoided
    # atingido chega a >=50%?
    avoided_at_wl10 = float("nan")
    for target_wl, _theta, _actual_wl, avoided in agg_rows_by_wl:
        if abs(target_wl - 0.10) < 1e-9:
            avoided_at_wl10 = avoided
            break

    print("\n=== VEREDITO ===", flush=True)
    if not np.isnan(avoided_at_wl10) and avoided_at_wl10 >= 0.50:
        print("SEPARAVEL: features distinguem EXT -- H9d seletivo "
              "promissor (winners_lost~10% -> search_avoided={:.1%} "
              "agregado)".format(avoided_at_wl10), flush=True)
    else:
        print("FRACO: features nao separam EXT bem -- seletivo ~ blanket "
              "(winners_lost~10% -> search_avoided={} agregado)".format(
                  "{:.1%}".format(avoided_at_wl10)
                  if not np.isnan(avoided_at_wl10) else "n/d"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
