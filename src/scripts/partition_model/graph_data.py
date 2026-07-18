#!/usr/bin/env python3
"""Approach B (Stage 1): monta o grafo do quadtree de cada superbloco a partir dos
pkls dataset_h9. Nós = nós de decisão (64/32/16) com as 36 features H9a; arestas =
pai<->filho e irmão<->irmão (não-causal, para o diagnóstico). Rótulo 3-classes
(NONE/SPLIT/REST). Também define a variante causal (sb_edges_causal): arestas
dirigidas apenas a partir de nós já decididos -- pai e irmãos de raster anterior --
para a ablação causal. Ver docs/superpowers/specs/2026-07-17-approachB-gnn-estrutural-design.md."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as datamod  # noqa: E402
import features as featmod  # noqa: E402
import student as studentmod  # noqa: E402
from regret import _children  # noqa: E402

DECISION_DIMS = (64, 32, 16)


def sb_edges(node_keys):
    """Arestas não-direcionadas (ambos os sentidos) entre índices de node_keys:
    pai<->filho e irmão<->irmão. Só entre keys presentes."""
    idx = {k: i for i, k in enumerate(node_keys)}
    edges = set()
    for k in node_keys:
        dim, r, c = k
        for ch in _children(dim, r, c):
            if ch in idx:
                edges.add((idx[k], idx[ch]))
                edges.add((idx[ch], idx[k]))
        # irmãos: filhos do mesmo pai.
        if dim < 64:
            parent = (2 * dim, r // 2, c // 2)
            for sib in _children(*parent):
                if sib != k and sib in idx:
                    edges.add((idx[k], idx[sib]))
    return sorted(edges)


def sb_edges_causal(node_keys):
    """Arestas DIRIGIDAS (src, dst), só para dentro de nós já 'decididos' na ordem
    top-down + raster do codificador: pai->filho e irmão-anterior->irmão-posterior
    (raster local no bloco 2x2 do pai: TL=0, TR=1, BL=2, BR=3). Sem aresta
    filho->pai nem irmão-futuro->irmão-passado. A raiz (dim=64) não recebe
    arestas. Só entre keys presentes em node_keys."""
    idx = {k: i for i, k in enumerate(node_keys)}
    edges = set()
    for k in node_keys:
        dim, r, c = k
        for ch in _children(dim, r, c):
            if ch in idx:
                edges.add((idx[k], idx[ch]))          # pai -> filho
        if dim < 64:
            parent = (2 * dim, r // 2, c // 2)
            raster_k = (r % 2) * 2 + (c % 2)
            for sib in _children(*parent):
                if sib == k or sib not in idx:
                    continue
                sr, sc = sib[1], sib[2]
                raster_sib = (sr % 2) * 2 + (sc % 2)
                if raster_sib < raster_k:
                    edges.add((idx[sib], idx[k]))      # irmão-anterior -> k
    return sorted(edges)


def build_graph_dataset(entries, per_pkl=None, limit=None, causal=False):
    """Lista de grafos por superbloco: {x:(n,36), y:(n,), level:(n,), edge_index:(2,E)}."""
    edge_fn = sb_edges_causal if causal else sb_edges
    graphs = []
    for e in entries:
        took = 0
        for sb in datamod.iter_superblock_members(e["path"]):
            if not sb.get("has_rd"):
                raise SystemExit("{}: sem contexto RD (dataset_h9).".format(e["path"]))
            keys, feats, ys, levels = [], [], [], []
            for k, (dim, r, c, _luma, label) in enumerate(sb["members"]):
                if dim not in DECISION_DIMS:
                    continue
                keys.append((dim, r, c))
                feats.append(featmod.node_features_h9a(
                    sb["luma"], dim, r, c, sb["qindex"], sb["ctx"][k]))
                ys.append(studentmod.collapse_label(int(label)))
                levels.append(dim)
            if not keys:
                continue
            e_list = edge_fn(keys)
            ei = (np.asarray(e_list, dtype=np.int64).T
                  if e_list else np.zeros((2, 0), np.int64))
            graphs.append({"x": np.asarray(feats, np.float32),
                           "y": np.asarray(ys, np.int64),
                           "level": np.asarray(levels, np.int64),
                           "edge_index": ei})
            took += 1
            if per_pkl and took >= per_pkl:
                break
            if limit and len(graphs) >= limit:
                return graphs
    return graphs


def collate(graphs):
    """Concatena grafos em um batch bloco-diagonal (MP não cruza superblocos)."""
    xs, ys, lv, eis, off = [], [], [], [], 0
    for g in graphs:
        xs.append(g["x"]); ys.append(g["y"]); lv.append(g["level"])
        if g["edge_index"].shape[1] > 0:
            eis.append(g["edge_index"] + off)
        off += g["x"].shape[0]
    return {"x": np.concatenate(xs, 0),
            "y": np.concatenate(ys, 0),
            "level": np.concatenate(lv, 0),
            "edge_index": (np.concatenate(eis, 1) if eis
                           else np.zeros((2, 0), np.int64))}
