#!/usr/bin/env python3
"""Distilled student model: a small per-block-size MLP that reuses libaom's
av1_nn_predict at inference time (no new C inference code).

The student consumes the handcrafted features (features.py) and emits 3 logits
-> softmax = [P(NONE), P(SPLIT), P(REST)], the only quantities the Fase D
pruning policy needs. Topology (hidden widths) is a calibration knob.

Weight layout matches av1_nn_predict exactly: for each layer the C code computes
    val[node] = bias[node] + sum_i weights[node * num_inputs + i] * input[i]
with ReLU on hidden layers and a linear output layer. PyTorch nn.Linear stores
weight as (out, in) row-major -> weight.reshape(-1) is precisely
weights[node * num_inputs + i]. So export is a direct dump.
"""

import numpy as np

# Collapse the 10 AV1 PARTITION_TYPEs to the 3 decisions the pruner cares about.
CLASS_NONE, CLASS_SPLIT, CLASS_REST = 0, 1, 2
NUM_STUDENT_CLASSES = 3


def collapse_label(part10):
    """PARTITION_TYPE (0..9) -> student class (NONE / SPLIT / REST)."""
    if part10 == 0:
        return CLASS_NONE
    if part10 == 3:
        return CLASS_SPLIT
    return CLASS_REST


def collapse_probs(p10):
    """10-class probability vector -> 3-class [P(NONE), P(SPLIT), P(REST)]."""
    p10 = np.asarray(p10, dtype=np.float64)
    rest = p10.sum() - p10[0] - p10[3]
    out = np.array([p10[0], p10[3], rest], dtype=np.float64)
    s = out.sum()
    return out / s if s > 0 else out


def make_student(in_features, hidden):
    import torch.nn as nn
    layers, prev = [], in_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        prev = h
    layers += [nn.Linear(prev, NUM_STUDENT_CLASSES)]
    return nn.Sequential(*layers)


def export_nn_config(model):
    """Extract av1_nn_predict-compatible parameters from a student Sequential.

    Returns dict: num_inputs, num_outputs, num_hidden_layers, num_hidden_nodes,
    weights (list of 1-D float32 arrays, row-major out*in), bias (list)."""
    import torch.nn as nn
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    weights, bias, hidden = [], [], []
    for k, lin in enumerate(linears):
        w = lin.weight.detach().cpu().numpy().astype(np.float32)  # (out, in)
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        weights.append(w.reshape(-1))  # row-major == weights[node*in + i]
        bias.append(b)
        if k < len(linears) - 1:
            hidden.append(lin.out_features)
    return {
        "num_inputs": linears[0].in_features,
        "num_outputs": linears[-1].out_features,
        "num_hidden_layers": len(linears) - 1,
        "num_hidden_nodes": hidden,
        "weights": weights,
        "bias": bias,
    }
