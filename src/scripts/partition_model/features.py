#!/usr/bin/env python3
"""Handcrafted block features for the distilled C-side student model.

This is the SINGLE SOURCE OF TRUTH for the feature vector. The student MLP runs
inside libaom (av1_nn_predict) at every partition node, so every formula here
must be transliterable to C with identical arithmetic. All statistics are built
from integer pixel sums (exact in both int64 Python and int64_t C); only the
final log1p / ratio steps are floating point, where a tiny delta is harmless to
the MLP (a parity check in Fase D confirms agreement).

Feature vector (initial proposal, D=10 -- calibrated by importance at the Fase C
gate). Given a square block of side n (uint8 luma) and its base_qindex:

  0  log1p(var)                    coarse texture / flatness
  1..4 log1p(var) of 4 quadrants   spatial variance layout (split cue)
  5  (max_q_var - min_q_var)/(max+1)  quadrant variance spread (0..1)
  6  log1p(horizontal grad sum)    energy of vertical edges
  7  log1p(vertical grad sum)      energy of horizontal edges
  8  (h_grad - v_grad)/(h_grad+v_grad+1)  orientation cue in [-1,1]
  9  qindex / 255                  quantization strength (monotone w/ dc_quant)
"""

import numpy as np

NUM_FEATURES = 10
FEATURE_NAMES = [
    "log_var", "log_var_q0", "log_var_q1", "log_var_q2", "log_var_q3",
    "quad_var_spread", "log_hgrad", "log_vgrad", "orient", "q_norm",
]


def _int_var(block_i64):
    """Population variance via exact integer sums: (n*sumsq - sum^2) / n^2."""
    n = block_i64.size
    s = int(block_i64.sum())
    ss = int((block_i64 * block_i64).sum())
    return (n * ss - s * s) / float(n * n)


def _grad_sums(block_i64):
    """Sum of absolute horizontal and vertical first differences (integers)."""
    h = int(np.abs(block_i64[:, 1:] - block_i64[:, :-1]).sum())
    v = int(np.abs(block_i64[1:, :] - block_i64[:-1, :]).sum())
    return h, v


def block_features(luma, qindex):
    """Feature vector (float32, NUM_FEATURES) for one square uint8 block."""
    b = luma.astype(np.int64)
    n = b.shape[0]
    half = n // 2

    var = _int_var(b)
    quads = [b[:half, :half], b[:half, half:], b[half:, :half], b[half:, half:]]
    qvars = [_int_var(q) for q in quads]
    qmax, qmin = max(qvars), min(qvars)
    spread = (qmax - qmin) / (qmax + 1.0)

    h, v = _grad_sums(b)
    orient = (h - v) / (h + v + 1.0)

    feats = np.empty(NUM_FEATURES, dtype=np.float32)
    feats[0] = np.log1p(var)
    for i, qv in enumerate(qvars):
        feats[1 + i] = np.log1p(qv)
    feats[5] = spread
    feats[6] = np.log1p(h)
    feats[7] = np.log1p(v)
    feats[8] = orient
    feats[9] = qindex / 255.0
    return feats


def batch_features(luma_list, qindex_arr):
    """Stack features for a list/array of blocks (variable side lengths OK)."""
    out = np.empty((len(luma_list), NUM_FEATURES), dtype=np.float32)
    for i, luma in enumerate(luma_list):
        out[i] = block_features(np.asarray(luma), int(qindex_arr[i]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in (8, 16, 32, 64):
        blk = rng.integers(0, 256, size=(n, n), dtype=np.uint8)
        f = block_features(blk, 128)
        print("{:>2}px:".format(n), np.round(f, 3))
