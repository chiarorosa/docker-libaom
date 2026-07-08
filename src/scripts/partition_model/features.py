#!/usr/bin/env python3
"""Handcrafted block features for the distilled C-side student model.

SINGLE SOURCE OF TRUTH for the feature vector. The student MLP runs inside libaom
(av1_nn_predict) at every partition node, so every formula must be transliterable
to C with identical arithmetic. All statistics build from integer pixel sums
(exact in int64 Python and int64_t C); only final log1p / ratios are float, where
a tiny delta is harmless to the MLP (a Fase D parity check confirms agreement).

Richer set (v2, D=18): a diagnostic showed the pixel ConvNeXt discriminates
NONE/SPLIT far better than the original 10 features, so this adds inter-sub-block
heterogeneity (variance of quadrant means/variances), directional row/column
profiles (HORZ vs VERT cues), and edge strength/density.

Given a square block of side n (uint8 luma) and base_qindex:
   0  log1p(var)                     overall texture / flatness
   1..4 log1p(var) of 4 quadrants    per-quadrant texture
   5  (max_qv - min_qv)/(max_qv+1)   quadrant-variance spread
   6  log1p(var of quadrant sums)    brightness heterogeneity across quadrants
   7  log1p(var of quadrant vars)    texture heterogeneity across quadrants
   8  log1p(horizontal grad sum)     vertical-edge energy
   9  log1p(vertical grad sum)       horizontal-edge energy
  10  (hgrad - vgrad)/(hgrad+vgrad+1) gradient orientation in [-1,1]
  11  log1p(var of row sums)         horizontal-band structure (HORZ cue)
  12  log1p(var of col sums)         vertical-band structure (VERT cue)
  13  (vrow - vcol)/(vrow+vcol+1)    row-vs-col structure orientation
  14  log1p(max |grad|)              strongest edge
  15  strong-edge density in [0,1]   fraction of |grad| > 16
  16  mean / 255                     DC level
  17  qindex / 255                   quantization strength
"""

import numpy as np

NUM_FEATURES = 18
EDGE_THRESH = 16
FEATURE_NAMES = [
    "log_var", "log_var_q0", "log_var_q1", "log_var_q2", "log_var_q3",
    "quad_var_spread", "log_var_qsums", "log_var_qvars",
    "log_hgrad", "log_vgrad", "grad_orient",
    "log_var_rowsums", "log_var_colsums", "rowcol_orient",
    "log_maxgrad", "edge_density", "mean_norm", "q_norm",
]


def _int_var(block_i64):
    """Population variance via exact integer sums: (N*sumsq - sum^2) / N^2."""
    n = block_i64.size
    s = int(block_i64.sum())
    ss = int((block_i64 * block_i64).sum())
    return (n * ss - s * s) / float(n * n)


def _var_of(vals):
    """Population variance of a short list of numbers (double precision)."""
    a = np.asarray(vals, dtype=np.float64)
    m = a.mean()
    return float(((a - m) ** 2).mean())


def block_features(luma, qindex):
    """Feature vector (float32, NUM_FEATURES) for one square uint8 block."""
    b = luma.astype(np.int64)
    n = b.shape[0]
    half = n // 2

    var = _int_var(b)
    quads = [b[:half, :half], b[:half, half:], b[half:, :half], b[half:, half:]]
    qsums = [int(q.sum()) for q in quads]
    qvars = [_int_var(q) for q in quads]
    qmax, qmin = max(qvars), min(qvars)
    spread = (qmax - qmin) / (qmax + 1.0)

    hdiff = np.abs(b[:, 1:] - b[:, :-1])
    vdiff = np.abs(b[1:, :] - b[:-1, :])
    hgrad = int(hdiff.sum())
    vgrad = int(vdiff.sum())
    maxgrad = int(max(hdiff.max(), vdiff.max()))
    strong = int((hdiff > EDGE_THRESH).sum() + (vdiff > EDGE_THRESH).sum())
    num_grads = hdiff.size + vdiff.size
    edge_density = strong / float(num_grads)
    orient = (hgrad - vgrad) / (hgrad + vgrad + 1.0)

    rowsums = b.sum(axis=1)
    colsums = b.sum(axis=0)
    vrow = _var_of(rowsums)
    vcol = _var_of(colsums)
    rowcol = (vrow - vcol) / (vrow + vcol + 1.0)

    mean = int(b.sum()) / float(n * n)

    f = np.empty(NUM_FEATURES, dtype=np.float32)
    f[0] = np.log1p(var)
    f[1] = np.log1p(qvars[0])
    f[2] = np.log1p(qvars[1])
    f[3] = np.log1p(qvars[2])
    f[4] = np.log1p(qvars[3])
    f[5] = spread
    f[6] = np.log1p(_var_of(qsums))
    f[7] = np.log1p(_var_of(qvars))
    f[8] = np.log1p(hgrad)
    f[9] = np.log1p(vgrad)
    f[10] = orient
    f[11] = np.log1p(vrow)
    f[12] = np.log1p(vcol)
    f[13] = rowcol
    f[14] = np.log1p(maxgrad)
    f[15] = edge_density
    f[16] = mean / 255.0
    f[17] = qindex / 255.0
    return f


def batch_features(luma_list, qindex_arr):
    out = np.empty((len(luma_list), NUM_FEATURES), dtype=np.float32)
    for i, luma in enumerate(luma_list):
        out[i] = block_features(np.asarray(luma), int(qindex_arr[i]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n in (8, 16, 32, 64):
        blk = rng.integers(0, 256, size=(n, n), dtype=np.uint8)
        print("{:>2}px:".format(n), np.round(block_features(blk, 128), 3))
