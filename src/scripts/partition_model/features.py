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

Hierarchical context (v3, D=24): the surrogate sees the whole superblock while
the v2 student only saw its own block -- a plausible share of the student vs
surrogate gap (Pre-H7 lever A4). node_features() adds the parent 2n x 2n block's
texture, the contrast against the three sibling quadrants, and the block's
position inside the 64x64 unit. All of it reads only source pixels the C side
already has (parent = pointer arithmetic from the block's mi coordinates), so no
re-instrumentation of the dataset is needed.

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
Hierarchical context (node_features only; at n=64 the parent is defined as the
block itself and the siblings as copies of it, zeroing the contrasts):
  18  log1p(parent var)              texture of the containing 2n x 2n block
  19  (var-pvar)/(var+pvar+1)        block-vs-parent texture contrast
  20  log1p(mean sibling var)        texture of the 3 sibling quadrants
  21  (var-maxsib)/(var+maxsib+1)    block-vs-worst-sibling contrast
  22  (cell_r * n) / 64              vertical position inside the 64px unit
  23  (cell_c * n) / 64              horizontal position inside the 64px unit
"""

import numpy as np

NUM_FEATURES = 24
EDGE_THRESH = 16
FEATURE_NAMES = [
    "log_var", "log_var_q0", "log_var_q1", "log_var_q2", "log_var_q3",
    "quad_var_spread", "log_var_qsums", "log_var_qvars",
    "log_hgrad", "log_vgrad", "grad_orient",
    "log_var_rowsums", "log_var_colsums", "rowcol_orient",
    "log_maxgrad", "edge_density", "mean_norm", "q_norm",
    "log_parent_var", "parent_contrast", "log_sib_mean_var", "sib_max_contrast",
    "pos_r", "pos_c",
]
NUM_BLOCK_FEATURES = 18  # features computed from the block alone (0..17)


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


def _block_feats18(b, qindex, out):
    """Fill out[0:18] from one int64 block; return the block's variance."""
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

    out[0] = np.log1p(var)
    out[1] = np.log1p(qvars[0])
    out[2] = np.log1p(qvars[1])
    out[3] = np.log1p(qvars[2])
    out[4] = np.log1p(qvars[3])
    out[5] = spread
    out[6] = np.log1p(_var_of(qsums))
    out[7] = np.log1p(_var_of(qvars))
    out[8] = np.log1p(hgrad)
    out[9] = np.log1p(vgrad)
    out[10] = orient
    out[11] = np.log1p(vrow)
    out[12] = np.log1p(vcol)
    out[13] = rowcol
    out[14] = np.log1p(maxgrad)
    out[15] = edge_density
    out[16] = mean / 255.0
    out[17] = qindex / 255.0
    return var


def block_features(luma, qindex):
    """v2-compatible vector: block-only features, context slots neutralized
    (parent := the block itself). Prefer node_features() when the superblock
    is available."""
    b = luma.astype(np.int64)
    f = np.empty(NUM_FEATURES, dtype=np.float32)
    var = _block_feats18(b, qindex, f)
    _context_feats(f, var, var, [var, var, var], 0.0, 0.0)
    return f


def _context_feats(f, var, pvar, sib_vars, pos_r, pos_c):
    """Fill f[18:24]. All doubles; ordering fixed to match the C side."""
    sib_mean = (sib_vars[0] + sib_vars[1] + sib_vars[2]) / 3.0
    sib_max = max(sib_vars)
    f[18] = np.log1p(pvar)
    f[19] = (var - pvar) / (var + pvar + 1.0)
    f[20] = np.log1p(sib_mean)
    f[21] = (var - sib_max) / (var + sib_max + 1.0)
    f[22] = pos_r
    f[23] = pos_c


def node_features(sb_luma, dim, r, c, qindex):
    """Feature vector for the (dim, r, c) node of one 64x64 superblock.

    sb_luma: (64,64) uint8; dim in {64,32,16,8}; (r,c) grid cell at that level.
    Parent/sibling context comes from the containing 2*dim block; at dim=64 the
    parent is the block itself (contrasts collapse to 0), keeping the formula
    uniform on both the Python and C sides."""
    sb = np.asarray(sb_luma).astype(np.int64)
    n = dim
    block = sb[r * n:(r + 1) * n, c * n:(c + 1) * n]
    f = np.empty(NUM_FEATURES, dtype=np.float32)
    var = _block_feats18(block, qindex, f)

    if n == 64:
        pvar, sib_vars = var, [var, var, var]
    else:
        pn = 2 * n
        pr, pc = r // 2, c // 2
        parent = sb[pr * pn:(pr + 1) * pn, pc * pn:(pc + 1) * pn]
        pvar = _int_var(parent)
        dr, dc = r & 1, c & 1
        sib_vars = []
        for qr in range(2):
            for qc in range(2):
                if qr == dr and qc == dc:
                    continue
                quad = parent[qr * n:(qr + 1) * n, qc * n:(qc + 1) * n]
                sib_vars.append(_int_var(quad))
    _context_feats(f, var, pvar, sib_vars, (r * n) / 64.0, (c * n) / 64.0)
    return f


def batch_features(luma_list, qindex_arr):
    out = np.empty((len(luma_list), NUM_FEATURES), dtype=np.float32)
    for i, luma in enumerate(luma_list):
        out[i] = block_features(np.asarray(luma), int(qindex_arr[i]))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sb = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    for dim, side in ((64, 1), (32, 2), (16, 4)):
        print("{:>2}px (0,{}):".format(dim, side - 1),
              np.round(node_features(sb, dim, 0, side - 1, 128), 3))
