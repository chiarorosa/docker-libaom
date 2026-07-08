#!/usr/bin/env python3
"""AV1 partitioning definitions shared across the surrogate pipeline.

Single source of truth for:
  * PARTITION_TYPE enum (index == label value logged by the C instrumentation),
  * which partition types are legal at each square block size (the validity mask
    the model applies before softmax / cross-entropy),
  * the superblock quadtree geometry (how a 64x64 unit maps to per-level grids),
  * the geometric cost matrix that encodes RD-asymmetry of misclassification.

Why a cost matrix (not plain cross-entropy): in rate-distortion terms, confusing
two geometrically-similar partitions (e.g. HORZ vs HORZ_A) costs little, while
confusing opposite decisions (HORZ vs VERT, or NONE vs SPLIT) wrecks the codec.
Plain CE treats every mistake equally; we instead build soft targets whose mass
leaks toward geometrically-near partitions in proportion to exp(-cost/T).

All weights here are INITIAL proposals to be calibrated with the training gate,
not final values.
"""

import numpy as np

# PARTITION_TYPE (av1/common/enums.h). Index == label value in PartitionSample.
PARTITION_NAMES = [
    "NONE", "HORZ", "VERT", "SPLIT",
    "HORZ_A", "HORZ_B", "VERT_A", "VERT_B",
    "HORZ_4", "VERT_4",
]
NUM_PARTITION_TYPES = len(PARTITION_NAMES)  # 10

# Square block sizes logged by the instrumentation, coarsest -> finest.
BLOCK_DIMS = [64, 32, 16, 8]

# Superblock unit modeled by the surrogate: one 64x64 quadtree. In mode-info
# units (4 px each) a 64x64 block spans 16 mi; each level is a grid of cells.
SB_SIZE_PX = 64
MI_SIZE = 4
SB_SIZE_MI = SB_SIZE_PX // MI_SIZE  # 16

# Per level: (block_dim px, grid side). Level 64 is the 1x1 root; 8 is 8x8.
# LEVELS spans the whole tree (used to assemble superblocks and to count the
# search a NONE-commit cuts). 8x8 is a terminal leaf in the 4K All-Intra regime
# (sub-8x8 is never explored -> 8x8 is ALWAYS NONE), so it is NOT a decision
# point: it is excluded from the MODEL (heads/loss/student/pruning) but kept in
# the tree because pruning a 16x16 to NONE saves exactly its four 8x8 children.
LEVELS = [
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
]

# Levels the surrogate/student actually predict and the C pruner acts on.
MODEL_LEVELS = [(64, 1), (32, 2), (16, 4)]

# Legal partition sets per block size (av1 partitioning rules):
#   - NONE/HORZ/VERT/SPLIT are legal for every square block >= 8x8.
#   - AB partitions (HORZ_A/B, VERT_A/B) and 4-way (HORZ_4, VERT_4) require
#     bsize > 8x8, i.e. they never occur at 8x8.
# Encoded as a boolean legality mask per block_dim (True == legal class).
_BASIC = {0, 1, 2, 3}  # NONE, HORZ, VERT, SPLIT
_ALL = set(range(NUM_PARTITION_TYPES))


def legal_classes(block_dim):
    """Set of legal PARTITION_TYPE indices for a square block of `block_dim` px."""
    return set(_BASIC) if block_dim == 8 else set(_ALL)


def legality_mask(block_dim):
    """Boolean vector (len 10); True where the class is legal for this size."""
    legal = legal_classes(block_dim)
    return np.array([c in legal for c in range(NUM_PARTITION_TYPES)], dtype=bool)


# --- Geometric descriptors of each partition type -------------------------
# orient: -1 horizontal-dominant, +1 vertical-dominant, 0 isotropic.
#   (HORZ* cut the block with horizontal lines; VERT* with vertical lines.)
# gran: subdivision granularity, 0 (NONE) .. 3 (SPLIT / 4-way).
_ORIENT = {
    "NONE": 0, "SPLIT": 0,
    "HORZ": -1, "HORZ_A": -1, "HORZ_B": -1, "HORZ_4": -1,
    "VERT": +1, "VERT_A": +1, "VERT_B": +1, "VERT_4": +1,
}
_GRAN = {
    "NONE": 0,
    "HORZ": 1, "VERT": 1,
    "HORZ_A": 2, "HORZ_B": 2, "VERT_A": 2, "VERT_B": 2,
    "SPLIT": 3, "HORZ_4": 3, "VERT_4": 3,
}


def geometric_cost_matrix(w_orient=1.0, w_gran=0.5, normalize=True):
    """Symmetric 10x10 cost matrix C[i, j] = geometric dissimilarity of the
    partitions, 0 on the diagonal. Orientation disagreement dominates; a
    granularity gap (e.g. NONE vs SPLIT) is the next-largest penalty.

    w_orient / w_gran are calibration knobs (initial proposal 1.0 / 0.5, which
    makes HORZ<->VERT the worst confusion and NONE<->SPLIT the next-worst).
    """
    o = np.array([_ORIENT[n] for n in PARTITION_NAMES], dtype=np.float32)
    g = np.array([_GRAN[n] for n in PARTITION_NAMES], dtype=np.float32)
    cost = (w_orient * np.abs(o[:, None] - o[None, :])
            + w_gran * np.abs(g[:, None] - g[None, :]))
    np.fill_diagonal(cost, 0.0)
    if normalize:
        m = cost.max()
        if m > 0:
            cost = cost / m
    return cost.astype(np.float32)


def geometric_soft_targets(cost, temp=0.5, eps=0.1):
    """Per-truth soft label distributions from the cost matrix.

    For truth class t, target = (1-eps)*onehot(t) + eps*softmax(-cost[t]/temp),
    so most mass stays on the true class but a fraction leaks toward
    geometrically-near partitions. Returns a (10, 10) row-stochastic matrix
    `T[t]` used as the cross-entropy target for a sample whose truth is t.

    Note: legality is NOT applied here (some near classes may be illegal for a
    given block size); the model masks illegal logits, and callers restrict the
    target to legal classes per level before use.
    """
    logits = -cost / max(temp, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    soft = np.exp(logits)
    soft = soft / soft.sum(axis=1, keepdims=True)
    onehot = np.eye(cost.shape[0], dtype=np.float32)
    return ((1.0 - eps) * onehot + eps * soft).astype(np.float32)


def sb_cell(block_dim, mi_row, mi_col, sb_mi_row, sb_mi_col):
    """Grid cell (row, col) of a block within its 64x64 superblock, for the
    level implied by `block_dim`. mi_* are absolute mode-info coordinates;
    sb_mi_* are the superblock origin (multiples of SB_SIZE_MI)."""
    cells_per_side = SB_SIZE_PX // block_dim  # 1,2,4,8
    mi_per_cell = SB_SIZE_MI // cells_per_side  # 16,8,4,2
    r = (mi_row - sb_mi_row) // mi_per_cell
    c = (mi_col - sb_mi_col) // mi_per_cell
    return int(r), int(c)
