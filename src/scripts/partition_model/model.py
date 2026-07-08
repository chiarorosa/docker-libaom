#!/usr/bin/env python3
"""Multi-level ConvNeXt surrogate for AV1 partitioning.

One 64x64 superblock (luma + a qindex plane) goes in; the model predicts the
partition decision at every quadtree level in a single pass, mirroring AV1's
recursive superblock -> 32 -> 16 -> 8 structure:

    input (B,2,64,64)
      ConvNeXt stages ->  f8 (8x8), f4 (4x4), f2 (2x2)   [multi-scale features]
      top-down fusion  ->  p8, p4, p2                     [context flows down]
      per-level heads  ->  logits at 64 (1x1), 32 (2x2), 16 (4x4), 8 (8x8)

Each head's illegal classes for its block size are masked out before softmax.
Backbone width / fusion dim are constructor args (calibration knobs), defaulting
to ConvNeXt-Tiny trained from scratch on 1-channel luma (RGB pretraining is a
poor domain match; a stem-averaged init is only an ablation).
"""

import torch
import torch.nn as nn

from partition_defs import MODEL_LEVELS, NUM_PARTITION_TYPES, legality_mask

# torchvision ConvNeXt.features stage output indices and their channel widths.
# We only need stage2 (4x4, feeds the 16px head) and stage3 (2x2, feeds the 32px
# head and the pooled 64px head); the 8x8 map/head is dropped (8x8 not modeled).
_CAPTURE = {5: "f4", 7: "f2"}
_NEG = -1e4  # additive mask for illegal classes


def _backbone(variant, drop_path=0.0):
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base)
    ctor = {"tiny": convnext_tiny, "small": convnext_small,
            "base": convnext_base}[variant]
    # No pretrained weights (luma domain); stochastic depth off -- we underfit,
    # so we drop the ImageNet-tuned regularizer that blocks fitting.
    net = ctor(weights=None, stochastic_depth_prob=drop_path)
    feats = net.features
    # Widths of the captured stages, read off the constructed network.
    chans = {}
    x = torch.zeros(1, 3, 64, 64)
    for i, block in enumerate(feats):
        x = block(x)
        if i in _CAPTURE:
            chans[_CAPTURE[i]] = x.shape[1]
    return feats, chans


class PartitionSurrogate(nn.Module):
    def __init__(self, variant="tiny", fusion_dim=128, in_channels=2):
        super().__init__()
        self.features, chans = _backbone(variant)
        # Adapt the stem to `in_channels` (luma + qindex), trained from scratch.
        stem_conv = self.features[0][0]
        new_conv = nn.Conv2d(in_channels, stem_conv.out_channels,
                             kernel_size=stem_conv.kernel_size,
                             stride=stem_conv.stride,
                             padding=stem_conv.padding)
        self.features[0][0] = new_conv

        d = fusion_dim
        self.lat_f2 = nn.Conv2d(chans["f2"], d, 1)  # 2x2
        self.lat_f4 = nn.Conv2d(chans["f4"], d, 1)  # 4x4
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # GELU on the fused maps (matches the ConvNeXt backbone; free here since
        # the surrogate is PyTorch-only). The student MLP, by contrast, is locked
        # to ReLU because av1_nn_predict implements only ReLU.
        self.act = nn.GELU()

        # One head per MODEL level. 64x64 is a single node -> a linear head on the
        # globally pooled deepest map; 32/16 are 1x1 convs over their grid.
        self.head_64 = nn.Linear(d, NUM_PARTITION_TYPES)
        self.head_32 = nn.Conv2d(d, NUM_PARTITION_TYPES, 1)  # on p2 (2x2)
        self.head_16 = nn.Conv2d(d, NUM_PARTITION_TYPES, 1)  # on p4 (4x4)

        # Additive legality mask per level (0 legal, -1e4 illegal).
        for dim, _ in MODEL_LEVELS:
            m = torch.zeros(NUM_PARTITION_TYPES)
            m[~legality_mask(dim)] = _NEG
            self.register_buffer("legal_{}".format(dim), m)

    def _multiscale(self, x):
        feats = {}
        for i, block in enumerate(self.features):
            x = block(x)
            if i in _CAPTURE:
                feats[_CAPTURE[i]] = x
        return feats["f4"], feats["f2"]

    def forward(self, x):
        f4, f2 = self._multiscale(x)               # 4x4, 2x2
        p2 = self.lat_f2(f2)                        # 2x2 (raw laterals summed
        p4 = self.lat_f4(f4) + self.up(p2)         # 4x4  top-down, FPN style)
        a2, a4 = self.act(p2), self.act(p4)
        g = a2.mean(dim=(2, 3))                     # (B, d)

        def grid(logits):  # (B,C,H,W) -> (B,H,W,C)
            return logits.permute(0, 2, 3, 1).contiguous()

        out = {
            64: self.head_64(g).view(-1, 1, 1, NUM_PARTITION_TYPES),
            32: grid(self.head_32(a2)),
            16: grid(self.head_16(a4)),
        }
        for dim, _ in MODEL_LEVELS:
            out[dim] = out[dim] + getattr(self, "legal_{}".format(dim))
        return out


if __name__ == "__main__":
    # Shape smoke test (no data, no CUDA needed).
    net = PartitionSurrogate()
    y = net(torch.zeros(2, 2, 64, 64))
    for dim, _ in MODEL_LEVELS:
        print("level {:>2}px logits: {}".format(dim, tuple(y[dim].shape)))
    n = sum(p.numel() for p in net.parameters())
    print("params: {:.1f}M".format(n / 1e6))
