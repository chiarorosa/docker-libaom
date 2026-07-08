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
import torch.nn.functional as F

from partition_defs import MODEL_LEVELS, NUM_PARTITION_TYPES, legality_mask

# torchvision ConvNeXt.features stage output indices we tap:
#   stage0 (index 1) -> 16x16, LOCAL features (small receptive field) -- crucial
#     so each 16px cell can discriminate its own block instead of the near-global
#     context the deep 4x4 map carries;
#   stage2 (index 5) -> 4x4  and stage3 (index 7) -> 2x2 -> deep global context.
_CAPTURE = {1: "f16", 5: "f4", 7: "f2"}
_NEG = -1e4  # additive mask for illegal classes


def _backbone(variant, drop_path=0.0, pretrained=False):
    from torchvision.models import (
        convnext_tiny, convnext_small, convnext_base,
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights)
    ctor = {"tiny": convnext_tiny, "small": convnext_small,
            "base": convnext_base}[variant]
    wts = None
    if pretrained:
        wts = {"tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
               "small": ConvNeXt_Small_Weights.IMAGENET1K_V1,
               "base": ConvNeXt_Base_Weights.IMAGENET1K_V1}[variant]
    net = ctor(weights=wts, stochastic_depth_prob=drop_path)
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
    def __init__(self, variant="tiny", fusion_dim=128, in_channels=2,
                 pretrained=False):
        super().__init__()
        self.features, chans = _backbone(variant, pretrained=pretrained)
        # Adapt the stem to `in_channels` (luma + qindex). With a pretrained
        # backbone, seed the luma channel from the averaged RGB stem filters so
        # the pretrained low-level features are preserved; qindex starts at 0.
        stem_conv = self.features[0][0]
        new_conv = nn.Conv2d(in_channels, stem_conv.out_channels,
                             kernel_size=stem_conv.kernel_size,
                             stride=stem_conv.stride,
                             padding=stem_conv.padding)
        if pretrained:
            with torch.no_grad():
                avg = stem_conv.weight.mean(dim=1, keepdim=True)  # (out,1,4,4)
                new_conv.weight.zero_()
                new_conv.weight[:, 0:1].copy_(avg)
                if stem_conv.bias is not None:
                    new_conv.bias.copy_(stem_conv.bias)
        self.features[0][0] = new_conv

        d = fusion_dim
        self.lat_f2 = nn.Conv2d(chans["f2"], d, 1)   # 2x2  (deep global)
        self.lat_f4 = nn.Conv2d(chans["f4"], d, 1)   # 4x4
        self.lat_f16 = nn.Conv2d(chans["f16"], d, 1)  # 16x16 (shallow local)
        self.act = nn.GELU()

        # A small conv on the fused high-res map lets each cell mix its local
        # features before the per-level pooling.
        self.smooth = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.GELU())

        # Heads act on per-level pooled maps (each cell aggregates exactly its
        # block's region of the 16x16 fused map: 4x4 pool -> 16px, 8x8 -> 32px).
        self.head_64 = nn.Linear(d, NUM_PARTITION_TYPES)
        self.head_32 = nn.Conv2d(d, NUM_PARTITION_TYPES, 1)
        self.head_16 = nn.Conv2d(d, NUM_PARTITION_TYPES, 1)

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
        return feats["f16"], feats["f4"], feats["f2"]

    def forward(self, x):
        f16, f4, f2 = self._multiscale(x)          # 16x16, 4x4, 2x2
        deep = self.lat_f2(f2)                      # 2x2  global context
        mid = self.lat_f4(f4) + F.interpolate(deep, scale_factor=2,
                                              mode="nearest")            # 4x4
        hi = self.lat_f16(f16) + F.interpolate(mid, scale_factor=4,
                                              mode="nearest")            # 16x16
        hi = self.smooth(self.act(hi))             # local + global, 16x16

        m16 = F.avg_pool2d(hi, 4)                   # 4x4  (one cell per 16px blk)
        m32 = F.avg_pool2d(hi, 8)                   # 2x2  (one cell per 32px blk)
        g64 = hi.mean(dim=(2, 3))                   # (B, d) whole superblock

        def grid(logits):  # (B,C,H,W) -> (B,H,W,C)
            return logits.permute(0, 2, 3, 1).contiguous()

        out = {
            64: self.head_64(g64).view(-1, 1, 1, NUM_PARTITION_TYPES),
            32: grid(self.head_32(m32)),
            16: grid(self.head_16(m16)),
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
