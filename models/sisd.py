"""Semantic-Injected Segmentation Decoder (SISD), paper Equations (6)-(10)."""

from math import gcd

import torch
from torch import nn
from torch.nn import functional as F


class ProgressiveUpsampling(nn.Module):
    """Recover spatial detail with two transposed-convolution stages and refinement."""

    def __init__(self, in_channels, nclass=1):
        super().__init__()
        if in_channels < 8 or in_channels % 8:
            raise ValueError("Upsampling channels must be a positive multiple of eight.")
        self.up1 = self._stage(in_channels, in_channels // 2, 32)
        self.up2 = self._stage(in_channels // 2, in_channels // 4, 16)
        channels = in_channels // 8
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels // 4, channels, 3, padding=1),
            nn.GroupNorm(gcd(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(gcd(8, channels), channels),
            nn.GELU(),
        )
        self.final_conv = nn.Conv2d(channels, nclass, 1)

    @staticmethod
    def _stage(in_channels, out_channels, groups):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.GroupNorm(gcd(groups, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(gcd(groups, out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, features):
        return self.final_conv(self.refine(self.up2(self.up1(features))))


class SISD(nn.Module):
    """Aggregate four visual levels and inject the unified DPTEF representation.

    The decoder returns logits. Sigmoid is applied only for inference so that
    training can use numerically stable binary cross entropy with logits.
    """

    def __init__(self, in_channels, features=128, out_channels=None, text_dim=256):
        super().__init__()
        if features < 2 or features % 2:
            raise ValueError("SISD features must be a positive even number.")
        if out_channels is None:
            out_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        if len(out_channels) != 4 or min(out_channels) < 1:
            raise ValueError("SISD requires four positive visual projection widths.")
        self.in_channels = in_channels
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, width, 1) for width in out_channels])
        self.scratch = nn.Module()
        for index, width in enumerate(out_channels, 1):
            self.scratch.add_module(
                f"layer{index}_rn", nn.Conv2d(width, features, 3, padding=1, bias=False)
            )
        self.text_modulation = nn.Sequential(
            nn.Linear(text_dim, features * 4),
            nn.ReLU(inplace=True),
            nn.Linear(features * 4, features * 4),
            nn.Sigmoid(),
        )
        self.progressive_upsampling = ProgressiveUpsampling(features * 4)

    def forward(self, visual_features, patch_grid, fused_text, output_size):
        if len(visual_features) != 4:
            raise ValueError("SISD requires exactly four DINOv3 feature levels.")
        patch_h, patch_w = patch_grid
        aligned = []
        for index, tokens in enumerate(visual_features):
            if tokens.ndim != 3 or tokens.shape[1:] != (patch_h * patch_w, self.in_channels):
                raise ValueError(
                    f"Unexpected visual feature shape {tuple(tokens.shape)}; "
                    f"expected [batch, {patch_h * patch_w}, {self.in_channels}]."
                )
            spatial = tokens.transpose(1, 2).reshape(tokens.shape[0], self.in_channels, patch_h, patch_w)
            projected = self.projects[index](spatial)
            mapped = getattr(self.scratch, f"layer{index + 1}_rn")(projected)
            aligned.append(F.interpolate(mapped, size=(patch_h, patch_w), mode="bilinear", align_corners=True))
        visual = torch.cat(aligned, dim=1)
        channel_weights = self.text_modulation(fused_text.to(visual.dtype)).unsqueeze(-1).unsqueeze(-1)
        logits = self.progressive_upsampling(visual * channel_weights)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=True)
