"""Frozen DINOv3 visual encoding for the first Instruct-TAFS stage."""

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


DEFAULT_DINO_REPO = Path(__file__).resolve().parent / "dinov3"

DINO_VARIANTS = {
    "s": ("dinov3_vits16", [2, 5, 8, 11], [96, 192, 384, 768]),
    "s+": ("dinov3_vits16plus", [2, 5, 8, 11], [96, 192, 384, 768]),
    "b": ("dinov3_vitb16", [2, 5, 8, 11], [96, 192, 384, 768]),
    "l": ("dinov3_vitl16", [4, 11, 17, 23], [256, 512, 1024, 1024]),
    "h": ("dinov3_vith16plus", [7, 15, 23, 31], [320, 640, 1280, 1280]),
}


class DINOv3Encoder(nn.Module):
    """Expose four intermediate token grids without changing the pretrained encoder."""

    def __init__(self, size="h", repo_dir=None, checkpoint=None, pretrained=True):
        super().__init__()
        if size not in DINO_VARIANTS:
            raise ValueError(f"Unsupported DINOv3 size: {size}")
        repo = Path(repo_dir or DEFAULT_DINO_REPO).expanduser().resolve()
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 hubconf.py not found in {repo}")
        if pretrained and (checkpoint is None or not Path(checkpoint).expanduser().is_file()):
            raise FileNotFoundError("Provide an existing local DINOv3 checkpoint using --dino-ckpt.")
        hub_name, self.layer_indices, self.out_channels = DINO_VARIANTS[size]
        options = {"pretrained": pretrained}
        if pretrained:
            options["weights"] = str(Path(checkpoint).expanduser().resolve())
        # Import only the backbone factories, avoiding optional detection/segmentation heads.
        sys.path.insert(0, str(repo))
        try:
            factories = importlib.import_module("dinov3.hub.backbones")
        finally:
            sys.path.remove(str(repo))
        if not Path(factories.__file__).resolve().is_relative_to(repo):
            raise ImportError("A different dinov3 package is already loaded; restart with the requested local repository.")
        self.backbone = getattr(factories, hub_name)(**options).float()
        self.embed_dim = self.backbone.embed_dim
        if size == "h" and self.embed_dim != 1280:
            self.out_channels = [self.embed_dim // 4, self.embed_dim // 2, self.embed_dim, self.embed_dim]
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.patch_size = 16

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("DINOv3 images must have shape [batch, 3, height, width].")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("DINOv3 image dimensions must be divisible by 16.")
        return self.backbone.get_intermediate_layers(images.float(), n=self.layer_indices)
