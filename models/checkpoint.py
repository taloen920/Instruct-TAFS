"""Load current checkpoints and migrate names from the original implementation."""

import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def migrate_legacy_state_dict(state_dict):
    """Map old adapter/decoder names to the paper-aligned module hierarchy."""
    replacements = (
        ("dino_decoder.head.dual_fusion.", "dptef."),
        ("dino_decoder.encoder.", "visual_encoder."),
        ("dino_decoder.head.", "sisd."),
        ("txt_norm.", "dptef.text_norm."),
        ("to_sparse.", "dptef.text_projection."),
    )
    migrated, ignored = {}, []
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("to_dense.") or key in {"temp", "scale"}:
            ignored.append(key)
            continue
        for old, new in replacements:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        if key in migrated:
            raise ValueError(f"Duplicate migrated checkpoint key: {key}")
        migrated[key] = value
    return migrated, ignored


def load_checkpoint(model, checkpoint, device="cpu"):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    legacy = any(key.removeprefix("module.").startswith("dino_decoder.") for key in state)
    if legacy:
        state, ignored = migrate_legacy_state_dict(state)
        logger.warning(
            "Migrating legacy parameter names; the corrected attention and gradient routing "
            "change training behavior. Retrain to use the paper-aligned implementation. "
            "Discarded unused legacy parameters: %s", ignored,
        )
    elif state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    return model.load_state_dict(state, strict=True)
