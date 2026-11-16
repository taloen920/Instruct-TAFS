"""The DINOv3-based Instruct-TAFS architecture and its paper modules."""

from .dptef import DPTEF
from .instruct_tafs import InstructTAFS
from .sisd import SISD
from .visual_encoder import DINOv3Encoder


def load_model(config: dict, device: str = "cuda"):
    """Construct Instruct-TAFS using current or legacy configuration field names."""
    import torch

    model = InstructTAFS(
        model_name=config.get("model_name", "Qwen/Qwen3-VL-4B-Instruct"),
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 16),
        dino_size=config.get("dino_size", "h"),
        dino_repo_dir=config.get("dino_repo_dir"),
        dino_ckpt=config.get("dino_ckpt"),
        text_dim=config.get("text_dim", config.get("transformer_dim", 256)),
        sisd_features=config.get("sisd_features", config.get("dpt_features", 128)),
        lora_alpha=config.get("lora_alpha"),
        lora_dropout=config.get("lora_dropout", 0.05),
        max_txt_len=config.get("max_txt_len", 256),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        dtype=torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32,
        device=device,
    )
    return model, model.processor


__all__ = ["InstructTAFS", "DPTEF", "SISD", "DINOv3Encoder", "load_model"]
