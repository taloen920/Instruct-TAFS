#!/usr/bin/env python3
"""Evaluate an Instruct-TAFS checkpoint using fixed TAFS-VL target instructions."""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets import SEGMENTATION_TASKS, TAFSVLDataset, tafs_vl_collate_fn
from models import load_model
from models.checkpoint import load_checkpoint
from train import evaluate


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Instruct-TAFS on TAFS-VL.")
    parser.add_argument("--checkpoint", required=True, help="A trained model.pt or legacy .torch file.")
    parser.add_argument("--val-data-dir", required=True)
    parser.add_argument("--base-model", help="Override the Qwen3-VL model location saved in model_config.json.")
    parser.add_argument("--dino-size", choices=["s", "s+", "b", "l", "h"])
    parser.add_argument("--dino-repo-dir")
    parser.add_argument("--dino-ckpt", help="Override the saved pretrained DINOv3 checkpoint path.")
    parser.add_argument("--sisd-features", "--dpt-features", dest="sisd_features", type=int)
    parser.add_argument("--text-dim", type=int)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-dir", help="Optional LoRA adapter override.")
    parser.add_argument("--freeze-dino", action="store_true", default=True)
    parser.add_argument("--eval-task", choices=["all", *SEGMENTATION_TASKS], default="farmland")
    parser.add_argument("--description-key", default="img_description_eg")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, help="Maximum number of image-task pairs.")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="./eval_results")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    config_path = checkpoint.parent / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    for name in ("dino_size", "dino_repo_dir", "dino_ckpt", "sisd_features", "text_dim", "use_lora", "lora_r"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.base_model is not None:
        config["model_name"] = args.base_model
    image_size = args.image_size if args.image_size is not None else config.get("image_size", 1024)
    if args.batch_size <= 0 or args.num_workers < 0 or image_size <= 0 or image_size % 16:
        parser.error("Use a positive batch size, nonnegative workers, and an image size divisible by 16.")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be positive.")
    dataset = TAFSVLDataset(args.val_data_dir, "val", args.eval_task, args.description_key)
    if args.num_samples is not None:
        dataset = Subset(dataset, range(min(len(dataset), args.num_samples)))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=tafs_vl_collate_fn)
    model, _ = load_model(config, args.device)
    load_checkpoint(model, checkpoint, device=args.device)
    if args.lora_dir:
        model.load_lora(args.lora_dir)
    metrics = evaluate(model, loader, Path(args.output_dir) / "eval_results", image_size)
    logging.info("Evaluation results: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
