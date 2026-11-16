#!/usr/bin/env python3
"""Joint language-generation and type-aware segmentation training for Instruct-TAFS."""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SEGMENTATION_TASKS, TAFSVLDataset, tafs_vl_collate_fn
from models import load_model
from models.visual_encoder import DEFAULT_DINO_REPO


logger = logging.getLogger(__name__)


def prepare_images(images, image_size=1024):
    """Resize RGB images and apply the ImageNet normalization used by DINOv3."""
    tensors = []
    for image in images:
        array = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)
    pixels = torch.stack(tensors)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def dice_loss_from_logits(logits, targets, smooth=1.0):
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    intersection = (probabilities * targets).sum(1)
    return (1 - (2 * intersection + smooth) / (probabilities.sum(1) + targets.sum(1) + smooth)).mean()


def segmentation_loss(logits, labels, task_types):
    height, width = logits.shape[-2:]
    masks = [
        cv2.resize(labels[task][index].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        for index, task in enumerate(task_types)
    ]
    targets = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets) + 0.25 * dice_loss_from_logits(logits, targets)


def text_generation_loss(model, batch, max_length=2048):
    losses = []
    for image, instruction, response in zip(batch["images"], batch["instructions"], batch["responses"]):
        user = {"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": instruction}
        ]}
        messages = [user, {"role": "assistant", "content": response}]
        text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = model.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        inputs = model.processor(
            text=[text], images=[image], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        prompt_inputs = model.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        prompt_length = int(prompt_inputs["attention_mask"][0].sum())
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100
        labels[inputs["attention_mask"] == 0] = -100
        if not (labels != -100).any():
            raise ValueError("The response was fully truncated; increase --max-sequence-length.")
        inputs = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        loss = model.backbone(**inputs, labels=labels.to(model.device), use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite text generation loss.")
        losses.append(loss)
    return torch.stack(losses).mean()


def train_epoch(model, train_loader, optimizer, image_size=1024, lambda_text=1.0, lambda_seg=1.0, max_sequence_length=2048):
    model.train()
    totals = {"loss": 0.0, "text": 0.0, "seg": 0.0}
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    progress = tqdm(train_loader, desc="Training")
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        text_loss = text_generation_loss(model, batch, max_sequence_length) if lambda_text else None
        seg_loss = None
        if lambda_seg:
            descriptions = model.generate_descriptions(batch["images"], batch["instructions"])
            pixels = prepare_images(batch["images"], image_size).to(model.device)
            logits = model(pixels, batch["instructions"], descriptions)
            seg_loss = segmentation_loss(logits, batch["labels"], batch["task_types"])
        loss = sum(
            weight * term for weight, term in ((lambda_text, text_loss), (lambda_seg, seg_loss))
            if term is not None
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite joint training loss.")
        if not loss.requires_grad:
            raise RuntimeError("No active loss reaches trainable parameters; check LoRA and loss weights.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        totals["loss"] += loss.item()
        totals["text"] += text_loss.item() if text_loss is not None else 0.0
        totals["seg"] += seg_loss.item() if seg_loss is not None else 0.0
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return {key: value / len(train_loader) for key, value in totals.items()}


def metrics_from_counts(counts):
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    divide = lambda value, total, empty=1.0: float(value / total) if total else empty
    foreground_iou = divide(tp, tp + fp + fn)
    background_iou = divide(tn, tn + fp + fn)
    foreground_dice = divide(2 * tp, 2 * tp + fp + fn)
    background_dice = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "foreground_iou": foreground_iou,
        "binary_miou": (foreground_iou + background_iou) / 2,
        "foreground_dice": foreground_dice,
        "binary_mdice": (foreground_dice + background_dice) / 2,
        "accuracy": divide(tp + tn, tp + fp + fn + tn),
        "recall": divide(tp, tp + fn),
    }


@torch.no_grad()
def evaluate(model, val_loader, save_dir, image_size=1024):
    model.eval()
    destination = Path(save_dir)
    for name in ("masks", "visualizations"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    results = []
    counts = {task: dict(tp=0, fp=0, fn=0, tn=0) for task in SEGMENTATION_TASKS}
    for batch in tqdm(val_loader, desc="Evaluating"):
        descriptions = model.generate_descriptions(batch["images"], batch["instructions"], max_new_tokens=256)
        pixels = prepare_images(batch["images"], image_size).to(model.device)
        predictions = model(pixels, batch["instructions"], descriptions).sigmoid()[:, 0].cpu().numpy() > 0.5
        for index, prediction in enumerate(predictions):
            task = batch["task_types"][index]
            target = cv2.resize(
                batch["labels"][task][index], (prediction.shape[1], prediction.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            sample_counts = {
                "tp": int((prediction & target).sum()), "fp": int((prediction & ~target).sum()),
                "fn": int((~prediction & target).sum()), "tn": int((~prediction & ~target).sum()),
            }
            for key, value in sample_counts.items():
                counts[task][key] += value
            iou = metrics_from_counts(sample_counts)["foreground_iou"]
            image_id = batch["image_ids"][index]
            name = f"{image_id}_{task}"
            cv2.imwrite(str(destination / "masks" / f"{name}_pred.png"), prediction.astype(np.uint8) * 255)
            cv2.imwrite(str(destination / "masks" / f"{name}_gt.png"), target.astype(np.uint8) * 255)
            image = np.array(batch["images"][index].convert("RGB"))
            display_h = max(1, round(512 * image.shape[0] / image.shape[1]))
            base = cv2.cvtColor(cv2.resize(image, (512, display_h)), cv2.COLOR_RGB2BGR)
            target_view = cv2.resize(target.astype(np.uint8), (512, display_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            prediction_view = cv2.resize(prediction.astype(np.uint8), (512, display_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            ground_truth, predicted = base.copy(), base.copy()
            ground_truth[target_view] = (0, 255, 0)
            predicted[prediction_view] = (0, 0, 255)
            visualization = np.hstack((base, ground_truth, predicted))
            cv2.putText(visualization, f"{task} IoU={iou:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(destination / "visualizations" / f"{name}.png"), visualization)
            results.append({
                "image_id": image_id, "task_type": task, "instruction": batch["instructions"][index],
                "generated_text": descriptions[index], "ground_truth": batch["responses"][index], "iou": iou,
            })
    if not results:
        raise ValueError("Evaluation produced no samples.")
    per_task = {
        task: metrics_from_counts(value) for task, value in counts.items() if sum(value.values()) > 0
    }
    summary = {
        "num_samples": len(results),
        "mean_iou": float(np.mean([item["iou"] for item in results])),
        "mean_iou_definition": "Mean per-sample foreground IoU over the requested tasks.",
        "per_task": per_task,
        "generated": [item["generated_text"] for item in results[:3]],
        "ground_truth": [item["ground_truth"] for item in results[:3]],
    }
    (destination / "eval_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def save_checkpoint(model, destination, image_size=1024):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model.pt")
    config = {**model.model_config, "image_size": image_size}
    (destination / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.processor.save_pretrained(destination / "processor")
    model.save_lora(str(destination / "lora_adapter"))


def build_parser():
    parser = argparse.ArgumentParser(description="Train Instruct-TAFS on TAFS-VL.")
    parser.add_argument("--data-dir", required=True, help="Training split containing img/ and json/.")
    parser.add_argument("--val-data-dir", help="Optional validation split.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dino-size", choices=["s", "s+", "b", "l", "h"], default="h")
    parser.add_argument("--dino-repo-dir", default=str(DEFAULT_DINO_REPO))
    parser.add_argument("--dino-ckpt", required=True, help="Local pretrained DINOv3 checkpoint.")
    parser.add_argument("--freeze-dino", action="store_true", default=True, help="The paper pipeline always freezes DINOv3.")
    parser.add_argument("--sisd-features", "--dpt-features", dest="sisd_features", type=int, default=128)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-text", type=float, default=1.0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--description-key", default="img_description_eg")
    parser.add_argument("--train-task", choices=["auto", "all", *SEGMENTATION_TASKS], default="auto")
    parser.add_argument("--eval-task", choices=["all", *SEGMENTATION_TASKS], default="farmland")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./checkpoints/instruct_tafs")
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.image_size, args.eval_interval, args.save_interval) <= 0:
        parser.error("Epochs, batch size, image size, and intervals must be positive.")
    if args.num_workers < 0 or args.image_size % 16:
        parser.error("Workers must be nonnegative and image size must be divisible by 16.")
    if args.lambda_text < 0 or args.lambda_seg < 0 or args.lambda_text + args.lambda_seg == 0:
        parser.error("Loss weights must be nonnegative with at least one positive weight.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_dataset = TAFSVLDataset(args.data_dir, "train", args.train_task, args.description_key)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=tafs_vl_collate_fn, pin_memory=torch.device(args.device).type == "cuda",
    )
    val_loader = None
    if args.val_data_dir:
        val_dataset = TAFSVLDataset(args.val_data_dir, "val", args.eval_task, args.description_key)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers, collate_fn=tafs_vl_collate_fn)
    config = {
        "model_name": args.base_model, "dino_size": args.dino_size,
        "dino_repo_dir": args.dino_repo_dir, "dino_ckpt": args.dino_ckpt,
        "text_dim": args.text_dim, "sisd_features": args.sisd_features,
        "use_lora": args.use_lora, "lora_r": args.lora_r,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    model, _ = load_model(config, args.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    logger.info("Trainable parameters: %.2fM", sum(parameter.numel() for parameter in parameters) / 1e6)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, args.image_size, args.lambda_text, args.lambda_seg, args.max_sequence_length)
        logger.info("Epoch %s/%s: %s", epoch, args.epochs, losses)
        checkpoint = output / f"checkpoint_epoch_{epoch}"
        if epoch % args.save_interval == 0:
            save_checkpoint(model, checkpoint, args.image_size)
        if val_loader is not None and epoch % args.eval_interval == 0:
            metrics = evaluate(model, val_loader, checkpoint / "eval_results", args.image_size)
            logger.info("Validation mean foreground IoU: %.4f", metrics["mean_iou"])
            if metrics["mean_iou"] > best_iou:
                best_iou = metrics["mean_iou"]
                save_checkpoint(model, output / "best_model", args.image_size)
    save_checkpoint(model, output / "final_model", args.image_size)
    if val_loader is not None:
        evaluate(model, val_loader, output / "final_eval_results", args.image_size)
    logger.info("Training completed: %s", output.resolve())


if __name__ == "__main__":
    main()
