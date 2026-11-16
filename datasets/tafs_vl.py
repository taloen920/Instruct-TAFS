"""TAFS-VL image, polygon, and attribute-description loading."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SEGMENTATION_TASKS = {
    "farmland": "Segment all farmland areas in the image.",
    "dry_field": "Segment dry field areas in the image.",
    "paddy_field": "Segment paddy field areas in the image.",
}
LABEL_TO_TASK = {"farmland": "farmland", "Dry field": "dry_field", "Paddy field": "paddy_field"}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class TAFSVLDataset(Dataset):
    """Rasterize Labelme polygons and select the instruction-conditioned target.

    Training samples an available target per image. Evaluation uses a fixed
    task, or all three tasks, so repeated validation is deterministic.
    """

    def __init__(self, data_dir, mode="train", task_type=None, description_key="img_description_eg"):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.task_type = task_type or ("auto" if mode == "train" else "farmland")
        if self.task_type not in {*SEGMENTATION_TASKS, "auto", "all"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        if mode != "train" and self.task_type == "auto":
            raise ValueError("Use a fixed task or 'all' for deterministic evaluation.")
        self.description_key = description_key
        self.json_files = sorted((self.data_dir / "json").glob("*.json"))
        if not self.json_files:
            raise FileNotFoundError(f"No JSON annotations found in {self.data_dir / 'json'}")
        self.image_paths = {}
        for path in (self.data_dir / "img").glob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                if path.stem in self.image_paths:
                    raise ValueError(f"Multiple images share annotation stem: {path.stem}")
                self.image_paths[path.stem] = path
        missing = [path.stem for path in self.json_files if path.stem not in self.image_paths]
        if missing:
            raise FileNotFoundError(f"Missing matching images in {self.data_dir / 'img'}: {missing[:5]}")

    def __len__(self):
        return len(self.json_files) * (3 if self.task_type == "all" else 1)

    @staticmethod
    def rasterize(shapes, height, width):
        masks = {task: np.zeros((height, width), dtype=np.uint8) for task in SEGMENTATION_TASKS}
        for shape in shapes:
            label = shape.get("label", "")
            if label not in LABEL_TO_TASK:
                raise ValueError(f"Unknown polygon label {label!r}; expected one of {list(LABEL_TO_TASK)}")
            points = np.asarray(shape.get("points", []), dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Every polygon must contain at least three finite [x, y] points.")
            polygon = points.astype(np.int32)
            task = LABEL_TO_TASK[label]
            cv2.fillPoly(masks[task], [polygon], 1)
            cv2.fillPoly(masks["farmland"], [polygon], 1)
        return masks

    def __getitem__(self, index):
        json_index = index // 3 if self.task_type == "all" else index
        path = self.json_files[json_index]
        with path.open(encoding="utf-8-sig") as stream:
            annotation = json.load(stream)
        with Image.open(self.image_paths[path.stem]) as source:
            image = source.convert("RGB")
        description = annotation.get(self.description_key, "")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Missing nonempty {self.description_key!r} description in {path}")
        width, height = image.size
        masks = self.rasterize(annotation.get("shapes", []), height, width)
        if self.task_type == "all":
            task = list(SEGMENTATION_TASKS)[index % 3]
        elif self.task_type == "auto":
            available = [task for task, mask in masks.items() if mask.any()]
            task = str(np.random.choice(available or ["farmland"]))
        else:
            task = self.task_type
        return {
            "image": image, "labels": masks,
            "instruction": SEGMENTATION_TASKS[task], "response": description.strip(),
            "image_id": path.stem, "image_size": (height, width), "task_type": task,
        }


def tafs_vl_collate_fn(batch):
    return {
        "images": [item["image"] for item in batch],
        "labels": {task: [item["labels"][task] for item in batch] for task in SEGMENTATION_TASKS},
        "instructions": [item["instruction"] for item in batch],
        "responses": [item["response"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_sizes": [item["image_size"] for item in batch],
        "task_types": [item["task_type"] for item in batch],
    }
