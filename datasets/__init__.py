"""TAFS-VL datasets for Instruct-TAFS."""

from .tafs_vl import SEGMENTATION_TASKS, TAFSVLDataset, tafs_vl_collate_fn

__all__ = ["TAFSVLDataset", "tafs_vl_collate_fn", "SEGMENTATION_TASKS"]
