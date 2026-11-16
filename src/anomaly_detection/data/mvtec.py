"""Cross-platform, leak-free MVTec AD dataset implementation.

Fixes three critical bugs identified in the official CDA repository (`dataloder.py`):
1. Fixes the module filename typo (`dataloder.py` vs `data_loader`).
2. Replaces hard-coded `self.gt_root = "./datasets/mvtec/..."` with `Path(root_dir) / obj_name / "ground_truth"`.
3. Replaces OS-fragile `img_path.split("/")` and `[:3]` stem slicing with `pathlib.Path` operations.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MVTEC_CATEGORIES: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_image_transform(
    resize_shape: int = 256,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Builds the standard STFPM/CDA image preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_mask_transform(resize_shape: int = 256) -> transforms.Compose:
    """Builds the ground-truth anomaly mask preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((resize_shape, resize_shape)),
            transforms.ToTensor(),
        ]
    )


class MVTecTrainDataset(Dataset):
    """Unsupervised training dataset loading only defect-free (`train/good`) images.

    Supports paper-defined 80/20 train/validation split on normal images for model selection.
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        obj_name: Optional[str] = None,
        resize_shape: int = 256,
        transform: Optional[Callable] = None,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        *,
        data_root: Optional[str | Path] = None,
        category: Optional[str] = None,
        img_size: Optional[int] = None,
    ) -> None:
        raw_root = root_dir if root_dir is not None else data_root
        raw_obj = obj_name if obj_name is not None else category
        raw_shape = resize_shape if img_size is None else img_size

        if raw_root is None:
            raise ValueError("root_dir (or data_root) must be provided.")
        if raw_obj is None:
            raise ValueError("obj_name (or category) must be provided.")

        self.root_dir = Path(raw_root)
        self.obj_name = str(raw_obj)
        self.resize_shape = int(raw_shape)
        self.split = str(split).lower()
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        train_dir = self.root_dir / self.obj_name / "train"
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {train_dir}. "
                f"Check root_dir (currently '{self.root_dir}')."
            )

        # In unsupervised one-class protocol, only 'good' images are allowed in train
        all_pngs = sorted(train_dir.glob("*/*.png"))
        non_good = [p for p in all_pngs if p.parent.name != "good"]
        if non_good:
            raise ValueError(
                f"Unsupervised protocol violation: found {len(non_good)} non-'good' "
                f"training images in {train_dir} (e.g., {non_good[0]})."
            )

        if not all_pngs:
            raise FileNotFoundError(f"No training .png images found in {train_dir}")

        # Deterministic 80/20 train/validation split on normal training images
        if self.split == "all":
            self.image_paths: List[Path] = all_pngs
        elif self.split not in ("train", "val"):
            raise ValueError(f"Invalid split '{self.split}'. Expected 'train', 'val', or 'all'.")
        elif self.val_ratio <= 0.0:
            if self.split == "train":
                self.image_paths = all_pngs
            else:  # "val" must be empty when val_ratio == 0.0 to prevent train->val leakage
                self.image_paths = []
        elif len(all_pngs) < 2:
            # Too few images to split: keep image in train and leave val empty
            if self.split == "train":
                self.image_paths = all_pngs
            else:
                self.image_paths = []
        else:
            rng = random.Random(self.seed)
            indices = list(range(len(all_pngs)))
            rng.shuffle(indices)
            val_count = min(len(all_pngs) - 1, max(1, int(round(len(all_pngs) * self.val_ratio))))
            val_indices = set(indices[:val_count])

            if self.split == "train":
                self.image_paths = [p for i, p in enumerate(all_pngs) if i not in val_indices]
            else:  # "val"
                self.image_paths = [p for i, p in enumerate(all_pngs) if i in val_indices]

        self.transform = transform or build_image_transform(self.resize_shape)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
        tensor_img = self.transform(img)
        return {
            "image": tensor_img,
            "path": str(img_path),
            "img_path": str(img_path),
        }


class MVTecTestDataset(Dataset):
    """Evaluation dataset loading normal (`good`) and anomalous test images + masks."""

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        obj_name: Optional[str] = None,
        resize_shape: int = 256,
        transform: Optional[Callable] = None,
        gt_transform: Optional[Callable] = None,
        *,
        data_root: Optional[str | Path] = None,
        category: Optional[str] = None,
        img_size: Optional[int] = None,
    ) -> None:
        raw_root = root_dir if root_dir is not None else data_root
        raw_obj = obj_name if obj_name is not None else category
        raw_shape = resize_shape if img_size is None else img_size

        if raw_root is None:
            raise ValueError("root_dir (or data_root) must be provided.")
        if raw_obj is None:
            raise ValueError("obj_name (or category) must be provided.")

        self.root_dir = Path(raw_root)
        self.obj_name = str(raw_obj)
        self.resize_shape = int(raw_shape)

        self.test_dir = self.root_dir / self.obj_name / "test"
        self.gt_root = self.root_dir / self.obj_name / "ground_truth"
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

        self.image_paths: List[Path] = sorted(self.test_dir.glob("*/*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No test .png images found in {self.test_dir}")

        self.transform = transform or build_image_transform(self.resize_shape)
        self.gt_transform = gt_transform or build_mask_transform(self.resize_shape)

    def _resolve_mask_path(self, img_path: Path, defect_type: str) -> Path:
        """Resolves `<gt_root>/<defect_type>/<stem>_mask.png` portably."""
        mask_candidate = self.gt_root / defect_type / f"{img_path.stem}_mask.png"
        if mask_candidate.exists():
            return mask_candidate
        # Fallback if mask shares exact filename
        fallback = self.gt_root / defect_type / img_path.name
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Ground-truth mask not found for anomalous image '{img_path}'. "
            f"Expected '{mask_candidate}'."
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        defect_type = img_path.parent.name

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
        tensor_img = self.transform(img)

        if defect_type == "good":
            gt_label = np.array([0], dtype=np.float32)
            gt_mask = torch.zeros((1, self.resize_shape, self.resize_shape), dtype=torch.float32)
        else:
            gt_label = np.array([1], dtype=np.float32)
            gt_path = self._resolve_mask_path(img_path, defect_type)
            with Image.open(gt_path) as raw_mask:
                mask_l = raw_mask.convert("L")
            gt_mask = self.gt_transform(mask_l)
            # Binarize mask at 0.5 matching official protocol
            gt_mask = (gt_mask > 0.5).float()

        return {
            "image": tensor_img,
            "label": gt_label,
            "gt_mask": gt_mask,
            "mask": gt_mask,
            "defect_type": defect_type,
            "path": str(img_path),
            "img_path": str(img_path),
        }


def validate_mvtec_category(
    root_dir: str | Path,
    obj_name: str,
    resize_shape: int = 256,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Validates directory structure, tensor shapes, and train/val/test disjointness."""
    all_train_ds = MVTecTrainDataset(
        root_dir=root_dir, obj_name=obj_name, resize_shape=resize_shape, split="all"
    )
    test_ds = MVTecTestDataset(root_dir=root_dir, obj_name=obj_name, resize_shape=resize_shape)

    train_set = {Path(p).resolve() for p in all_train_ds.image_paths}
    test_set = {Path(p).resolve() for p in test_ds.image_paths}

    train_test_overlap = train_set.intersection(test_set)
    if train_test_overlap:
        raise RuntimeError(f"Data leakage detected! {len(train_test_overlap)} files appear in both train and test.")

    zero_train_val_overlap = True
    num_val_good = 0
    if val_ratio > 0.0 and len(all_train_ds) >= 2:
        split_train = MVTecTrainDataset(
            root_dir=root_dir, obj_name=obj_name, resize_shape=resize_shape, split="train", val_ratio=val_ratio, seed=seed
        )
        split_val = MVTecTrainDataset(
            root_dir=root_dir, obj_name=obj_name, resize_shape=resize_shape, split="val", val_ratio=val_ratio, seed=seed
        )
        s_train = {Path(p).resolve() for p in split_train.image_paths}
        s_val = {Path(p).resolve() for p in split_val.image_paths}
        train_val_overlap = s_train.intersection(s_val)
        if train_val_overlap:
            raise RuntimeError(f"Data leakage detected! {len(train_val_overlap)} files appear in both train and val.")
        num_val_good = len(split_val)
        zero_train_val_overlap = len(train_val_overlap) == 0

    sample_train = all_train_ds[0]
    sample_test = test_ds[0]

    n_good_test = sum(1 for p in test_ds.image_paths if p.parent.name == "good")
    n_anom_test = len(test_ds) - n_good_test
    defect_types = sorted({p.parent.name for p in test_ds.image_paths if p.parent.name != "good"})

    return {
        "category": obj_name,
        "num_train_good": len(all_train_ds),
        "num_val_good": num_val_good,
        "num_test_total": len(test_ds),
        "num_test_good": n_good_test,
        "num_test_anomalous": n_anom_test,
        "defect_types": defect_types,
        "train_image_shape": tuple(sample_train["image"].shape),
        "test_image_shape": tuple(sample_test["image"].shape),
        "test_mask_shape": tuple(sample_test["gt_mask"].shape),
        "zero_train_test_overlap": len(train_test_overlap) == 0,
        "zero_train_val_overlap": zero_train_val_overlap,
    }
