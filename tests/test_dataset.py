"""Unit tests for YAML configuration loading and MVTec AD dataset parsing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from anomaly_detection.data import (
    MVTecTestDataset,
    MVTecTrainDataset,
    validate_mvtec_category,
)
from anomaly_detection.utils import load_yaml_config


def _create_dummy_png(path: Path, size: tuple[int, int] = (64, 64), mode: str = "RGB", val: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        arr = np.full((size[1], size[0], 3), val, dtype=np.uint8)
    else:
        arr = np.zeros((size[1], size[0]), dtype=np.uint8)
        arr[16:48, 16:48] = 255
    Image.fromarray(arr, mode=mode).save(path)


class TestConfigAndDataset(unittest.TestCase):
    def test_yaml_configs_load_cleanly(self) -> None:
        config_paths = [
            ROOT / "configs" / "mvtec_ad.yaml",
        ]
        for path in config_paths:
            cfg = load_yaml_config(path)
            self.assertIn("dataset", cfg)
            self.assertIn("evaluation", cfg)

    def test_mvtec_train_and_test_dataset_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cat = "carpet"
            # Create normal train images (including non-3-char stem to test fix for [:3] bug)
            _create_dummy_png(root / cat / "train" / "good" / "000.png")
            _create_dummy_png(root / cat / "train" / "good" / "sample_1001.png")

            # Create normal and anomalous test images + ground-truth mask
            _create_dummy_png(root / cat / "test" / "good" / "000.png")
            _create_dummy_png(root / cat / "test" / "color" / "defect_0042.png")
            _create_dummy_png(
                root / cat / "ground_truth" / "color" / "defect_0042_mask.png",
                mode="L",
            )

            summary = validate_mvtec_category(root_dir=root, obj_name=cat, resize_shape=128)
            self.assertEqual(summary["num_train_good"], 2)
            self.assertEqual(summary["num_test_total"], 2)
            self.assertEqual(summary["num_test_good"], 1)
            self.assertEqual(summary["num_test_anomalous"], 1)
            self.assertEqual(summary["train_image_shape"], (3, 128, 128))
            self.assertEqual(summary["test_image_shape"], (3, 128, 128))
            self.assertEqual(summary["test_mask_shape"], (1, 128, 128))
            self.assertTrue(summary["zero_train_test_overlap"])

            test_ds = MVTecTestDataset(root_dir=root, obj_name=cat, resize_shape=128)
            anom_item = [ test_ds[i] for i in range(len(test_ds)) if test_ds[i]["defect_type"] == "color" ][0]
            self.assertEqual(int(anom_item["label"][0]), 1)
            unique_vals = set(anom_item["gt_mask"].unique().tolist())
            self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_unsupervised_protocol_rejects_anomalies_in_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cat = "bottle"
            _create_dummy_png(root / cat / "train" / "good" / "000.png")
            _create_dummy_png(root / cat / "train" / "broken_large" / "001.png")

            with self.assertRaises(ValueError):
                _ = MVTecTrainDataset(root_dir=root, obj_name=cat, resize_shape=128)

    def test_train_val_split_and_keyword_tolerance(self) -> None:
        """Verify 80/20 train/val split is disjoint and keyword aliases are accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cat = "carpet"
            # Create 10 normal train images
            for i in range(10):
                _create_dummy_png(root / cat / "train" / "good" / f"{i:03d}.png")

            # Create 2 test images
            _create_dummy_png(root / cat / "test" / "good" / "000.png")
            _create_dummy_png(root / cat / "test" / "hole" / "001.png")
            _create_dummy_png(root / cat / "ground_truth" / "hole" / "001_mask.png", mode="L")

            # Test alternative keywords (data_root, category, img_size)
            train_ds = MVTecTrainDataset(data_root=root, category=cat, img_size=128, split="train", val_ratio=0.2, seed=42)
            val_ds = MVTecTrainDataset(data_root=root, category=cat, img_size=128, split="val", val_ratio=0.2, seed=42)
            test_ds = MVTecTestDataset(data_root=root, category=cat, img_size=128)

            self.assertEqual(len(train_ds), 8)
            self.assertEqual(len(val_ds), 2)
            self.assertEqual(len(test_ds), 2)

            # Check zero overlap between train and val
            train_paths = {p.resolve() for p in train_ds.image_paths}
            val_paths = {p.resolve() for p in val_ds.image_paths}
            test_paths = {p.resolve() for p in test_ds.image_paths}

            self.assertEqual(len(train_paths.intersection(val_paths)), 0)
            self.assertEqual(len(train_paths.intersection(test_paths)), 0)
            self.assertEqual(len(val_paths.intersection(test_paths)), 0)

            # Check compatibility keys in test dataset
            sample = test_ds[1]
            self.assertIn("mask", sample)
            self.assertIn("gt_mask", sample)
            self.assertIn("img_path", sample)
            self.assertIn("path", sample)

    def test_val_ratio_zero_and_single_image_edge_cases(self) -> None:
        """Verify val_ratio=0.0 produces an empty val split (zero leakage) and single-image train does not empty train."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cat = "carpet"
            for i in range(5):
                _create_dummy_png(root / cat / "train" / "good" / f"{i:03d}.png")

            # Case 1: val_ratio=0.0 -> train gets all 5, val gets 0
            train_ds_0 = MVTecTrainDataset(root_dir=root, obj_name=cat, split="train", val_ratio=0.0, seed=42)
            val_ds_0 = MVTecTrainDataset(root_dir=root, obj_name=cat, split="val", val_ratio=0.0, seed=42)
            self.assertEqual(len(train_ds_0), 5)
            self.assertEqual(len(val_ds_0), 0)

            # Case 2: Only 1 image in train/good with val_ratio=0.2 -> train keeps 1, val gets 0
            cat_single = "bottle"
            _create_dummy_png(root / cat_single / "train" / "good" / "000.png")
            train_single = MVTecTrainDataset(root_dir=root, obj_name=cat_single, split="train", val_ratio=0.2, seed=42)
            val_single = MVTecTrainDataset(root_dir=root, obj_name=cat_single, split="val", val_ratio=0.2, seed=42)
            self.assertEqual(len(train_single), 1)
            self.assertEqual(len(val_single), 0)


if __name__ == "__main__":
    unittest.main()
