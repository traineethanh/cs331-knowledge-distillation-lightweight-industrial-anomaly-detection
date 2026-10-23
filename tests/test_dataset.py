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
            ROOT / "configs" / "baseline" / "cda_mvtec.yaml",
            ROOT / "configs" / "research_a" / "default.yaml",
            ROOT / "configs" / "research_b" / "default.yaml",
            ROOT / "configs" / "experiments" / "smoke_test.yaml",
        ]
        for path in config_paths:
            cfg = load_yaml_config(path)
            self.assertIn("experiment", cfg)
            self.assertIn("dataset", cfg)
            self.assertIn("model", cfg)

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


if __name__ == "__main__":
    unittest.main()
