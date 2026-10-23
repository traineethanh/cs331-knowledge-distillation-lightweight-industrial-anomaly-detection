"""Unit tests for Image/Pixel AUROC, AP, and Per-Region Overlap (AUPRO) metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from anomaly_detection.evaluation import evaluate_predictions


class TestEvaluationMetrics(unittest.TestCase):
    def test_perfect_synthetic_predictions_at_non_256_resolution(self) -> None:
        # Tests that 64x64 resolution works cleanly (fixing the official 256*256 hard-code bug)
        gts = np.zeros((4, 64, 64), dtype=np.float32)
        gts[2, 10:20, 10:20] = 1.0
        gts[3, 40:45, 40:45] = 1.0  # Small defect region
        labels = [0, 0, 1, 1]

        # Construct anomaly maps that cleanly separate foreground defects from background
        maps = np.random.RandomState(42).uniform(0.0, 0.1, size=(4, 64, 64)).astype(np.float32)
        maps[2, 10:20, 10:20] = 0.95
        maps[3, 40:45, 40:45] = 0.92

        res = evaluate_predictions(anomaly_maps=maps, image_labels=labels, gt_masks=gts)
        self.assertEqual(res["auroc_img"], 1.0)
        self.assertEqual(res["ap_img"], 1.0)
        self.assertEqual(res["auroc_pix"], 1.0)
        self.assertEqual(res["ap_pix"], 1.0)
        self.assertGreater(res["aupro_30"], 0.95)
        self.assertGreater(res["aupro_05"], 0.95)


if __name__ == "__main__":
    unittest.main()
