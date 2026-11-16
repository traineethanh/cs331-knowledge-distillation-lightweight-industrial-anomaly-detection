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

    def test_evaluate_predictions_shorthand_keyword_aliases(self) -> None:
        """Verify evaluate_predictions accepts shorthand aliases (maps, labels, masks, scores)."""
        gts = np.zeros((2, 32, 32), dtype=np.float32)
        gts[1, 5:10, 5:10] = 1.0
        labels = [0, 1]
        maps = np.zeros((2, 32, 32), dtype=np.float32)
        maps[1, 5:10, 5:10] = 1.0

        res = evaluate_predictions(maps=maps, labels=labels, masks=gts)
        self.assertEqual(res["auroc_img"], 1.0)
        self.assertIn("aupro_30", res)
        self.assertIn("aupro_05", res)

    def test_aupro_edge_cases_and_bounds(self) -> None:
        """Verify AUPRO edge cases: empty masks, zero scores, identical scores."""
        from anomaly_detection.evaluation.metrics import compute_aupro

        # Case 1: All masks are zero (no defect)
        all_zero_gts = np.zeros((2, 32, 32), dtype=np.float32)
        maps = np.random.rand(2, 32, 32)
        self.assertEqual(compute_aupro(maps, all_zero_gts, max_fpr=0.30), 0.0)

        # Case 2: Inverted prediction (anomaly is predicted lower than background)
        gts = np.zeros((2, 32, 32), dtype=np.float32)
        gts[1, 5:10, 5:10] = 1.0
        bad_maps = np.ones((2, 32, 32), dtype=np.float32) * 0.9
        bad_maps[1, 5:10, 5:10] = 0.01  # Defect predicted as very low score
        aupro = compute_aupro(bad_maps, gts, max_fpr=0.30)
        self.assertGreaterEqual(aupro, 0.0)
        self.assertLessEqual(aupro, 0.1)

        # Case 3: Constant maps (min_score == max_score)
        const_maps = np.ones((2, 32, 32), dtype=np.float32)
        self.assertEqual(compute_aupro(const_maps, gts, max_fpr=0.30), 0.0)

        # Case 4: Discrete / tied background scores with foreground score between ties
        # 15% of background pixels have score 0.8 (> defect 0.5), 85% have score 0.0.
        # For threshold in (0.5, 0.8], FPR = 0.15, PRO = 0.0.
        # For threshold in (0.0, 0.5], FPR = 0.15, PRO = 1.0.
        # Theoretical AUPRO@0.30 = (1 / 0.30) * int_0.15^0.30 1.0 dFPR = 0.15 / 0.30 = 0.50.
        tied_gts = np.zeros((1, 20, 20), dtype=np.float32)
        tied_gts[0, 0:4, 0:5] = 1.0  # 20 defect pixels, 380 background pixels
        tied_maps = np.zeros((1, 20, 20), dtype=np.float32)
        tied_maps[0, 0:4, 0:5] = 0.5  # Defect score = 0.5
        # Set exactly 57 of 380 background pixels (15%) to 0.8
        bg_coords = np.argwhere(tied_gts[0] == 0)
        for r, c in bg_coords[:57]:
            tied_maps[0, r, c] = 0.8
        aupro_tied_30 = compute_aupro(tied_maps, tied_gts, max_fpr=0.30)
        aupro_tied_05 = compute_aupro(tied_maps, tied_gts, max_fpr=0.05)
        self.assertAlmostEqual(aupro_tied_30, 0.50, places=2)
        self.assertAlmostEqual(aupro_tied_05, 0.00, places=3)

    def test_aupro_numpy_trapezoid_and_trapz_fallback(self) -> None:
        """Verify compute_aupro works identically with np.trapezoid (NumPy 2.x) and np.trapz (NumPy 1.x)."""
        from types import SimpleNamespace
        from unittest.mock import patch
        import anomaly_detection.evaluation.metrics as metrics_mod
        from anomaly_detection.evaluation.metrics import compute_aupro

        gts = np.zeros((1, 20, 20), dtype=np.float32)
        gts[0, 0:4, 0:5] = 1.0
        maps = np.zeros((1, 20, 20), dtype=np.float32)
        maps[0, 0:4, 0:5] = 0.5
        bg_coords = np.argwhere(gts[0] == 0)
        for r, c in bg_coords[:57]:
            maps[0, r, c] = 0.8

        # Reference integration function (standard composite trapezoidal rule)
        def _ref_trapz(y: np.ndarray, x: np.ndarray) -> float:
            y_arr = np.asarray(y, dtype=np.float64)
            x_arr = np.asarray(x, dtype=np.float64)
            return float(np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) * 0.5))

        base_np_attrs = {
            k: getattr(np, k)
            for k in dir(np)
            if not k.startswith("__") and k not in ("trapezoid", "trapz")
        }

        # Case A: NumPy has `trapezoid` (NumPy 2.x)
        trapezoid_calls = []

        def _spy_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
            trapezoid_calls.append(True)
            return _ref_trapz(y, x)

        np_with_trapezoid = SimpleNamespace(
            **base_np_attrs,
            trapezoid=_spy_trapezoid,
            trapz=None,
        )
        with patch.object(metrics_mod, "np", np_with_trapezoid):
            val_trapezoid = compute_aupro(maps, gts, max_fpr=0.30)
        self.assertEqual(len(trapezoid_calls), 1)
        self.assertAlmostEqual(val_trapezoid, 0.50, places=2)

        # Case B: NumPy lacks `trapezoid` but has `trapz` (NumPy 1.x)
        trapz_calls = []

        def _spy_trapz(y: np.ndarray, x: np.ndarray) -> float:
            trapz_calls.append(True)
            return _ref_trapz(y, x)

        np_with_trapz_only = SimpleNamespace(
            **base_np_attrs,
            trapz=_spy_trapz,
        )
        with patch.object(metrics_mod, "np", np_with_trapz_only):
            val_trapz = compute_aupro(maps, gts, max_fpr=0.30)
        self.assertEqual(len(trapz_calls), 1)
        self.assertAlmostEqual(val_trapz, 0.50, places=2)
        self.assertAlmostEqual(val_trapezoid, val_trapz, places=12)

        # Case C: Neither `trapezoid` nor `trapz` is available -> raises explicit AttributeError
        np_without_both = SimpleNamespace(**base_np_attrs)
        with patch.object(metrics_mod, "np", np_without_both):
            with self.assertRaises(AttributeError):
                compute_aupro(maps, gts, max_fpr=0.30)


if __name__ == "__main__":
    unittest.main()
