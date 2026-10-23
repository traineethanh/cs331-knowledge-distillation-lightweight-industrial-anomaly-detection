"""Unit tests for CDA distillation loss functions and gradient propagation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anomaly_detection.losses import CDALoss, cosine_distance_map


class TestCDALoss(unittest.TestCase):
    def test_zero_distance_when_features_identical(self) -> None:
        e1 = torch.randn(2, 256, 16, 16)
        e2 = torch.randn(2, 512, 8, 8)
        e3 = torch.randn(2, 1024, 4, 4)

        dist = cosine_distance_map(e1, e1.clone())
        self.assertEqual(tuple(dist.shape), (2, 16, 16))
        self.assertAlmostEqual(float(dist.abs().max().item()), 0.0, places=5)

        loss_fn = CDALoss(cross_loss_mode="official_train")
        student_outputs = [e1, e2, e3, e1, e1, e2, e2, e3, e3]
        total_loss, parts = loss_fn([e1, e2, e3], student_outputs)
        self.assertTrue(torch.isfinite(total_loss))
        self.assertAlmostEqual(parts["loss_total"], 0.0, places=5)

    def test_loss_finiteness_and_backward_gradients(self) -> None:
        e1 = torch.randn(2, 256, 16, 16)
        e2 = torch.randn(2, 512, 8, 8)
        e3 = torch.randn(2, 1024, 4, 4)

        d1 = torch.randn(2, 256, 16, 16, requires_grad=True)
        d2 = torch.randn(2, 512, 8, 8, requires_grad=True)
        d3 = torch.randn(2, 1024, 4, 4, requires_grad=True)
        l1m = torch.randn(2, 256, 16, 16, requires_grad=True)
        l1n = torch.randn(2, 256, 16, 16, requires_grad=True)
        l2m = torch.randn(2, 512, 8, 8, requires_grad=True)
        l2n = torch.randn(2, 512, 8, 8, requires_grad=True)
        l3m = torch.randn(2, 1024, 4, 4, requires_grad=True)
        l3n = torch.randn(2, 1024, 4, 4, requires_grad=True)

        for mode in ("official_train", "symmetric_metric"):
            loss_fn = CDALoss(cross_loss_mode=mode)
            loss, parts = loss_fn([e1, e2, e3], [d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n])
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(parts["loss_total"], 0.0)
            loss.backward()
            self.assertIsNotNone(d1.grad)
            self.assertTrue(torch.isfinite(d1.grad).all())


if __name__ == "__main__":
    unittest.main()
