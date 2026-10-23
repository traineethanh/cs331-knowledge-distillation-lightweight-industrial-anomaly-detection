"""Unit tests for inference without gradient tracking, fusion modes A0-A3, B1 pruning, and checkpoints."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anomaly_detection.inference import CDAAnomalyMapper, predict_batch
from anomaly_detection.models.baseline import TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.training import load_checkpoint, save_checkpoint


class TestInferenceAndCheckpointing(unittest.TestCase):
    def test_inference_modes_without_gradient_tracking(self) -> None:
        encoder = TeacherEncoder(pretrained=False).eval()
        student = build_student_model("B2_sep_adapter").eval()
        imgs = torch.randn(2, 3, 128, 128)

        for mode in ("A0_baseline", "A1_calibrated", "A2_reliability_gate", "A3_adaptive_topq"):
            mapper = CDAAnomalyMapper(img_size=128, fusion_mode=mode)
            maps, scores = predict_batch(encoder, student, imgs, mapper)
            self.assertEqual(maps.shape, (2, 128, 128))
            self.assertEqual(scores.shape, (2,))

    def test_b1_inference_pruning_skips_cross_scale_adapters(self) -> None:
        student_b1 = build_student_model("B1_aux_csd").eval()
        e1 = torch.randn(1, 256, 32, 32)
        e2 = torch.randn(1, 512, 16, 16)
        e3 = torch.randn(1, 1024, 8, 8)

        with torch.no_grad():
            outs = student_b1(e1, e2, e3)
        # First 3 are s1, s2, s3; remaining 6 cross-scale outputs are None in eval mode
        self.assertIsNotNone(outs[0])
        self.assertIsNotNone(outs[1])
        self.assertIsNotNone(outs[2])
        for cross_out in outs[3:]:
            self.assertIsNone(cross_out)

        mapper = CDAAnomalyMapper(img_size=128, fusion_mode="A0_baseline")
        maps, scores = mapper([e1, e2, e3], outs)
        self.assertEqual(maps.shape, (1, 128, 128))
        self.assertEqual(scores.shape, (1,))

    def test_checkpoint_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_file = Path(tmp) / "checkpoints" / "last.pth"
            s1 = build_student_model("B3_cda_lite")
            opt1 = torch.optim.Adam(s1.parameters(), lr=0.005)
            save_checkpoint(ckpt_file, student=s1, optimizer=opt1, epoch=15, best_metrics={"auroc_img": 0.98})

            s2 = build_student_model("B3_cda_lite")
            opt2 = torch.optim.Adam(s2.parameters(), lr=0.005)
            meta = load_checkpoint(ckpt_file, student=s2, optimizer=opt2)
            self.assertEqual(meta["epoch"], 15)
            self.assertAlmostEqual(meta["best_metrics"]["auroc_img"], 0.98)

            for p1, p2 in zip(s1.parameters(), s2.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
