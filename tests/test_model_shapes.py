"""Unit tests for CDA baseline (`B0`) and compressed variants (`B1`-`B3`) shapes and parameter counts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anomaly_detection.models.baseline import GCMAndDecoder, TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.utils import count_parameters


class TestModelShapesAndParameters(unittest.TestCase):
    def test_baseline_parameter_counts_match_official_code(self) -> None:
        student_b0 = GCMAndDecoder()
        self.assertEqual(count_parameters(student_b0), 43_182_851)
        self.assertEqual(count_parameters(student_b0.gcm), 19_704_579)
        self.assertEqual(count_parameters(student_b0.dec), 6_959_616)
        adapter_params = sum(
            count_parameters(getattr(student_b0, name))
            for name in ("c11", "c12", "c21", "c22", "c31", "c32")
        )
        self.assertEqual(adapter_params, 16_518_656)

    def test_research_b_variants_parameter_reduction(self) -> None:
        b1 = build_student_model("B1_aux_csd")
        b2 = build_student_model("B2_sep_adapter")
        b3 = build_student_model("B3_cda_lite")

        self.assertEqual(count_parameters(b1), 43_182_851)
        self.assertEqual(b1.active_inference_parameters(), 26_664_195)
        self.assertEqual(count_parameters(b2), 28_542_211)
        self.assertEqual(count_parameters(b3), 21_216_259)

    def test_forward_shapes_across_variants(self) -> None:
        encoder = TeacherEncoder(pretrained=False).eval()
        self.assertEqual(count_parameters(encoder), 24_862_528)

        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            e1, e2, e3 = encoder(x)

        self.assertEqual(tuple(e1.shape), (2, 256, 32, 32))
        self.assertEqual(tuple(e2.shape), (2, 512, 16, 16))
        self.assertEqual(tuple(e3.shape), (2, 1024, 8, 8))

        for var in ("B0", "B2_sep_adapter", "B3_cda_lite"):
            student = build_student_model(var).eval()
            with torch.no_grad():
                outputs = student(e1, e2, e3)
            self.assertEqual(len(outputs), 9)
            s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n = outputs
            self.assertEqual(tuple(s1.shape), (2, 256, 32, 32))
            self.assertEqual(tuple(s2.shape), (2, 512, 16, 16))
            self.assertEqual(tuple(s3.shape), (2, 1024, 8, 8))
            self.assertEqual(tuple(l1m.shape), (2, 256, 32, 32))
            self.assertEqual(tuple(l1n.shape), (2, 256, 32, 32))
            self.assertEqual(tuple(l2m.shape), (2, 512, 16, 16))
            self.assertEqual(tuple(l2n.shape), (2, 512, 16, 16))
            self.assertEqual(tuple(l3m.shape), (2, 1024, 8, 8))
            self.assertEqual(tuple(l3n.shape), (2, 1024, 8, 8))


if __name__ == "__main__":
    unittest.main()
