"""Unit tests for foundational STFPM baseline."""

import unittest
import torch
from src.anomaly_detection.models.baseline.stfpm import (
    ResNet18FeatureExtractor,
    STFPM,
    STFPMLoss,
    compute_stfpm_anomaly_map,
)


class TestSTFPM(unittest.TestCase):
    """Test STFPM feature extractor, model forward, loss, and anomaly mapping."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.dummy_input = torch.randn(2, 3, 256, 256, device=self.device)

    def test_feature_extractor_shapes(self) -> None:
        """Verify layer1, layer2, layer3 channel and spatial resolutions."""
        extractor = ResNet18FeatureExtractor(pretrained=False).to(self.device)
        feats = extractor(self.dummy_input)

        self.assertEqual(len(feats), 3)
        # layer1: 64 ch, 64x64
        self.assertEqual(feats[0].shape, (2, 64, 64, 64))
        # layer2: 128 ch, 32x32
        self.assertEqual(feats[1].shape, (2, 128, 32, 32))
        # layer3: 256 ch, 16x16
        self.assertEqual(feats[2].shape, (2, 256, 16, 16))

    def test_stfpm_teacher_frozen_and_student_trainable(self) -> None:
        """Confirm teacher weights have requires_grad=False and student has requires_grad=True."""
        model = STFPM(teacher_pretrained=False).to(self.device)
        for param in model.teacher.parameters():
            self.assertFalse(param.requires_grad)

        student_trainable_params = sum(
            p.numel() for p in model.student.parameters() if p.requires_grad
        )
        self.assertGreater(student_trainable_params, 0)
        # Note: A full ResNet-18 is 11.18M, but STFPM truncates after layer3 (omitting layer4 and fc).
        # Exact trainable student parameters = 2,782,784 (~2.78M)
        self.assertEqual(student_trainable_params, 2782784)

    def test_stfpm_loss_and_gradients(self) -> None:
        """Verify normalized L2 loss calculation and backward pass into student."""
        model = STFPM(teacher_pretrained=False).to(self.device)
        loss_fn = STFPMLoss()

        t_feats, s_feats = model(self.dummy_input)
        loss = loss_fn(t_feats, s_feats)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

        # Zero distance when features are identical
        zero_loss = loss_fn(t_feats, t_feats)
        self.assertAlmostEqual(zero_loss.item(), 0.0, places=5)

        # Backward gradients exist on student but not teacher
        loss.backward()
        for param in model.student.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
        for param in model.teacher.parameters():
            self.assertIsNone(param.grad)

    def test_stfpm_anomaly_map_generation(self) -> None:
        """Verify multi-scale anomaly map product and max scoring."""
        model = STFPM(teacher_pretrained=False).to(self.device)
        t_feats, s_feats = model(self.dummy_input)

        maps, scores = compute_stfpm_anomaly_map(t_feats, s_feats, target_size=(256, 256))

        self.assertEqual(maps.shape, (2, 256, 256))
        self.assertEqual(scores.shape, (2,))
        # Anomaly map values are non-negative
        self.assertTrue(torch.all(maps >= 0.0))
        # Image score is maximum of map
        expected_score_0 = torch.max(maps[0]).item()
        self.assertAlmostEqual(scores[0].item(), expected_score_0, places=4)


if __name__ == "__main__":
    unittest.main()
