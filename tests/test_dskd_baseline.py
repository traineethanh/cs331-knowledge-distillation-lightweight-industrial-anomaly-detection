"""Local sanity verification test for DSKD baseline architecture and loss."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.anomaly_detection.inference.anomaly_scorer import DSKDAnomalyScorer
from src.anomaly_detection.losses.dskd_loss import DSKDLoss
from src.anomaly_detection.models.dskd import DSKDModel


def test_dskd_forward_and_backward() -> None:
    print("Testing DSKDModel forward and backward pass...")
    device = torch.device("cpu")

    # 1. Test model instantiation with detach=True
    model = DSKDModel(
        pretrained_teacher=False,  # Use random weights for fast local test
        has_f3_conv=True,
        dfe_input_type="raw",
        detach_dfe_input=True,
        upsample_mode="bilinear",
    ).to(device)

    # 2. Check parameter separation
    enc_params = model.get_encoder_params()
    dec_params = model.get_decoder_system_params()
    enc_set = set(id(p) for p in enc_params)
    dec_set = set(id(p) for p in dec_params)

    assert len(enc_params) > 0, "Encoder parameters should not be empty!"
    assert len(dec_params) > 0, "Decoder parameters should not be empty!"
    assert enc_set.isdisjoint(dec_set), "Encoder and Decoder parameters must be strictly disjoint!"
    print(f"  [PASS] Parameter sets disjoint: {len(enc_params)} enc params, {len(dec_params)} dec params.")

    # 3. Test forward pass
    x = torch.randn(2, 3, 256, 256, device=device)
    f_t, f_e, f_d = model(x)

    assert len(f_t) == 3 and len(f_e) == 3 and len(f_d) == 3, "Should extract 3 feature stages!"
    assert f_t[0].shape == (2, 64, 64, 64), f"Stage 1 shape mismatch: {f_t[0].shape}"
    assert f_t[1].shape == (2, 128, 32, 32), f"Stage 2 shape mismatch: {f_t[1].shape}"
    assert f_t[2].shape == (2, 256, 16, 16), f"Stage 3 shape mismatch: {f_t[2].shape}"
    assert f_d[0].shape == (2, 64, 64, 64), f"Decoder stage 1 shape mismatch: {f_d[0].shape}"
    assert f_d[1].shape == (2, 128, 32, 32), f"Decoder stage 2 shape mismatch: {f_d[1].shape}"
    assert f_d[2].shape == (2, 256, 16, 16), f"Decoder stage 3 shape mismatch: {f_d[2].shape}"
    print("  [PASS] Multi-scale feature shapes matched exactly.")

    # 4. Test Loss computation
    loss_fn = DSKDLoss(loss_lambda=0.1, norm_variant="N1")
    loss_e, loss_d, maps_te, maps_td = loss_fn(f_t, f_e, f_d)

    assert not torch.isnan(loss_e) and not torch.isinf(loss_e), "Loss E contains NaN/Inf!"
    assert not torch.isnan(loss_d) and not torch.isinf(loss_d), "Loss D contains NaN/Inf!"
    assert loss_e.item() > 0 and loss_d.item() > 0, "Losses should be positive!"
    print(f"  [PASS] Losses computed: Loss_E = {loss_e.item():.4f}, Loss_D = {loss_d.item():.4f}")

    # 5. Test Backward pass
    optim_e = torch.optim.Adam(enc_params, lr=1e-3, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(dec_params, lr=1e-3, betas=(0.5, 0.999))

    optim_e.zero_grad()
    optim_d.zero_grad()

    loss_e.backward()
    loss_d.backward()

    # Verify gradients exist
    has_enc_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in enc_params)
    has_dec_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in dec_params)
    assert has_enc_grad, "Encoder should have non-zero gradients!"
    assert has_dec_grad, "Decoder should have non-zero gradients!"

    optim_e.step()
    optim_d.step()
    print("  [PASS] Backward pass and optimizer step executed successfully.")

    # 6. Test Anomaly Scorer
    scorer = DSKDAnomalyScorer(img_size=256, loss_lambda=0.1, gaussian_sigma=4.0)
    pred_maps, pred_scores = scorer.predict_batch(model, x)

    assert pred_maps.shape == (2, 256, 256), f"Anomaly map shape mismatch: {pred_maps.shape}"
    assert pred_scores.shape == (2,), f"Score shape mismatch: {pred_scores.shape}"
    assert not np.isnan(pred_maps).any(), "Predicted maps contain NaN!"
    assert not np.isnan(pred_scores).any(), "Predicted scores contain NaN!"
    print(f"  [PASS] Anomaly scorer produced valid smoothed maps (shape: {pred_maps.shape}) and scores.")

    # 7. Test model with detach_dfe_input=False (joint mode)
    model_joint = DSKDModel(
        pretrained_teacher=False,
        detach_dfe_input=False,
    ).to(device)

    f_t, f_e, f_d = model_joint(x)
    l_e, l_d, _, _ = loss_fn(f_t, f_e, f_d)
    l_e.backward(retain_graph=True)
    l_d.backward()
    print("  [PASS] Joint mode (detach=False) backward pass works with retain_graph=True.")

    print("\nALL SANITY TESTS PASSED! DSKD Baseline implementation is verified.")


if __name__ == "__main__":
    test_dskd_forward_and_backward()
