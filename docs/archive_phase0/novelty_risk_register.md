# Novelty Risk Register and Scientific Claim Boundaries

This register tracks scientific, methodological, and novelty risks for our CS331 research project and defines strict mitigation protocols and claim boundaries for the final report.

---

## 1. Novelty and Prior-Art Risks

| Risk ID | Risk Description | Prior Art / Overlap Source | Severity | Mitigation & Honest Framing Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **NR-01** | Claiming "cross-scale knowledge distillation" or "global-local bottleneck compression" as our own contribution. | **CDA** (Hu et al., *Pattern Recognition* 2026) already proposes Cross-Scale Distillation (`c11..c32`) and `GCM` (`CLE` + `SGA`). | **Critical** | Explicitly credit Hu et al. (2026) as the baseline author of CDA, GCM, CLE, SGA, and GSU. Frame all our work as **post-CDA analysis and targeted improvements**. |
| **NR-02** | Claiming "Mamba / gated sequence decoder" as a novel architecture. | **MambaAD** (NeurIPS 2024) and **CDA's `GSU`** (2026) already use gated 1D depthwise conv sequence blocks. | **High** | Keep `GSU` untouched as the shared baseline decoder; do not present `GSU` as our invention. |
| **NR-03** | Claiming normal-distribution quantile normalization of anomaly maps is brand new. | **EfficientAD** (WACV 2024) uses validation quantile normalization to merge PDN and Autoencoder maps. | **Medium** | Cite EfficientAD explicitly for normal-distribution map normalization; specify that our **Direction A** investigates **same-scale vs. cross-scale concordance gating ($a_k \odot ac_k$)** and **scale-adaptive smoothing** specifically to fix CDA's 50:1 cross-scale weight imbalance. |
| **NR-04** | Claiming depthwise-separable convolutions are a newly invented operator. | MobileNet (2017), ConvNeXt (2022), etc. | **Medium** | Frame **Direction B** not as inventing separable convolutions, but as **discovering and resolving CDA's 16.52M Cross-Scale Adapter Parameter Paradox** (where `c11..c32` are $2.37\times$ larger than the GSU decoder) and decoupling the **training-regularizer vs. inference-predictor role** of cross-scale branches (`B1` vs. `B2` vs. `B3`). |

---

## 2. Experimental Validity & Threats-to-Validity Risks

| Risk ID | Threat to Validity | Cause in Official Code or Setup | Severity | Prevention Rule in Our Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **EV-01** | **Test-Set Checkpoint Selection Leakage** | Official `train.py` evaluates on the test set every 5 epochs and picks `best_img.pth` and `best_pix.pth` independently on test labels/masks. | **Critical** | Save checkpoints at fixed final epoch (`last.pth`, epoch 160) AND track validation-split selection (`val_best.pth`) in addition to logging the official repo's `test_best` numbers for reference only. All comparisons between `B0`, `A1..A3`, and `B1..B3` must use the **same fixed-epoch / validation-selected checkpoint rule**. |
| **EV-02** | **Test-Set Calibration Leakage in Direction A** | Computing $\mu_k, \sigma_k$ or quantiles on the test set before fusing maps. | **Critical** | Calibration statistics (`stats.mu`, `stats.std`, `stats.quantiles`) **must** be computed strictly on normal training/validation images (`train/good`), never on test images. |
| **EV-03** | **Cross-Hardware Timing Incomparability** | Comparing latency measured on a Kaggle Tesla T4 (or P100) against paper timings measured on an RTX 3090/4090 or local CPU. | **High** | Measure `B0`, `A1..A3`, and `B1..B3` in the **exact same Kaggle GPU session** with identical warmup (`50` iters), timed iterations (`200` iters), batch size (`1` and `4`), and `torch.cuda.synchronize()`. |
| **EV-04** | **Asymmetric Cross-Scale Loss Formula Ambiguity** | Official `train.py` weights `0.4 * l1n + 1.0 * l1m`, whereas `metric.py` weights `0.4 * (l1n + l1m)`. | **Medium** | Keep `cross_loss_mode: "official_train"` frozen across `B0` and `B1..B3` so the loss function is never a confounding variable when evaluating architecture compression. |
| **EV-05** | **Single-Metric Over-Interpretation** | Reporting only mean Image-AUROC (which saturates $>98\%$) while ignoring Pixel-AP, AUPRO, or small-defect categories. | **High** | Always report **5 quality metrics** (Image-AUROC, Image-AP, Pixel-AUROC, Pixel-AP, AUPRO) and **4 cost metrics** (Params, MACs/FLOPs, Latency ms/img, Peak VRAM MB) per category. |
