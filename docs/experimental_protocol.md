# Scientific Experimental Protocol

This document establishes the non-negotiable experimental rules governing data splits, checkpoint selection, hyperparameter tuning, and cross-variant comparisons in this project.

---

## 1. Data Partitioning and Zero-Leakage Protocol

### 1.1 Standard Unsupervised MVTec AD Split
For each MVTec AD category $c \in \{1, \dots, 15\}$:
- **Training Split (`train/good/*.png`):** Contains exclusively defect-free images ($N_{\text{train}} \in [60, 391]$ depending on category).
- **Normal Calibration / Validation Subset (`val_good`):**
  - When Direction A (`A1`–`A3`) computes normal discrepancy statistics ($\mu_m^{\text{norm}}, \sigma_m^{\text{norm}}$ or quantiles $q_{50}, q_{99}$) or when validation loss is monitored, these statistics are computed **strictly on the normal training images (or a held-out 10% normal validation split from `train/good`)** with the model in `.eval()` mode.
  - **Strict Prohibition:** Ground-truth test labels $y_n$ and ground-truth defect masks $G_n$ from `test/*` and `ground_truth/*` must **never** be used to compute normalization statistics, learn fusion weights, or tune post-processing hyperparameters.

### 1.2 Resolution of the Official Repository's Test-Set Checkpoint Selection Issue
As documented in [official_code_audit.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/official_code_audit.md), the official `train.py` evaluates on the test set every `test_interval = 5` epochs and saves `best_img.pth` and `best_pix.pth` whenever test-set AUROC improves. To maintain both **comparability with the official paper/code** and **strict scientific validity**:
1. **Primary Scientific Reporting (`last_epoch` = Epoch 160):**
   All primary comparisons between our baseline (`B0`/`A0`) and our proposed variants (`A1`–`A3`, `B1`–`B3`, `C1`) use the **fixed final checkpoint at Epoch 160 (`last.pth`)** (or normal-validation-loss selected checkpoint `best_val.pth`). This guarantees zero test-set selection leakage.
2. **Secondary Reference Reporting (`official_best_test`):**
   When comparing our reproduced `B0` baseline against the numbers reported by Hu et al. (2026), we also log `best_test_img` and `best_test_pix` in a separate column clearly labelled *"Official Repo Protocol (Best Test Epoch)"*, quantifying the exact optimism gap $\Delta_{\text{leakage}} = \text{AUROC}_{\text{best\_test}} - \text{AUROC}_{\text{epoch160}}$.

---

## 2. Controlled Single-Variable Ablation Rules

To prevent confounded conclusions, every experiment must obey the **One-Factor-At-A-Time (OFAT)** rule:

| Comparison Pair | What Changes | What Must Remain Strictly Identical |
| :--- | :--- | :--- |
| **`B0` vs. `A1` $\to$ `A2` $\to$ `A3`** | Only the inference-stage `AnomalyMapper` calibration, reliability gating, smoothing $\sigma_k$, and top-$q\%$ aggregation. | **Exact same trained `B0` checkpoint (`last.pth`)**, same teacher weights, same $256\times256$ test images and masks. |
| **`B0` vs. `B1` (`CDA-AuxCSD`)** | Only whether `c11..c32` cross-scale branches are executed and added at inference time (`infer_same_scale_only=True`). | **Exact same trained `B0` checkpoint (`last.pth`)**, same Gaussian $\sigma=4$, same `max()` scoring. |
| **`B0` vs. `B2` (`CDA-SepAdapter`)** | Only the parameterization of `c11..c32` (Dense $3\times3$ `16.52M` $\to$ Depthwise-Separable `1.86M`). | Same `Encoder`, same `GCM`, same `GSU Decoder`, same loss function, same optimizer (`Adam lr=0.005`), same 160 epochs, same seed (`42`), same `A0` evaluation formula. |
| **`B2` vs. `B3` (`CDA-Lite`)** | Only the parameterization of `GCM` `ConvBlock`s (Dense $3\times3$ $\to$ Depthwise-Separable $3\times3$). | Same `SepAdapter` (`c11..c32`), same `GSU Decoder`, same loss, optimizer, epochs, seed, and evaluation formula. |
| **`B2/B3` vs. `C1` (`Combined`)** | Only replacing the static `A0` mapper with Thành's calibrated `A3` mapper on the trained `B2`/`B3` checkpoint. | **Exact same trained `B2`/`B3` checkpoint (`last.pth`)**. |

---

## 3. Category Selection and Staged Benchmark Expansion

Following the project constraints:
1. **Tier 1 — Core Representative Triad (Mandatory for all runs, seeds, and ablations):**
   - `carpet` (Texture — complex woven surface prone to shallow feature false positives).
   - `bottle` (Object — medium/large structural defects).
   - `screw` (Object — tiny thread/tip defects and varying orientations; hardest category in MVTec AD).
2. **Tier 2 — Extended 6-Category Subset (3 Textures + 3 Objects):**
   - Textures: `carpet`, `tile`, `grid` (or `leather`).
   - Objects: `bottle`, `screw`, `capsule` (or `transistor`).
3. **Tier 3 — Full 15-Category MVTec AD Benchmark:**
   - Executed for `B0`, `A1`–`A3`, and `B1` (since `A1`–`A3` and `B1` reuse `B0` checkpoints at zero training cost!), and for `B2`/`B3`/`C1` as Kaggle GPU hours permit.

---

## 4. Reporting Standards for Negative and Mixed Results

Scientific integrity requires reporting all outcomes honestly:
- If `A2`/`A3` improves Pixel-AP and AUPRO on texture categories (`carpet`, `grid`) but slightly lowers Pixel-AUROC on a large-defect object category, **both** numbers must be reported and explained in terms of spatial scale properties.
- If `B3` (`-54.9%` parameters) loses $1.0\%–2.0\%$ Pixel-AP on `screw`, this is **not** a failure of the project; it directly answers **RQ-B3** (*"which metric degrades first under deep student compression?"*).
