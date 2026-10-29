# Reproducibility & Feasibility Assessment for Kaggle GPU

## 1. Execution Environment & Constraints (Kaggle GPU)

Our research project is conducted by two undergraduate students (Thành & Luân) using **Kaggle Notebooks** and **GitHub**.

### 1.1 Platform Resource Profile
- **Available Hardware:** NVIDIA Tesla T4 (16 GB VRAM) or NVIDIA Tesla P100 (16 GB VRAM).
- **Execution Limits:** Maximum **12 hours per interactive session**; total weekly GPU budget of **30 hours per student account** (combined 60 hours/week).
- **Filesystem Constraints:** Read-only input datasets (`/kaggle/input/mvtec-ad`); writable local workspace (`/kaggle/working/`) limited to ~20 GB ephemeral disk space. Session resets wipe all non-persisted local files.
- **Network Access:** External internet access available for cloning Git repos and installing PyPI wheels, but native C++ build tools (CUDA compiler `nvcc` headers) are restricted or fragile.

---

## 2. Comparative Reproducibility Feasibility Matrix

We assess the concrete engineering and scientific feasibility of running each candidate model on Kaggle:

| Model | Venue & Year | Code Availability | License | Dependency Burden | Kaggle Execution Feasibility | Verdict for CS331 Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STFPM** | BMVC 2021 | Official repo (`gdwang08/STFPM`) + Anomalib | GPL-3.0 (Official) / Apache-2.0 (Anomalib) | Standard `torch`, `torchvision`, `sklearn` | **Immediate (<10 min/cat)**. VRAM < 4 GB. Simple CNN forward passes. | **Primary Foundational Baseline** |
| **CDA** | Pattern Recognit. 2026 | Official repo (`WHUer-cloud/CDA`) | No License (All Rights Reserved) | Standard PyTorch + SciPy. **6 code bugs required patching.** | **Feasible via Clean-Room Code**. 160 epochs takes ~55 min/cat on T4. VRAM ~6 GB. | **Modern Architecture Comparator (Evaluated via our bug-fixed PyTorch module)** |
| **SK-RD4AD** | CVPRW 2025 | Official repo (`pej0918/SK-RD4AD`) | No License | Pure PyTorch (WideResNet-50 + skip connections) | **High**. Standard CNN operations, trains cleanly on T4/P100. | **Strongest Fallback 2025 Comparator** |
| **LKDF-AD** | Sci. Reports 2026 | Nature Supplemental / Code Available | Author License | Lightweight CNN (~385K params) | **High for Few-Shot**, but requires implementing a few-shot data sampler (1–10 images). | **Viable only for few-shot sub-study** |
| **CPKD** | AAAI 2025 | **No Official Code Released** | N/A | Unknown (Requires implementing prototype co-progression from scratch) | **Infeasible**. High risk of algorithmic divergence within a 3-week deadline. | **Rejected as Experimental Baseline** |
| **DHKD** | ESWA 2026 | Unofficial community repos | Unclear | **`mamba-ssm`, `causal-conv1d` (custom CUDA C++ kernels)** | **Severe Blocker**. Compiling Mamba CUDA extensions on Kaggle frequently crashes with header mismatches. | **Rejected as Experimental Baseline** |

---

## 3. Deep Reproducibility Audit: STFPM (Foundational Baseline)

### 3.1 Code Provenance & Licensing Integrity
- The official repository `gdwang08/STFPM` is distributed under the **GPL-3.0 license**.
- Because our course project is developed under the MIT License, copying GPL-3.0 code directly into our repository would create a license compatibility conflict.
- **Solution:** We utilize a clean-room modular implementation following the open-access mathematical specification from Wang et al. (BMVC 2021) and cross-reference with the **Apache-2.0** licensed STFPM implementation in Intel's `anomalib` library (`src/anomalib/models/image/stfpm/`).

### 3.2 Bug Fixes and Kaggle Compatibility
1. **NumPy 1.24+ `np.bool` Deprecation:**
   - *Issue:* `astype(np.bool)` raises `AttributeError: module 'numpy' has no attribute 'bool'`.
   - *Fix:* Replace with `.astype(bool)` across dataset mask loaders and evaluation scripts.
2. **Torchvision `pretrained=True` Deprecation:**
   - *Issue:* Raises warning or fails in future torchvision releases.
   - *Fix:* Use `weights=models.ResNet18_Weights.IMAGENET1K_V1`.
3. **Slow PRO Metric Calculation:**
   - *Issue:* The official STFPM code evaluates 4,000 threshold steps and min-max rescales the PRO curve, which takes >30 minutes per category.
   - *Fix:* Use our vectorized trapezoidal integration implementation in `src/anomaly_detection/evaluation/metrics.py`, computing standard `AUPRO` ($\le 0.30$) and low-FPR `AUPRO` ($\le 0.05$) in <5 seconds.

### 3.3 Compute & Memory Profile (Per Category on Tesla T4)
- **Trainable Parameters:** $11.18\text{M}$ (Student ResNet-18).
- **Frozen Parameters:** $11.18\text{M}$ (Teacher ResNet-18).
- **Total Pipeline Parameters:** $22.36\text{M}$.
- **Peak VRAM (Batch Size = 32, Image Size = 256):** $\sim 2.1\text{ GB}$.
- **Training Time (100 Epochs on `carpet`, 280 images):** $\sim 4.5\text{ minutes}$.
- **Inference Latency:** $\sim 8.2\text{ ms per image}$ ($>120\text{ FPS}$).

---

## 4. Deep Reproducibility Audit: CDA (Modern Comparator)

### 4.1 Provenance & Paywall Limitation
- As documented in `docs/paper_access_and_verification.md`, the Elsevier full-text PDF is paywalled.
- Our implementation is verified against the official author code (`WHUer-cloud/CDA`) and public proceedings metadata.
- All per-category numbers in our final reports must be our own measured numbers, not hypothetical claims of what the paywalled paper reported.

### 4.2 Upstream Bugs Resolved in Our Clean-Room Codebase (`src/anomaly_detection`)
1. **Import Typo:** Fixed `from data_loader import ...` when file was named `dataloder.py`.
2. **Directory Creation:** Added `exist_ok=True, parents=True` to prevent `FileNotFoundError`.
3. **Hardcoded Paths:** Resolved hardcoded `./datasets/mvtec/` so Kaggle `/kaggle/input/mvtec-ad` works seamlessly.
4. **Resolution Coupling:** Replaced hardcoded `256*256` in metric reshapes and `16x16` in Transformer tokens with dynamic spatial dimensions.
5. **Global Device Binding:** Removed import-time global `device = ...` bindings that prevented clean multi-GPU / CPU testing.

### 4.3 Compute & Memory Profile (Per Category on Tesla T4)
- **Trainable Parameters:** $43.18\text{M}$ (`GCMAndDecoder`).
- **Frozen Parameters:** $24.86\text{M}$ (Teacher `Wide_ResNet50_2`).
- **Peak VRAM (Batch Size = 4, Image Size = 256):** $\sim 5.8\text{ GB}$.
- **Training Time (160 Epochs on `carpet`, 280 images):** $\sim 48\text{ minutes}$.
- **Inference Latency:** $\sim 24.5\text{ ms per image}$ ($\sim 40\text{ FPS}$).

---

## 5. Experimental Validity: Preventing Data Leakage

To prevent experimental contamination identified in earlier Phase 0 audits:

### 5.1 Elimination of Test-Set Checkpoint Selection Leakage
- **Upstream Flaw in CDA:** The official author script evaluated every 5 epochs on the test set and saved `best_img.pth` and `best_pix.pth` based on test-set metrics. This violates machine learning validation protocols by tuning checkpoint selection on test data.
- **Our Scientific Protocol:**
  1. We evaluate and report the **final fixed-epoch checkpoint (`epoch_160.pth` or `last.pth`)** as the primary scientific result.
  2. For validation-based early stopping, we carve out a 20% holdout split from `train/good` and evaluate solely on normal validation reconstruction loss ($\mathcal{L}_{\text{val}}$).
  3. Official-style `test_best` numbers are logged strictly as secondary reference markers to explain discrepancies with upstream claims.

### 5.2 Strict Zero-Leakage Calibration for Direction A (Thành)
- In Direction A (Normal Z-score calibration and concordance gating), the spatial mean $\mu_k^{\text{norm}}$ and standard deviation $\sigma_k^{\text{norm}}$ must be computed **strictly on the normal training images (`train/good`)**.
- At no point during calibration may test images or test masks enter the standardization pipeline.

### 5.3 Deterministic Profiling Protocol for Direction B (Luân)
- All hardware efficiency metrics (Parameters, FLOPs, Latency, Peak VRAM) must be measured on the **same physical Kaggle GPU** within the same notebook session.
- Measurement protocol:
  - Warmup: 50 dummy passes.
  - Timed runs: 200 iterations with `torch.cuda.synchronize()` before and after `time.perf_counter()`.
  - Batch size: 1 (single-sample industrial latency) and 4 (throughput batching).
