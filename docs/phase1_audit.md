# Phase 1 Scientific Audit: Evidence Verification of Prior Claims

**Project:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection  
**Reviewers:** Senior Computer Vision Research Advisor & ML Reproducibility Engineer  
**Date:** 2026-10-29  
**Branch:** `main` (commit `c7e5036`)  

---

## 1. Audit Methodology & Verification Taxonomy

Every technical, architectural, and experimental claim made across Phase 0, Phase 0.1, and Phase 0.2 is classified into one of four strict scientific categories:
- **`[VERIFIED]`**: Grounded by accessible public primary sources (arXiv HTML/PDF, official publisher proceedings, verified open-source repository commit, or actual passing unit test on local hardware).
- **`[PARTIALLY VERIFIED]`**: Verified mathematically, syntactically, or on synthetic tensors, but lacks empirical verification in an end-to-end training run on physical hardware with real industrial data.
- **`[UNVERIFIED]`**: Stated in documentation as an estimate, projection, or assumption without empirical logs, accessible publisher PDF, or executed benchmark runs.
- **`[INCORRECT]`**: Factually mistaken, contradicted by code, or violating scientific machine learning validation protocols.

---

## 2. Item-by-Item Verification Registry

| Claim / Artifact | Phase Claimed | Audit Finding & Evidence | Scientific Status |
| :--- | :--- | :--- | :--- |
| **STFPM Paper Formulations & Headline Numbers** | Phase 0.2 | Verified against arXiv:2103.04257 v3 HTML (BMVC 2021). Tables 1–2 report ResNet-18 mean Image AUROC: **0.955**, Pixel AUROC: **0.970**, PRO: **0.921**. Weakest category: Transistor (0.825). | **`[VERIFIED]`** |
| **STFPM Official Code Provenance & License** | Phase 0.2 | GitHub `gdwang08/STFPM` (Beihang University, commit `2598a5e`). License is **GPL-3.0**. Deprecated `np.bool` and torchvision `pretrained=True` identified in code. Intel Anomalib implementation verified under **Apache-2.0**. | **`[VERIFIED]`** |
| **CDA Official Code Completeness & Bug Inventory** | Phase 0.1 | Cloned and audited `WHUer-cloud/CDA` (commit `67224ca`). Identified 6 real bugs: `dataloder.py` import typo, missing `parents=True`, hardcoded dataset paths, Windows slash split error, fixed $16\times16$ sequence length, global device binding. | **`[VERIFIED]`** |
| **Clean-Room Modular Codebase & Unit Tests** | Phase 0.1 | 12 unit tests in `tests/` pass cleanly in 12.5s on local Python 3.13 / PyTorch 2.11 (`test_dataset`, `test_inference`, `test_losses`, `test_metrics`, `test_model_shapes`). Tensor shapes and backward gradients verified. | **`[VERIFIED]`** |
| **SK-RD4AD Paper & Code Provenance** | Phase 1 | Verified via CVF Open Access (CVPRW 2025, VAND) and official GitHub `pej0918/SK-RD4AD`. PyTorch-based skip-connected reverse distillation. | **`[VERIFIED]`** |
| **CDA Parameter Counts** | Phase 0.1 | Teacher: 24.86M, GCM: 19.70M, GSU Decoder: 6.96M, Adapters `c11..c32`: 16.52M (38.25% of student), Total Trainable: 43.18M. Verified on synthetic tensors via PyTorch `count_parameters()`. | **`[PARTIALLY VERIFIED]`**<br/>*(Verified in PyTorch code; unverified against paywalled paper text)* |
| **CDA Full-Text Paper & Per-Category Tables** | Phase 0.1 | Elsevier full-text PDF (`10.1016/j.patcog.2025.112761`) is behind a subscription paywall. Per-category numerical tables for MVTec AD and VisA cannot be verified directly. | **`[UNVERIFIED]`** |
| **CDA Training Times & GPU Memory on Kaggle** | Phase 0.2 | Statements such as *"160 epochs takes ~48 minutes on Tesla T4, peak VRAM ~5.8 GB"* are theoretical back-of-the-envelope projections based on batch size and tensor sizes. **No full 160-epoch training run has yet executed on Kaggle.** | **`[UNVERIFIED]`**<br/>*(Theoretical Estimate Only)* |
| **STFPM Training Times on Kaggle** | Phase 0.2 | Statements such as *"~4.5 minutes on Tesla T4, peak VRAM ~2.1 GB"* are theoretical estimates. Must be measured during Stage B/C smoke test. | **`[UNVERIFIED]`**<br/>*(Theoretical Estimate Only)* |
| **MVTec AD Benchmark Results for B0, A1–A3, B1–B3** | Phase 0.2 | Master Experiment Matrix (`docs/experiment_matrix.md`) shows all benchmark runs (`R00`–`R10`) marked as `PLANNED`. No weights exist in `outputs/`. | **`[UNVERIFIED]`**<br/>*(Explicitly acknowledged as pending execution)* |
| **Test-Set Checkpoint Selection as Valid Protocol** | Upstream Repos | Upstream `WHUer-cloud/CDA` and older RD4AD scripts evaluated test set every 5 epochs and saved `best_img.pth` on test AUROC. This is test-set leakage. | **`[INCORRECT]`**<br/>*(Upstream method flawed; patched in our protocol)* |

---

## 3. Corrective Measures Enforced in Phase 1

1. **Zero Empirical Claims Without Logs:** No latency, runtime, VRAM, or AUROC number may be cited as an experimental result unless accompanied by a Kaggle session log, random seed, hardware timestamp, and saved artifact.
2. **Explicit Labeling of Estimates:** All pre-experimental calculations must be explicitly flagged as `[ESTIMATED / THEORETICAL]` rather than `[VERIFIED]`.
3. **Strict Baseline Elevation:** STFPM is elevated to the primary execution baseline to establish verified empirical ground truth on Kaggle before any other model is benchmarked.
4. **SK-RD4AD as Modern Comparator:** SK-RD4AD replaces unverified CDA claims as our verified 2025 modern comparator candidate.
