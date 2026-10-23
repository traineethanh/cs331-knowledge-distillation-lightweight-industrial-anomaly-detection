# CS331 — Knowledge Distillation-Based Lightweight Industrial Anomaly Detection

> **Course Project:** Advanced Computer Vision (CS331)  
> **Research Team:** Thành (Researcher A — Reliable Localization & Calibration) & Luân (Researcher B — Structural Compression & Efficiency)  
> **Primary Baseline Paper:** Ronglong Hu, Jizheng Yi, Xiaoyao Li, Aibin Chen, and Guangjie Han, *"Efficient industrial anomaly detection via cross-scale distillation with enhanced feature compression,"* **Pattern Recognition**, Vol. 173, Article 112761, 2026 (`https://doi.org/10.1016/j.patcog.2025.112761`).  
> **Official Author Repository Audited:** `https://github.com/WHUer-cloud/CDA`

---

## 1. Project Motivation and Objectives

Unsupervised Industrial Anomaly Detection (IAD) seeks to identify defective products and localize anomalous pixels using only defect-free training images. Recent Reverse Knowledge Distillation frameworks—most notably **CDA (*Pattern Recognition*, 2026)**—combine a frozen pretrained teacher encoder (`Wide_ResNet50_2`) with a **Global-local Compression Module (`GCM`)**, a **Gated-depth Sequence Unit (`GSU`)** decoder, and **all-to-all cross-scale feature alignment**.

Our in-depth mathematical and code audit of the official CDA repository ([docs/official_code_audit.md](docs/official_code_audit.md)) uncovered two critical scientific opportunities:
1. **The Cross-Scale Alignment Parameter Paradox (`16.52M` Parameters = 38.25% of Student):** While CDA's 3-stage `GSU` student decoder contains only **`6.96M` parameters**, the six dense $3\times3$ cross-scale channel-alignment convolutions (`c11..c32`) consume **`16.52M` parameters** ($2.37\times$ larger than the decoder itself!), and `GCM.res` consumes another **`9.45M` parameters**.
2. **Static Uncalibrated Fusion & 50:1 Cross-Scale Under-Weighting:** At inference time, CDA linearly sums same-scale discrepancy maps ($a_1, a_2, a_3$, weight `2.0`) and cross-scale maps ($ac_1, ac_2, ac_3$, weight `0.02–0.04`) without normalizing for scale-specific background variance on textured surfaces, applies a fixed $\sigma=4$ Gaussian blur across all scales, and relies on single-pixel `max()` pooling for image-level scoring.

### Research Objectives
1. **Reproduce a Clean, Bug-Fixed CDA Baseline (`B0`):** Resolve the 6 software bugs and test-set checkpoint selection leakage identified in the official repository while preserving exact mathematical equivalence.
2. **Direction A (Thành — Researcher A):** Investigate **Normal-Calibrated & Cross-Scale Reliability-Guided Anomaly Map Fusion (`A1`–`A3`)** to suppress texture false positives and preserve small defects without retraining overhead.
3. **Direction B (Luân — Researcher B):** Investigate **Structural Adapter Factorization & Lightweight GCM Compression (`B1`–`B3`)**, testing whether cross-scale adapters act primarily as training-time regularizers (`B1: -38.3%` active inference params) and factorizing dense $3\times3$ adapters/bottlenecks (`B2: -33.9%` params, `B3: -54.9%` params).

---

## 2. Current Project Status (Phase 0 Complete)

| Work Package | Status | Primary Artifacts |
| :--- | :--- | :--- |
| **WP0 — Workspace & Environment Setup** | **Completed** | Dedicated repository scaffold & environment checks |
| **WP1 — Deep Paper Analysis** | **Completed** | [paper_analysis.md](docs/paper_analysis.md), [paper_equations.md](docs/paper_equations.md), [paper_access_and_verification.md](docs/paper_access_and_verification.md) |
| **WP2 — Official Code Audit** | **Completed** | [official_code_audit.md](docs/official_code_audit.md), [paper_to_code_mapping.md](docs/paper_to_code_mapping.md) |
| **WP3 — Literature Review (10+ Papers)** | **Completed** | [literature_review.md](docs/literature_review.md), [novelty_gap_analysis.md](docs/novelty_gap_analysis.md) |
| **WP4 — Candidate Research Directions** | **Completed** | [research_directions.md](docs/research_directions.md), [research_questions.md](docs/research_questions.md), [novelty_risk_register.md](docs/novelty_risk_register.md) |
| **WP5 — Team Work Plan (Thành & Luân)** | **Completed** | [team_work_plan.md](docs/team_work_plan.md), [milestones.md](docs/milestones.md) |
| **WP6 — Baseline & Kaggle Execution Spec** | **Completed** | [baseline_specification.md](docs/baseline_specification.md), [kaggle_execution_plan.md](docs/kaggle_execution_plan.md), [experimental_protocol.md](docs/experimental_protocol.md), [experiment_matrix.md](docs/experiment_matrix.md) |
| **WP7 & WP8 — Modular Codebase & Unit Tests** | **Completed** | `src/anomaly_detection/`, `scripts/`, `configs/`, `tests/` (All unit tests verified) |
| **Stage 3–6 Kaggle Benchmark Runs** | **Pending Execution on Kaggle GPU** | No benchmark training numbers are claimed yet; see [experiment_matrix.md](docs/experiment_matrix.md). |

---

## 3. Environment Setup

```bash
# Clone repository
git clone https://github.com/traineethanh/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection.git
cd cs331-knowledge-distillation-lightweight-industrial-anomaly-detection

# Install dependencies (compatible with local Python >= 3.10 and Kaggle GPU)
pip install -r requirements.txt
pip install -e .

# Verify environment & hardware
python scripts/check_environment.py

# Run unit test suite (requires no dataset download)
python -m unittest discover -s tests -v
```

---

## 4. Dataset Preparation (MVTec AD)

1. **On Kaggle Notebooks:** Add the public `mvtec-ad` dataset to your notebook input (`/kaggle/input/mvtec-ad`).
2. **Local Setup:** Extract MVTec AD to `./datasets/mvtec/` (never committed to Git).
3. **Validate Dataset Structure and Split Integrity:**
   ```bash
   python scripts/validate_dataset.py --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw
   ```

---

## 5. Baseline Reproduction, Evaluation, and Profiling Commands

```bash
# 1. Stage 3 Smoke Test (2 epochs on carpet + bottle)
python scripts/train.py --config configs/experiments/smoke_test.yaml --data-root /kaggle/input/mvtec-ad --categories carpet bottle

# 2. Stage 4 Paper-Faithful CDA Baseline (160 epochs)
python scripts/train.py --config configs/baseline/cda_mvtec.yaml --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw

# 3. Evaluate a Saved Checkpoint (Baseline or Research A/B Variants)
python scripts/evaluate.py --config configs/baseline/cda_mvtec.yaml --data-root /kaggle/input/mvtec-ad --checkpoint outputs/baseline/carpet/checkpoints/last.pth --category carpet

# 4. Profile Parameters, FLOPs/MACs, Latency, and Peak Memory
python scripts/benchmark_latency.py --variants B0 B1_aux_csd B2_sep_adapter B3_cda_lite --img-size 256 --warmup 50 --iters 200
```

---

## 6. Team Responsibilities and Git Branches

- **`main`:** Stable, reviewed code and verified documentation.
- **`develop`:** Integration branch for joint testing.
- **`feature/thanh-reliability` (Thành — Researcher A):** Normal-distribution statistical calibration (`A1`), cross-scale concordance reliability gating (`A2`), scale-adaptive smoothing & top-$q\%$ scoring (`A3`).
- **`feature/luan-efficiency` (Luân — Researcher B):** Inference-time cross-scale pruning (`B1: CDA-AuxCSD`), depthwise-separable cross-scale adapters (`B2: CDA-SepAdapter`), factorized GCM (`B3: CDA-Lite`), and hardware cost profiling.

---

## 7. Known Limitations, Licensing, and Attribution Notes

1. **Full-Text Paper PDF Access:** As documented in [docs/paper_access_and_verification.md](docs/paper_access_and_verification.md), the Elsevier publisher PDF (`10.1016/j.patcog.2025.112761`) is behind a subscription paywall; our reconstruction is verified against the paper's public metadata and all 8 source files of the official repository (`https://github.com/WHUer-cloud/CDA`).
2. **Source Code Provenance & License:** Our code in `src/anomaly_detection/` is a clean, modular, bug-fixed academic implementation written for CS331, released under the MIT License for our original code, with full academic attribution to Hu et al. (2026) (`WHUer-cloud/CDA`, which does not include a license file in its upstream repository).
3. **Dataset License:** MVTec AD is licensed under **CC BY-NC-SA 4.0** (non-commercial academic research only). Never upload dataset images or weights to public repositories without checking license terms.
