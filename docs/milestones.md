# Project Milestones and Acceptance Criteria

This document defines the staged execution roadmap, deliverables, acceptance gates, and student ownership from Phase 0 initialization through final CS331 defense.

---

## Milestone Overview Table

| Milestone | Title | Owner | Target Stage | Key Deliverable | Acceptance Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M0** | Paper Analysis, Code Audit & Repo Scaffold | Thành & Luân (Joint) | Phase 0 (Current) | `docs/*.md`, `src/`, `tests/`, GitHub repo | All 18 docs complete; unit tests pass locally; pushed to GitHub. |
| **M1** | Kaggle Environment & Data Smoke Test | Thành & Luân (Joint) | Stage 1–2 | `notebooks/01..02`, verified dataset log | GPU detected; 1 normal + 1 anomalous image + mask loaded with zero leakage; shapes verified. |
| **M2** | Baseline Smoke Test (2 Categories) | Thành & Luân (Joint) | Stage 3 (`R00`) | 5-epoch smoke run on `carpet` (texture) + `bottle` (object) | Loss decreases monotonically; checkpoint saved & resumed; anomaly heatmaps exported. |
| **M3** | Baseline Reproduction (Representative Subset $\to$ Full Benchmark) | Thành & Luân (Joint) | Stage 4 (`R01`–`R02`) | Frozen `B0` checkpoints & baseline metric table | 160-epoch runs complete on 3 core categories (`carpet`, `bottle`, `screw`) + expansion to 15 categories as Kaggle quota permits; both `last_epoch` and `best_test` logged. |
| **M4** | Direction A — Calibrated Reliability Fusion (`A1`–`A3`) | **Thành (Researcher A)** | Stage 5A | `A1`, `A2`, `A3` evaluation tables & qualitative heatmaps | Evaluated on frozen `B0` checkpoints; zero test-set calibration leakage; per-category Pixel-AUROC, Pixel-AP, AUPRO, and small-defect analysis complete. |
| **M5** | Direction B — Adapter & GCM Compression (`B1`–`B3`) | **Luân (Researcher B)** | Stage 5B | `B1` (inference pruning), `B2` (`SepAdapter`), `B3` (`CDA-Lite`) + cost table | Exact params, FLOPs, latency (ms), and VRAM measured on same GPU; quality–efficiency Pareto curve plotted. |
| **M6** | Combined System (`C1`) & Interaction Analysis | Thành & Luân (Joint) | Stage 6 (`C1`) | `C1` (`B2/B3 + A3`) evaluation & ablation matrix | Interaction effects between structural compression and calibrated fusion quantified on same categories. |
| **M7** | Final CS331 Report, Demo & Reproducibility Audit | Thành & Luân (Joint) | Final Defense | Final PDF report, slides, clean GitHub release | Every table number traceable to a run log; clean checkout passes all tests. |

---

## Detailed Milestone Specifications

### M1 — Kaggle Environment & Data Validation (Stage 1 & Stage 2)
- **Objective:** Confirm Kaggle GPU environment (T4/P100), package compatibility, MVTec AD path resolution (`/kaggle/input/mvtec-ad`), and leak-free dataset splitting.
- **Inputs:** `scripts/check_environment.py`, `scripts/validate_dataset.py`, `configs/experiments/smoke_test.yaml`.
- **Expected Outputs:** Environment JSON report; dataset validation summary confirming `train/good` contains only normal images (`label=0`) and `test/*` ground-truth masks align spatially ($256\times256$, binary $\{0, 1\}$).
- **Acceptance Criteria:** `validate_dataset.py` exits with code 0; zero missing masks; `TestDataset` works regardless of OS path separators or `--data_path` location.
- **Estimated Difficulty:** Low (2/10).
- **Owner:** Joint (Luân leads `check_environment.py`, Thành leads `validate_dataset.py`).

### M2 — Baseline Pipeline Smoke Test (`R00`, Stage 3)
- **Objective:** Validate forward pass, loss computation, backward pass, checkpoint save/resume, evaluation metrics, and heatmap visualization on 1 texture (`carpet`) and 1 object (`bottle`) category for 2–5 epochs.
- **Inputs:** `scripts/train.py --config configs/experiments/smoke_test.yaml`.
- **Expected Outputs:** `last.pth`, `best_val.pth`, `metrics.json`, and sample heatmaps in `outputs/smoke_test/`.
- **Acceptance Criteria:** Finite loss values; checkpoint resume restores exact epoch and optimizer state; evaluation script produces valid Image-AUROC, Pixel-AUROC, Image-AP, Pixel-AP, and AUPRO.
- **Estimated Difficulty:** Low-Medium (3/10).
- **Owner:** Joint.

### M3 — Paper-Faithful CDA Baseline Reproduction (`R01`–`R02`, Stage 4)
- **Objective:** Train paper-faithful CDA (`B0`, 160 epochs, `bs=4`, `lr=0.005`, `betas=(0.5, 0.999)`) on the 3 core categories (`carpet` [texture], `bottle` [large object], `screw` [small-defect object]), then expand to additional MVTec AD categories (`tile`, `leather`, `capsule`, `pill`, `transistor`, `metal_nut`, `hazelnut`, `cable`, `grid`, `wood`, `toothbrush`, `zipper`).
- **Inputs:** `configs/baseline/cda_mvtec.yaml`.
- **Expected Outputs:** Saved checkpoints (`B0`) uploaded to persistent Kaggle Dataset / Google Drive so neither student ever has to retrain `B0`; per-category baseline table comparing `last_epoch` vs. `best_val` vs. `best_test`.
- **Acceptance Criteria:** Complete training logs (`160` epochs per category), reproducible seed (`seed=42`), and honest documentation of any gap vs. paper numbers.
- **Estimated Difficulty:** Medium (5/10 — mainly GPU wall-clock management on Kaggle).
- **Owner:** Joint (Thành runs 7–8 categories, Luân runs 7–8 categories on Kaggle).

### M4 — Direction A: Calibrated Cross-Scale Reliability Fusion (`A1`–`A3`, Stage 5A)
- **Objective:** Evaluate normal-distribution calibration (`A1`), cross-scale concordance gating (`A2`), and scale-adaptive smoothing + top-$q\%$ scoring (`A3`) on top of frozen `B0` checkpoints.
- **Inputs:** Frozen `B0` checkpoints, `configs/research_a/default.yaml`, `scripts/evaluate.py`.
- **Expected Outputs:** Quantitative ablation table (`A0` vs. `A1` vs. `A2` vs. `A3`) across all metrics (especially Pixel-AP and AUPRO) + side-by-side failure/success heatmaps on small defects and noisy textures.
- **Acceptance Criteria:** Calibration uses **only** normal training images (`train/good`); statistical significance / per-category breakdown documented, including any categories where adaptive fusion underperforms.
- **Estimated Difficulty:** Medium (5/10).
- **Owner:** **Thành (Researcher A)**.

### M5 — Direction B: Structural Adapter & GCM Compression (`B1`–`B3`, Stage 5B)
- **Objective:** Evaluate inference-time cross-scale pruning (`B1: CDA-AuxCSD`, `26.66M` params, zero retraining), retrain depthwise-separable cross-scale adapters (`B2: CDA-SepAdapter`, `28.52M` params), and retrain factorized GCM + separable adapters (`B3: CDA-Lite`, `19.46M` params).
- **Inputs:** `configs/research_b/default.yaml`, `scripts/train.py`, `scripts/benchmark_latency.py`.
- **Expected Outputs:** Quality-vs-Efficiency trade-off table and Pareto plot (Params, FLOPs/MACs, Latency ms, Peak VRAM MB vs. Image-AUROC, Pixel-AUROC, Pixel-AP, AUPRO).
- **Acceptance Criteria:** All latency and memory benchmarks executed on the **exact same Kaggle GPU model** in a single session with 50 warmup + 200 timed iterations and `torch.cuda.synchronize()`.
- **Estimated Difficulty:** Medium-High (6/10).
- **Owner:** **Luân (Researcher B)**.

### M6 — Combined Model (`C1`, Stage 6)
- **Objective:** Apply Thành's best calibrated reliability mapper (`A3`) to Luân's best compressed student (`B2` or `B3`) and evaluate interaction effects.
- **Inputs:** Trained `B2`/`B3` checkpoints + `A3` inference configuration.
- **Expected Outputs:** Final comparison table (`B0`, `A3`, `B1`, `B2`, `B3`, `C1`).
- **Acceptance Criteria:** Clear analysis of whether `A3` recovers localization precision on the compressed model `B2`/`B3`.
- **Estimated Difficulty:** Low-Medium (4/10).
- **Owner:** Joint.
