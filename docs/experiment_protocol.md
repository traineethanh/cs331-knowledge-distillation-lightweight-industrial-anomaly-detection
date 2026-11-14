# Experiment Protocol

## 1. Dataset & Data-Leakage Prevention
- **Dataset:** MVTec AD (15 industrial categories: 10 object, 5 texture).
- **Unsupervised One-Class Rule:** Training uses defect-free images (`train/good`) exclusively. `MVTecTrainDataset` in `src/anomaly_detection/data/mvtec.py` enforces this check and raises an error if non-`good` images appear in `train/`.
- **Deterministic Validation Split:** `train/good` is deterministically split 80/20 (`val_split_ratio: 0.2`, `seed: 42`) for validation/calibration without touching `test/`.
- **Zero Test-Set Leakage:** Test images and ground-truth masks are used strictly for final evaluation.

## 2. Evaluation Metrics
Implemented and unit-tested in `src/anomaly_detection/evaluation/metrics.py`:
- **Image-level Detection:** `auroc_img`, `ap_img`
- **Pixel-level Localization:** `auroc_pix`, `ap_pix`, `aupro_30` (FPR <= 0.30), `aupro_05` (FPR <= 0.05)
- **Efficiency Profiling (`src/anomaly_detection/utils/profiling.py`):** Parameter count, FLOPs/MACs, inference latency, and peak GPU memory.

## 3. Execution Environment (Kaggle GPU + GitHub)
- **Training & Evaluation:** Executed on Kaggle GPU sessions with MVTec AD attached as a read-only input dataset (`/kaggle/input/mvtec-ad`).
- **Environment Verification:** Run `python scripts/check_environment.py` and `python scripts/validate_dataset.py` before training.
- **Artifact Hygiene:** Large checkpoints (`*.pth`, `*.pt`), raw datasets, and Kaggle temporary files are excluded by `.gitignore` and must never be committed to GitHub. Only verified lightweight summaries belong in `results/`.
- **Experiment Status:** `PENDING` (no experiments have been executed for the new project).
