# Master Experiment Matrix and Run Logging Specification

## 1. Master Experiment Matrix

Every experiment run in our project is assigned a unique, deterministic `run_id` defined in the table below:

| Run ID | Branch / Owner | Base Checkpoint | Single Controlled Change | Trainable / Active Student Params | Retrain Required? | Primary Metrics Evaluated | Completion Gate |
| :--- | :--- | :--- | :--- | ---: | :---: | :--- | :--- |
| **`R00`** | Baseline (Joint) | From scratch (2 epochs) | Pipeline smoke test on `carpet` + `bottle` | `43.18M` / `43.18M` | Yes (2 ep) | Loss finiteness, shapes, save/resume, metrics | Clean exit & heatmap export |
| **`B0` (`A0`)** | Baseline (Joint) | From scratch (160 epochs) | Bug-fixed official CDA baseline (`cda_mvtec.yaml`) | `43.18M` / `43.18M` | Yes (160 ep) | Image/Pixel AUROC & AP, $\text{AUPRO}_{0.30/0.05}$, Params, FLOPs, Latency | Checkpoints archived for Tier 1 $\to$ Tier 3 |
| **`A1`** | Direction A (Thành) | Frozen `B0` (`last.pth`) | `+NormalCalibration`: Z-score / Robust IQR normalization of $a_k, ac_k$ on normal train set | `43.18M` / `43.18M` | **No** | Pixel-AP, $\text{AUPRO}_{0.30/0.05}$, Pixel-AUROC, Image-AUROC | Zero test leakage verified |
| **`A2`** | Direction A (Thành) | Frozen `B0` (`last.pth`) | `+ReliabilityGate`: Multiplicative concordance $\hat{a}_k(1 + \eta\tanh(\hat{a}_k \odot \hat{ac}_k))$ | `43.18M` / `43.18M` | **No** | False-positive reduction on textures & small-defect recall | Ablation vs. `A0` & `A1` |
| **`A3`** | Direction A (Thành) | Frozen `B0` (`last.pth`) | `+ScaleSmooth+TopQ`: Scale-specific $\sigma=(2,3,4)$ + Top-$0.5\%$ image score pooling | `43.18M` / `43.18M` | **No** | Image-AUROC/AP stability + sharp small-defect localization | Full per-category error analysis |
| **`B1`** | Direction B (Luân) | Frozen `B0` (`last.pth`) | `CDA-AuxCSD`: Prune `c11..c32` (`16.52M`) at inference time (`infer_same_scale_only=True`) | `43.18M` / **`26.66M`** (`-38.3%` params, `-74.7%` FLOPs) | **No** | Active Params, FLOPs (`15.17G`), Latency (ms) vs. Accuracy retention | Tests regularizer vs. predictor role of CSD |
| **`B2`** | Direction B (Luân) | From scratch (160 epochs) | `CDA-SepAdapter`: Replace dense $3\times3$ `c11..c32` (`16.52M`) with Depthwise-Separable (`1.88M`) | **`28.54M` / `28.54M`** (`-33.9%` params, `-66.1%` FLOPs) | Yes (160 ep) | Params (`-14.64M`), FLOPs (`20.27G`), Latency, VRAM, all 5 quality metrics | Compare vs. `B0` and `B1` on same GPU |
| **`B3`** | Direction B (Luân) | From scratch (160 epochs) | `CDA-Lite`: `B2` + Depthwise-Separable `ConvBlock`s in `GCM` (`local1, local2, res`) | **`21.22M` / `21.22M`** (`-50.9%` params, `-69.7%` FLOPs) | Yes (160 ep) | Pareto frontier curve (Cost vs. Image/Pixel/Small-defect metrics) | Identify first metric to degrade |
| **`C1`** | Combined (Joint) | Frozen `B2` or `B3` (`last.pth`) | Apply Thành's `A3` Calibrated Reliability Mapper to Luân's best compressed student (`B2`/`B3`) | **`28.54M` or `21.22M`** | **No** | All 5 quality metrics + all 4 efficiency metrics | Quantify synergy between A and B |

---

## 2. Mandatory Metadata Schema for Every Run (`metrics_summary.json`)

Every execution of `scripts/train.py` and `scripts/evaluate.py` automatically writes a structured JSON and CSV log containing:

```json
{
  "run_id": "B0_mvtec_carpet_seed42",
  "timestamp_utc": "2026-10-23T11:30:00Z",
  "git_commit": "<auto-detected-hash>",
  "git_branch": "main",
  "seed": 42,
  "dataset": "mvtec_ad",
  "category": "carpet",
  "config_path": "configs/baseline/cda_mvtec.yaml",
  "hardware": {
    "device": "cuda",
    "gpu_name": "Tesla T4",
    "pytorch_version": "2.11.0"
  },
  "model_cost": {
    "teacher_params": 24862528,
    "student_trainable_params": 43182851,
    "student_active_infer_params": 43182851,
    "student_macs_g": 18.42,
    "latency_forward_ms": null,
    "latency_e2e_ms": null,
    "peak_vram_mb": null
  },
  "metrics_last_epoch": {
    "epoch": 160,
    "auroc_img": null,
    "ap_img": null,
    "auroc_pix": null,
    "ap_pix": null,
    "aupro_30": null,
    "aupro_05": null
  },
  "metrics_best_test_reference": {
    "best_img_epoch": null,
    "auroc_img": null,
    "best_pix_epoch": null,
    "auroc_pix": null
  }
}
```
*(Note: `null` placeholders above illustrate the schema only; actual values are populated strictly from real execution runs.)*

---

## 3. Research Log Entry Template (For `experiments/README.md`)

After every Kaggle session, the student running the experiment appends a signed entry:

```markdown
### Run ID: `<run_id>`
- **Researcher:** Thành / Luân
- **Date & Commit:** `YYYY-MM-DD` | `<git_commit_hash>` | Branch: `<branch>`
- **Hardware:** Kaggle GPU (`Tesla T4` / `P100`)
- **Hypothesis Tested:** <1-sentence falsifiable hypothesis>
- **Single Controlled Change:** <exact config key changed vs. reference run>
- **Quantitative Outcome (`last.pth`):**
  - Image-AUROC / AP: `...` / `...`
  - Pixel-AUROC / AP / AUPRO@0.30 / AUPRO@0.05: `...` / `...` / `...` / `...`
  - Params / Latency (ms) / Peak VRAM (MB): `...` / `...` / `...`
- **Qualitative Observation & Decision:** <Where did it improve or fail? Keep, refine, or reject?>
```
