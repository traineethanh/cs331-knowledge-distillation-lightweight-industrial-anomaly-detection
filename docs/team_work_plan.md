# Team Work Plan and Responsibility Division (Thành & Luân)

**Course:** Advanced Computer Vision (CS331)  
**Shared Project Title:** *Knowledge Distillation-Based Lightweight Industrial Anomaly Detection*  
**Primary Baseline:** CDA (*Efficient Industrial Anomaly Detection via Cross-Scale Distillation with Enhanced Feature Compression*, Pattern Recognition 2026)

---

## 1. Research Ownership Allocation

Following the literature review and novelty-gap audit ([novelty_gap_analysis.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/novelty_gap_analysis.md)), the two research directions are formally assigned as follows:

### Researcher A — Thành (`feature/thanh-reliability`)
- **Sub-Project Title:** *Normal-Calibrated & Cross-Scale Reliability-Guided Anomaly Map Fusion (`A1`–`A3`)*
- **Core Scientific Ownership:**
  1. Investigate why CDA's static anomaly scoring (`2.0 * a + 0.1 * w * ac`, $\sigma=4$, `max()`) under-utilizes the `16.52M`-parameter cross-scale branches (50:1 weight ratio) and suffers from scale variance mismatch on textured surfaces.
  2. Implement and ablate **Normal-Distribution Statistical Calibration (`A1`)**, **Same-vs-Cross Scale Concordance Gating (`A2`)**, and **Scale-Adaptive Gaussian Smoothing + Top-$q\%$ Tail Scoring (`A3`)**.
  3. Conduct fine-grained error analysis across **texture vs. object categories** and **small vs. large defects** (measuring Pixel-AUROC, Pixel-AP, and $\text{AUPRO}_{0.30}/\text{AUPRO}_{0.05}$).
- **Primary Code Ownership:**
  - `src/anomaly_detection/models/research_a/`
  - `src/anomaly_detection/inference/calibration.py` & `anomaly_mapper.py` (calibrated/reliability modes)
  - `configs/research_a/`

### Researcher B — Luân (`feature/luan-efficiency`)
- **Sub-Project Title:** *Structural Bottleneck & Cross-Scale Adapter Compression for Lightweight Reverse Distillation (`B1`–`B3`)*
- **Core Scientific Ownership:**
  1. Diagnose CDA's **Cross-Scale Adapter Parameter Paradox** (`c11..c32` take `16.52M` parameters = 38.25% of student, while `GCM.res` takes `9.45M` = 21.89%, compared to only `6.96M` for the `GSU` decoder).
  2. Test the **Auxiliary Regularizer vs. Inference Predictor Hypothesis (`B1: CDA-AuxCSD`)** by pruning `c11..c32` at inference time (`-38.25%` active student parameters with zero retraining).
  3. Implement and train **Depthwise-Separable Cross-Scale Adapters (`B2: CDA-SepAdapter`, `-33.95%` params)** and **Factorized GCM + Separable Adapters (`B3: CDA-Lite`, `-54.94%` params)**.
  4. Profile exact parameter counts, MACs/FLOPs, GPU inference latency (ms/image), throughput (FPS), and peak VRAM across all variants on the same Kaggle GPU.
- **Primary Code Ownership:**
  - `src/anomaly_detection/models/research_b/`
  - `scripts/benchmark_latency.py` & `src/anomaly_detection/utils/profiling.py`
  - `configs/research_b/`

---

## 2. Shared Responsibilities (Both Thành and Luân)

Both researchers must understand the complete CDA pipeline and be able to explain every equation and code module during the oral defense:

| Shared Area | Joint Tasks |
| :--- | :--- |
| **1. Baseline Reproduction (`R00`–`R02`)** | Both students run and verify the bug-fixed CDA baseline (`B0`/`A0`), divide MVTec AD categories on Kaggle if needed, and archive verified baseline checkpoints (`last.pth`, `best_val.pth`, `best_test.pth`) to shared cloud storage so both branches evaluate against the exact same baseline checkpoints. |
| **2. Core Pipeline & Unit Tests** | Co-maintain `src/anomaly_detection/data/`, `models/baseline/`, `losses/`, `evaluation/metrics.py`, and `tests/`. Any change to shared files requires a Pull Request reviewed by the other partner. |
| **3. Combined Experiment (`C1`)** | Jointly run and analyze Stage 6 (`C1`: combining Luân's best compressed architecture `B2`/`B3` with Thành's best calibrated reliability mapper `A3`). |
| **4. Final Report & Presentation** | Co-author Introduction, Related Work, Baseline Reconstruction, Combined Results, Limitations, and Reproducibility Appendix. |

---

## 3. Clean Software Interfaces Between Research Branches

To prevent merge conflicts and confounding variables, the two branches interact through two clean Python interfaces:

1. **Model Output Contract (`GCMAndDecoder` / `LiteGCMAndDecoder`):**
   - Input: Teacher features `(e1, e2, e3)` with shapes `((B,256,64,64), (B,512,32,32), (B,1024,16,16))`.
   - Output: Dictionary or 9-tuple `(s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n)` (where `l1m..l3n` may be `None` only when `infer_same_scale_only=True` in `B1`).
2. **Anomaly Mapper Contract (`AnomalyMapper`):**
   - Input: `e_list = [e1, e2, e3]`, `d_list = [s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n]`, `img_size = 256`.
   - Output: `(pixel_anomaly_map, image_anomaly_score)` with shapes `((B, H, W), (B,))`.
   - Because Luân modifies **how `d_list` is parameterized** and Thành modifies **how `(e_list, d_list)` are calibrated and fused into `(pixel_anomaly_map, image_anomaly_score)`**, the two improvements are 100% orthogonal in code!

---

## 4. Shared Files Requiring Coordination & Git Workflow

### Protected Shared Files (Require PR + Partner Review Before Merging)
- `src/anomaly_detection/data/mvtec.py` (dataset split, resize, normalization, mask binarization)
- `src/anomaly_detection/models/baseline/cda.py` (reference baseline architecture)
- `src/anomaly_detection/losses/cda_loss.py` (baseline loss calculation)
- `src/anomaly_detection/evaluation/metrics.py` (Image/Pixel AUROC, AP, AUPRO calculation)
- `configs/baseline/cda_mvtec.yaml`

### Git Branching Strategy
- `main`: Stable, tested, reviewed releases and verified documentation.
- `develop`: Integration branch where `feature/thanh-reliability` and `feature/luan-efficiency` merge after passing unit tests (`python -m unittest discover -s tests`).
- `feature/thanh-reliability`: Thành's working branch for `A1`, `A2`, `A3` and localization analysis.
- `feature/luan-efficiency`: Luân's working branch for `B1`, `B2`, `B3` and latency/memory profiling.

---

## 5. Rules for Fair Combination of Results (`C1`)

1. **No Cherry-Picking Categories:** Report results on the exact same set of categories for `B0`, `A1..A3`, `B1..B3`, and `C1`.
2. **Reuse Checkpoints Whenever Valid:**
   - `A0`, `A1`, `A2`, `A3`, and `B1` (`CDA-AuxCSD`) **all evaluate on the exact same trained `B0` checkpoint** per category! This guarantees zero training-seed variance between those 5 rows of our experiment table.
   - `C1` (`B2/B3 + A3`) evaluates directly on the trained `B2`/`B3` checkpoint without retraining!
3. **Same-Session Hardware Profiling:** Run `scripts/benchmark_latency.py` for `B0`, `A3`, `B1`, `B2`, `B3`, and `C1` inside a single Kaggle GPU session so all latency (ms) and VRAM (MB) numbers share identical hardware and thermal conditions.
