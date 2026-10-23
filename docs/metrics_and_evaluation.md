# Metrics and Evaluation Specification

This document defines the mathematical formulation, implementation details, and reporting standards for all **anomaly detection**, **pixel localization**, and **computational efficiency** metrics used in our repository (`src/anomaly_detection/evaluation/metrics.py` and `scripts/benchmark_latency.py`).

---

## 1. Anomaly Detection & Localization Quality Metrics

Let $\mathcal{D}_{\text{test}} = \{(I_n, y_n, G_n)\}_{n=1}^{N}$ be the test dataset for a given category, where:
- $y_n \in \{0, 1\}$ is the ground-truth image-level label ($0$: normal/`good`, $1$: anomalous).
- $G_n \in \{0, 1\}^{H \times W}$ is the binarized ground-truth pixel mask ($G_n \equiv 0$ when $y_n = 0$; resized to $H \times W$ and thresholded at $0.5$).
- $\mathcal{M}_n \in \mathbb{R}^{H \times W}$ is the predicted pixel-level anomaly map, and $S_n \in \mathbb{R}$ is the predicted image-level anomaly score.

### 1.1 Image-Level AUROC (`auroc_img`)
- **Definition:** Area Under the Receiver Operating Characteristic curve computed over image-level pairs $\{(S_n, y_n)\}_{n=1}^{N}$ using `sklearn.metrics.roc_auc_score(labels, img_scores)`.
- **Baseline Scoring Rule (`A0`/`B0`):** $S_n = \max_{h,w} \mathcal{M}_n(h, w)$.
- **Direction A Scoring Rule (`A3`):** Top-$q\%$ tail mean $S_n = \frac{1}{K}\sum_{i=1}^{K} \mathcal{M}_{n,(i)}$ where $K = \max(1, \lfloor q H W \rfloor)$.
- **What It Measures:** Global ranking quality of anomalous vs. normal images across all thresholds.

### 1.2 Image-Level Average Precision (`ap_img`)
- **Definition:** Area under the Precision–Recall curve over $\{(S_n, y_n)\}_{n=1}^{N}$ using `sklearn.metrics.average_precision_score(labels, img_scores)`.
- **Why Included:** More informative than Image-AUROC when normal and anomalous test counts are imbalanced.

### 1.3 Pixel-Level AUROC (`auroc_pix`)
- **Definition:** Area under the ROC curve computed over all flattened pixels $\bigcup_{n=1}^{N} \{(\mathcal{M}_n(h,w), G_n(h,w))\}_{h,w}$ using `sklearn.metrics.roc_auc_score`.
- **Known Limitation:** Because $>95\%$ of pixels in anomalous images are normal background ($G_n(h,w)=0$), a model can achieve $>97\%$ Pixel-AUROC even if it produces blurred false positives or misses tiny defects! Therefore, Pixel-AUROC **must** be accompanied by **Pixel-AP** and **AUPRO**.

### 1.4 Pixel-Level Average Precision (`ap_pix`)
- **Definition:** Area under the pixel-level Precision–Recall curve (`sklearn.metrics.average_precision_score(gts.ravel(), scores.ravel())`).
- **Why Essential for Our Research:** Highly sensitive to false-positive background clutter on textures (`carpet`, `grid`, `tile`) and precision of defect boundaries. Directly tests Thành's Direction A hypothesis.

### 1.5 Per-Region Overlap (`aupro_30` and `aupro_05`)
- **Definition:** Unlike Pixel-AUROC (which weights every pixel equally, allowing one large defect to dominate ten tiny defects), **PRO (Per-Region Overlap)** weights every connected ground-truth defect component $C_{n,k}$ equally!
  For a binarization threshold $\tau$:
  $$\text{PRO}(\tau) = \frac{1}{K_{\text{total}}} \sum_{n=1}^{N} \sum_{k=1}^{K_n} \frac{|\{(h,w) \in C_{n,k} : \mathcal{M}_n(h,w) > \tau\}|}{|C_{n,k}|}$$
  where $C_{n,k}$ are the 8-connected components (`scipy.ndimage.label`) of ground-truth mask $G_n$.
- **Normalized Integration (AUPRO):** Integrate $\text{PRO}(\tau)$ against the False Positive Rate (FPR) on normal background pixels up to an upper FPR limit $\alpha \in \{0.30, 0.05\}$, normalized by $\alpha$:
  $$\text{AUPRO}_{\alpha} = \frac{1}{\alpha} \int_{0}^{\alpha} \text{PRO}(\text{FPR}) \, d(\text{FPR})$$
- **Why Essential:** Standard MVTec AD uses $\alpha = 0.30$ (`aupro_30`), while MVTec AD 2 (2025) recommends $\alpha = 0.05$ (`aupro_05`) to strictly penalize false positives and reward small-defect recall. Our `metrics.py` implements both!

---

## 2. Computational Cost and Efficiency Metrics

Measured via `src/anomaly_detection/utils/profiling.py` and `scripts/benchmark_latency.py`:

### 2.1 Parameter Count (`params_total`, `params_student_train`, `params_student_infer`)
- Reported separately for:
  1. **Frozen Teacher Encoder (`Encoder`):** `24,862,528` (`24.86M`).
  2. **Trainable Student (`GCM + Decoder + Adapters`):** `43,182,851` (`43.18M` for `B0`, `28.52M` for `B2`, `19.46M` for `B3`).
  3. **Active Inference Student:** For `B1` (`CDA-AuxCSD`), `c11..c32` (`16.52M`) are skipped during inference, reducing active student parameters to `26,664,195` (`26.66M`).

### 2.2 Multiply-Accumulate Operations / FLOPs (`macs_G`, `flops_G`)
- Measured deterministically using `torch.utils.flop_counter.FlopCounterMode` (built into PyTorch $\ge 2.1$, zero external dependency bugs!) at fixed input resolution `(1, 3, 256, 256)`.
- Reported separately for **Teacher Encoder**, **Student (`GCM + Decoder + Adapters`)**, and **Total Pipeline**.

### 2.3 Inference Latency (`latency_ms_mean`, `latency_ms_std`, `latency_ms_p95`) & Throughput (`fps`)
- **Mandatory Timing Protocol (`scripts/benchmark_latency.py`):**
  1. Record exact hardware string (`torch.cuda.get_device_name(0)` or CPU model), PyTorch version, CUDA version, input resolution `(1, 3, 256, 256)` (and batch size `4`), warmup iterations (`50`), and measured iterations (`200`).
  2. Call `torch.cuda.synchronize()` before and after each timed step when CUDA is available.
  3. Report **two timing breakdowns**:
     - **Model Forward Time (`forward_ms`):** `Encoder(img)` + `Student(e1, e2, e3)` on GPU.
     - **End-to-End Inference Time (`e2e_ms`):** Model Forward + Anomaly Map Fusion + Gaussian Smoothing + Image Score Aggregation.

### 2.4 Peak GPU Memory (`peak_vram_mb`)
- Measured via `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated() / (1024 ** 2)` during both training (`bs=4`) and inference (`bs=1`).
