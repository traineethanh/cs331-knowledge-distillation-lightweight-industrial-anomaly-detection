# Candidate Research Directions and Critical Evaluation

In accordance with **Work Package 4**, we formulate and critically evaluate **four technically distinct candidate research directions** on top of the verified **CDA** baseline (*Pattern Recognition* 2026, `43.18M` student parameters), covering every required scientific, mathematical, and operational dimension before recommending two complementary directions for Thành (Researcher A) and Luân (Researcher B).

---

## Candidate A — Normal-Calibrated & Cross-Scale Reliability-Guided Anomaly Map Fusion (Localization Focus)

### 1. Research Question
Can normalizing scale-wise discrepancy maps against defect-free training statistics and weighting spatial locations by **same-scale vs. cross-scale agreement (reliability gating)** reduce background false positives and improve small-defect pixel localization (Pixel-AUROC, Pixel-AP, AUPRO) compared to CDA's static linear fusion, without adding trainable parameters or significant inference latency?

### 2. Motivation and Evidence from Literature
- **EfficientAD (WACV 2024)** showed that heterogeneous anomaly maps have incompatible raw numerical ranges on normal images and require normal-validation quantile calibration before fusion.
- **MVTec AD 2 (2025)** demonstrated that fixed Gaussian smoothing ($\sigma=4$) and uncalibrated map summation degrade localization of tiny defects and increase false alarms under normal surface texture variation (`carpet`, `tile`, `grid`, `screw`).

### 3. Exact Limitation of the CDA Baseline Targeted
In `metric.py` (line 32), CDA fuses its 6 upsampled cosine distance maps ($\tilde{a}_1, \tilde{a}_2, \tilde{a}_3$ same-scale; $\tilde{ac}_1, \tilde{ac}_2, \tilde{ac}_3$ cross-scale) via:
$$\mathcal{M}_{\text{CDA}} = G_{\sigma=4} * \Big(2.0(\tilde{a}_1 + \tilde{a}_2 + \tilde{a}_3) + 0.04\,\tilde{ac}_1 + 0.02\,\tilde{ac}_2 + 0.04\,\tilde{ac}_3\Big)$$
This has three concrete flaws:
1. **50:1 Weight Imbalance:** $\tilde{ac}_k$ receives weight $0.04$ or $0.02$ vs. $2.0$ for $\tilde{a}_k$, rendering the `16.52M`-parameter cross-scale predictions nearly invisible at inference.
2. **Scale Variance Mismatch:** $\tilde{a}_1$ ($64\times64$) has higher mean and variance on normal textures than $\tilde{a}_3$ ($16\times16$), causing shallow texture noise to dominate the sum.
3. **Outlier Fragility of `s.max()`:** Image-level scoring uses the single highest pixel (`s.max()`), which triggers false alarms on isolated noisy pixels.

### 4. Proposed Mechanism
A three-stage **Training-Free / Post-Hoc Calibrated Reliability Fusion (CRF)** pipeline operating on the frozen CDA checkpoint:
1. **Stage A1 — Normal-Distribution Z-Score / Robust Quantile Calibration:** Compute spatial mean $\mu_m^{\text{norm}}$ and standard deviation $\sigma_m^{\text{norm}}$ (or median and IQR $[q_{50}, q_{99.5}-q_{50}]$) for each of the 6 maps $m \in \{a_1, a_2, a_3, ac_1, ac_2, ac_3\}$ over a held-out subset of normal training images (zero test-set leakage). Standardize each map:
   $$\hat{m}(h, w) = \max\left(0,\, \frac{\tilde{m}(h, w) - \mu_m^{\text{norm}}}{\sigma_m^{\text{norm}} + \epsilon}\right)$$
2. **Stage A2 — Cross-Scale Reliability Gating & Persistence-Preserving Fusion:** At each scale $k \in \{1, 2, 3\}$, measure concordance between same-scale residual $\hat{a}_k$ and cross-scale residual $\hat{ac}_k$, combined with a scale-persistence term that preserves high-confidence tiny defects visible at fine scale $k=1$:
   $$R_k(h, w) = \sigma_{\text{gate}}\big(\gamma (\hat{a}_k(h, w) \cdot \hat{ac}_k(h, w))\big), \quad \mathcal{M}_{\text{CRF}}(h, w) = \sum_{k=1}^{3} \beta_k \hat{a}_k(h, w) \big(1 + \eta R_k(h, w)\big)$$
   Followed by **scale-aware smoothing** ($\sigma_1 = 2.0$ for $k=1$ to preserve sharp small defects, $\sigma_3 = 4.0$ for coarse scale $k=3$).
3. **Stage A3 — Robust Top-$q\%$ Tail Aggregation for Image Scoring:** Replace `s.max()` with the mean of the top-$q\%$ pixels ($q \in \{0.1\%, 0.5\%, 1.0\%\}$) of $\mathcal{M}_{\text{CRF}}$.

### 5. Mathematical Formulation / Pseudocode
```python
# Calibration on normal training split (computed once after training):
# mu_k, std_k = mean_and_std_over_normal_loader(a_k, ac_k)

def calibrated_reliability_score(a_list, ac_list, stats, eta=0.5, top_q=0.005):
    fused = 0.0
    sigmas = [2.0, 3.0, 4.0]  # scale-specific Gaussian smoothing
    for k in range(3):
        a_hat = torch.clamp((a_list[k] - stats.mu_a[k]) / (stats.std_a[k] + 1e-6), min=0.0)
        ac_hat = torch.clamp((ac_list[k] - stats.mu_ac[k]) / (stats.std_ac[k] + 1e-6), min=0.0)
        # Cross-scale agreement reliability gate
        rel_k = torch.tanh(a_hat * ac_hat)
        map_k = gaussian_blur_2d(a_hat * (1.0 + eta * rel_k), sigma=sigmas[k])
        fused = fused + map_k
    # Image score via top-q% tail mean instead of single-pixel max
    flat = fused.flatten(1)
    k_pixels = max(1, int(flat.shape[1] * top_q))
    img_score = torch.topk(flat, k=k_pixels, dim=1).values.mean(dim=1)
    return fused, img_score
```

### 6. Expected Benefits and Possible Failure Modes
- **Benefits:** Zero retraining cost (reuses the exact baseline checkpoint!); suppresses normal texture clutter on `carpet`, `grid`, `tile`; sharpens boundaries of small defects (`screw`, `capsule`, `pill`) by avoiding uniform $\sigma=4$ blur on $a_1$; improves Pixel-AP and AUPRO.
- **Failure Modes:** If a defect is extremely large (covering $>50\%$ of the image), top-$0.1\%$ and `max()` behave similarly; if normal training images have lighting drift relative to test images, z-score stats $(\mu_k, \sigma_k)$ may shift.

### 7. Similarity to Existing Published Methods
- Shares the concept of normal-set statistical calibration with **EfficientAD (WACV 2024)**, but differs by formulating **intra-scale same-vs-cross distillation concordance ($a_k \odot ac_k$)** and **scale-adaptive smoothing** specifically for Cross-Scale Reverse Distillation.

### 8. Estimated Implementation Difficulty
- **Low-to-Medium (3/10):** Cleanly localized in `src/anomaly_detection/inference/` and `models/research_a/`; requires no architectural retraining for A1–A3.

### 9. Expected GPU Cost
- **Very Low:** Calibration over ~250 normal training images takes $<15$ seconds per category on a single Kaggle T4/P100 GPU; inference overhead is $<0.5\text{ ms}$ per image.

### 10. Required Experiments and Ablations
- `A0`: Official CDA baseline formula ($\sigma=4$, `max()`).
- `A1`: Normal Z-score / Quantile calibration only (`+Calibration`).
- `A2`: `A1` + Cross-scale reliability gating (`+ReliabilityGate`).
- `A3`: `A2` + Scale-adaptive Gaussian smoothing ($\sigma \in \{2,3,4\}$) + Top-$q\%$ image scoring.

### 11. Criteria for Accepting or Rejecting the Hypothesis
- **Accept if:** `A3` improves mean Pixel-AP or AUPRO (or reduces false-positive rate at 95% TPR) across tested MVTec AD categories without degrading Image-AUROC by $>0.3\%$ or increasing latency by $>5\%$.
- **Reject if:** Calibration or reliability gating degrades small-defect recall on object categories (`screw`, `capsule`, `pill`).

### 12. Statement of What Cannot Be Claimed as Novel
- We **cannot** claim multi-scale anomaly map fusion or quantile normalization itself is newly invented; our contribution is an empirical and mathematical study of **normal-calibrated cross-scale concordance fusion** addressing the 50:1 cross-scale weight imbalance in CDA.

---

## Candidate B — Input-Adaptive Early-Exit / Conditional Cross-Scale Inference (Dynamic Compute Focus)

### 1. Research Question
Can an input-conditional early-exit gate skip the cross-scale alignment convolutions (`c11..c32`, `16.52M` params) or shallow decoder stage (`l1`) on "obviously normal" test images while maintaining baseline detection and localization accuracy?

### 2. Motivation and Evidence from Literature
- In industrial inspection lines, $>95\%$ of inspected parts are defect-free. Dynamic neural networks aim to spend less compute on easy normal samples and full compute only on suspicious samples.

### 3. Exact Limitation of the CDA Baseline Targeted
- CDA executes all 3 decoder stages (`l3, l2, l1`) and all 6 cross-scale alignment convolutions (`c11..c32`) for every test image, regardless of whether coarse scale $a_3$ ($16\times16$) already indicates a defect-free part.

### 4. Proposed Mechanism
- Evaluate coarse discrepancy $S_3 = \max D_{\cos}(e_3, s_3)$ at resolution $16\times16$. If $S_3 < \tau_{\text{exit}}$ (calibrated at the $p$-th percentile of normal training images), skip `c11..c32` (or skip `l1`) and output a fast coarse map; otherwise run the full pipeline.

### 5. Mathematical Formulation / Pseudocode
$$\mathcal{M}(I) = \begin{cases} G_{\sigma=4} * (2.0 \sum_{k=1}^3 \tilde{a}_k) & \text{if } \max_{h,w} a_1(h,w) < \tau \\ \mathcal{M}_{\text{CDA}}(I) & \text{otherwise} \end{cases}$$

### 6. Expected Benefits and Possible Failure Modes
- **Benefits:** Saves compute of `c11..c32` on normal samples.
- **Severe Failure Mode (Critical Flaw!):** Small surface scratches (`screw`, `pill`, `capsule`, `wood`) are **invisible at coarse scale $e_3$ ($16\times16$)** and only appear in $e_1$ ($64\times64$)! Therefore, gating on $a_3$ causes catastrophic false negatives on small defects. Conversely, if the gate waits until $s_1$ ($64\times64$) is computed, `GCM` and `Decoder` (`l3, l2, l1`) have *already* been executed, and only `c11..c32` remain — which can be eliminated *statically* (see Candidate C) without dynamic branching overhead! Furthermore, dynamic data-dependent branching breaks batched GPU execution and introduces threshold sensitivity.

### 7. Similarity to Existing Published Methods
- Standard early-exit thresholding in dynamic inference literature.

### 8. Estimated Implementation Difficulty
- **Medium (5/10):** Requires threshold calibration per category and unbatched (`bs=1`) CUDA timing.

### 9. Expected GPU Cost
- Low training cost, but messy evaluation across varying exit rates $\tau$.

### 10. Required Experiments and Ablations
- Sweep exit threshold $\tau$ across quantiles $\{0.5, 0.7, 0.9, 0.95\}$ and plot accuracy vs. average latency.

### 11. Criteria for Accepting or Rejecting the Hypothesis
- Reject if early exit misses small defects or if static adapter compression (Candidate C) achieves superior speedups on *all* images (both normal and anomalous) without branching.

### 12. Statement of What Cannot Be Claimed as Novel
- Threshold-based early exit is a standard engineering heuristic.

---

## Candidate C — Structural Factorization of Cross-Scale Adapters & Lightweight GCM Bottleneck (Efficiency & Compression Focus)

### 1. Research Question
Given that CDA's six dense $3\times3$ cross-scale alignment convolutions (`c11..c32`) consume **`16.52M` parameters (38.25% of the student)** and `GCM.res` consumes **`9.45M` parameters (21.89%)** — together accounting for **60.14% of all trainable parameters** while the GSU Decoder itself is only `6.96M` — how much can we compress the student network via **Depthwise-Separable / Low-Rank Cross-Scale Adapters** and **Bottleneck-Factorized GCM** before image detection, pixel localization, or small-defect sensitivity begins to degrade?

### 2. Motivation and Evidence from Literature
- **CDA (Pattern Recognition 2026)** motivates its GSU decoder as a lightweight alternative to RD4AD's ResNet decoder, reducing the decoder to `6.96M` parameters. Yet our code audit ([official_code_audit.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/official_code_audit.md)) revealed a striking architectural imbalance:
  - The 6 cross-scale alignment layers `c11..c32` (`nn.Conv2d(C_in, C_out, 3, 1, 1)`) take **16,518,656 parameters (`16.52M`)** — $2.37\times$ larger than the GSU decoder!
  - Furthermore, at inference (`metric.py`), the outputs of `c11..c32` ($ac_1, ac_2, ac_3$) are weighted by only $0.1 \times (0.4, 0.2, 0.4) = (0.04, 0.02, 0.04)$ against $2.0$ for same-scale maps!
- This raises two fundamental, high-impact scientific questions that no prior paper has asked about CDA:
  1. **Role of Cross-Scale Branches (Auxiliary Regularizer vs. Inference Predictor):** If `c11..c32` are used during training to regularize the decoder $s_1, s_2, s_3$ (`Train-CSD`) but pruned at inference (`Infer-SameScaleOnly`), does CDA retain $99.9\%$ of its accuracy while saving **16.52M parameters (38.3%)** and all 6 cross-scale interpolations/convolutions at test time?
  2. **Factorized Cross-Scale Alignment & GCM Compression (`LiteCDA`):** When cross-scale alignment is retained during both training and inference, replacing dense $3\times3$ convolutions in `c11..c32` with **Depthwise-Separable Convolutions (`DWConv3x3 + PWConv1x1`)** reduces adapter parameters from **`16.52M` to `1.86M` (an 88.7% reduction in adapter parameters!)**, and factorizing `GCM.res` + `CLE` reduces total trainable parameters from **`43.18M` down to `26.66M` (B1) and `16.80M` (B2, a 61.1% total parameter reduction!)**.

### 3. Exact Limitation of the CDA Baseline Targeted
1. **Over-parameterized Cross-Scale Alignment (`c11..c32` = `16.52M` params):** Dense $3\times3$ convolutions are applied *after* `PatchExpand2D` + `GSULayer` and bilinear interpolation have already mixed spatial neighborhood information.
2. **Over-parameterized `GCM.res` (`9.45M` params):** Dense $3\times3$ conv ($512 \to 1024$, `4.72M` params) + dense $1\times1$ shortcut ($1024 \to 2048$, `2.10M` params) at $8\times8$ resolution.

### 4. Proposed Mechanism (Progressive Compression Hierarchy `B1` $\to$ `B2` $\to$ `B3`)
We design three controlled, mathematically clean efficiency variants:
- **Variant `B1` — Inference-Stage Cross-Scale Pruning (`CDA-AuxCSD`, Zero Retraining):**
  - Keep `c11..c32` during training as an auxiliary cross-scale distillation regularizer on the GSU decoder, but **prune `c11..c32` at inference time**, computing anomaly scores purely from $(s_1, s_2, s_3)$.
  - **Active Inference Student Parameters:** Drops from **`43.18M` to `26.66M` (-38.25%)** with **zero retraining required** (uses the exact `B0` checkpoint!).
- **Variant `B2` — Depthwise-Separable Cross-Scale Alignment (`CDA-SepAdapter`, Retrained):**
  - Replace each dense $3\times3$ adapter $c_{ij}: \mathbb{R}^{C_{\text{in}} \times H_i \times W_i} \to \mathbb{R}^{C_{\text{out}} \times H_i \times W_i}$ (`9 * C_in * C_out` params) with a **Depthwise-Separable Alignment Block**:
    $$\tilde{c}_{ij}(X) = \text{Conv}_{1\times1}^{C_{\text{in}} \to C_{\text{out}}}\big(\text{GELU}(\text{BN}(\text{DWConv}_{3\times3, \text{groups}=C_{\text{in}}}(X)))\big)$$
  - Adapter parameter count drops from **`16,518,656` (`16.52M`) to `1,859,328` (`1.86M`)** — saving **`14.66M` parameters (-88.7% adapter params, -33.9% total model params)** while keeping full cross-scale capabilities at both training and inference!
- **Variant `B3` — Factorized GCM + Separable Adapters (`CDA-Lite`, Retrained Pareto Variant):**
  - Combine `B2` (`SepAdapter`) with **Depthwise-Separable `ConvBlock`s inside `GCM` (`local1`, `local2`, `res`)**, replacing the dense $3\times3$ inner convolution $\text{Conv}_{3\times3}(f_1 \to f_2)$ with $\text{DWConv}_{3\times3}(f_1) + \text{Conv}_{1\times1}(f_1 \to f_2)$.
  - Total trainable student parameters drop from **`43.18M` to `19.46M` (-54.9% total parameters!)** while preserving the exact teacher interface, bottleneck dimensions, and GSU decoder.

### 5. Mathematical Formulation / Parameter Verification
For a cross-scale adapter mapping $C_{\text{in}} \to C_{\text{out}}$ with spatial kernel $k=3$:
- **Baseline Dense Adapter (`model.py: c11..c32`):**
  $$P_{\text{dense}} = k^2 C_{\text{in}} C_{\text{out}} + C_{\text{out}} = 9 C_{\text{in}} C_{\text{out}} + C_{\text{out}} \implies \sum_{i \neq j} P_{\text{dense}}^{(i,j)} = 16,518,656$$
- **Proposed Depthwise-Separable Adapter (`B2`):**
  $$P_{\text{sep}} = k^2 C_{\text{in}} + 2 C_{\text{in}} + C_{\text{in}} C_{\text{out}} + C_{\text{out}} \approx C_{\text{in}} C_{\text{out}} \implies \sum_{i \neq j} P_{\text{sep}}^{(i,j)} = 1,859,328 \quad (-88.74\%)$$

### 6. Expected Benefits and Possible Failure Modes
- **Benefits:**
  - `B1` (`CDA-AuxCSD`) requires **zero training GPU time** (evaluates directly on `B0` checkpoint) and immediately tests whether `c11..c32` are needed at test time or act mainly as training-time regularizers.
  - `B2` and `B3` drastically reduce parameter count (-33.9% to -54.9%), checkpoint disk size (from ~173 MB to ~78 MB), FLOPs, and GPU memory while maintaining the full cross-scale architecture.
- **Possible Failure Modes:** In `B3`, compressing `GCM.res` alongside `c11..c32` may slightly reduce reconstruction fidelity on complex object categories (`transistor`, `cable`, `metal_nut`). Tracking where degradation appears first (Image-AUROC vs. Pixel-AP vs. small defects) is itself a core scientific contribution.

### 7. Similarity to Existing Published Methods
- Depthwise-separable convolutions (MobileNet/ConvNeXt) and auxiliary training heads are established building blocks, **not** new inventions. However, **identifying and resolving the 16.52M cross-scale adapter parameter paradox in CDA** and isolating the **Training-Regularizer vs. Inference-Predictor role of Cross-Scale Distillation** is a novel, sharp, and scientifically rigorous analysis specific to CDA.

### 8. Estimated Implementation Difficulty
- **Low-to-Medium (4/10):** Modular drop-in replacement of `c11..c32` and `ConvBlock` inside `src/anomaly_detection/models/research_b/`, controlled via YAML config (`adapter_type: "dense" | "separable" | "train_only"`).

### 9. Expected GPU Cost
- `B1`: **Zero training cost** (inference-only on `B0` checkpoints).
- `B2`, `B3`: Same or ~25% faster per-epoch training time than `B0` on 3–5 representative categories (or full MVTec AD if budget allows).

### 10. Required Experiments and Ablations
- `B0`: Official CDA (`43.18M` student params).
- `B1`: `CDA-AuxCSD` (`26.66M` active inference params; `c11..c32` pruned at test time).
- `B2`: `CDA-SepAdapter` (`28.52M` params; depthwise-separable `c11..c32`).
- `B3`: `CDA-Lite` (`19.46M` params; separable `c11..c32` + separable `GCM` blocks).

### 11. Criteria for Accepting or Rejecting the Hypothesis
- **Accept if:** `B1` or `B2` reduces active student parameters and FLOPs by $>30\%$ while keeping Image-AUROC and Pixel-AUROC within $\le 0.5\%$ of `B0`, or if the Pareto curve clearly identifies which defect types degrade first under `B3`.

### 12. Statement of What Cannot Be Claimed as Novel
- We do **not** claim depthwise-separable convolutions are novel; we claim a **structural bottleneck diagnosis and Pareto efficiency optimization of Cross-Scale Reverse Distillation**.

---

## Candidate D — Frequency-Domain / Wavelet Bottleneck Enhancement in GCM (Alternative Evaluated & Rejected)

- **Concept:** Add a Discrete Wavelet Transform (DWT) / Fast Fourier Transform (FFT) branch inside `GCM` to preserve high-frequency edges.
- **Why Rejected:** CDA's `GCM` already has two parallel branches (`CLE` + `SGA`) totaling `19.70M` parameters. Adding a third frequency branch increases model size and latency, directly conflicting with the project's **Lightweight** title, and confounds with Researcher A's localization experiments by requiring full retraining for every frequency weight change.

---

## Reasoned Comparison Table Across Candidate Directions

| Evaluation Criterion | **Candidate A:** Calibrated Cross-Scale Reliability Fusion | **Candidate B:** Input-Adaptive Early-Exit Inference | **Candidate C:** Structural Adapter Factorization & Lite GCM | **Candidate D:** Frequency/Wavelet GCM Branch |
| :--- | :--- | :--- | :--- | :--- |
| **1. Scientific Justification** | **Strong (9/10):** Directly solves CDA's 50:1 cross-scale weight imbalance & uncalibrated scale variance. | **Weak (4/10):** Coarse $16\times16$ gate misses small defects that only appear at $64\times64$. | **Very Strong (9.5/10):** Directly resolves our discovery that `c11..c32` take `16.52M` params ($2.37\times$ the GSU decoder). | **Moderate (6/10):** Plausible for textures, but redundant with `CLE`. |
| **2. Differentiation from CDA** | **High:** CDA uses static unnormalized sum + $\sigma=4$ + `max()`. | **High:** CDA is static. | **High:** CDA uses dense $3\times3$ adapters (`16.52M`) and dense `GCM.res` (`9.45M`). | **Low-Moderate:** Just adds a 3rd branch to `GCM`. |
| **3. Differentiation from Prior Art** | Adapts normal-set calibration specifically to **same-vs-cross scale agreement ($a_k \odot ac_k$)**. | Standard dynamic thresholding. | First structural audit separating **train-time vs. test-time cross-scale adapters** and factorizing CSD adapters. | Many FFT/Wavelet AD papers exist in 2024–2025. |
| **4. Feasibility for 2 Undergraduates** | **Excellent (9.5/10):** Clean math, modular code, fast iteration. | **Moderate (6/10):** Messy batch=1 branching logic. | **Excellent (9/10):** Clean PyTorch module replacements & exact parameter/FLOP profiling. | **Low-Moderate (5/10):** Complex tensor padding/wavelet tuning. |
| **5. Kaggle GPU Compatibility** | **Ideal (10/10):** Reuses `B0` checkpoints with zero retraining! | **Good (7/10):** Requires threshold sweeps. | **Very Good (9/10):** `B1` needs 0 retraining; `B2/B3` train faster and use less VRAM than `B0`. | **Poor (4/10):** Slower training, higher VRAM, full retraining per ablation. |
| **6. Computational Overhead** | **Negligible ($<0.5\text{ ms}$/img)**; `0` extra parameters. | Saves compute only on subset of images; adds gate overhead. | **Reduces parameters by 33.9%–54.9%** and lowers FLOPs/latency on *all* images. | **Increases** parameters and latency. |
| **7. Ease of Fair Ablations** | **Ideal:** Evaluated on the exact same checkpoints and test splits. | Harder to compare at fixed operating points. | **Ideal:** Stepwise `B0 -> B1 -> B2 -> B3` isolates adapter pruning vs. adapter factorization vs. GCM compression. | Confounds bottleneck capacity with frequency priors. |
| **8. Risk of Negative Results** | **Low-Moderate:** Even if reliability gating helps only textures/small defects, per-category analysis is highly informative. | **High:** High risk of missing small defects on `screw`/`pill`. | **Very Low:** Parameter/FLOP reductions are guaranteed; quality trade-off curve is scientifically valuable regardless of slope. | **High:** May overfit normal high-frequency noise. |
| **9. Suitability for CS331 Report** | **Outstanding** | **Weak** | **Outstanding** | **Moderate** |

---

## Final Recommendation: Two Complementary, Non-Overlapping Research Directions

We recommend pairing **Candidate A (assigned to Thành — Researcher A)** and **Candidate C (assigned to Luân — Researcher B)**:

1. **Why They Are Strictly Complementary (Zero Overlap):**
   - **Thành (Direction A — Localization & Calibration)** operates on the **output discrepancy maps and scoring rules** ($\{a_k, ac_k\} \to \mathcal{M}_{\text{CRF}}, S_{\text{img}}$), keeping the neural network weights fixed (`B0` checkpoint) to maximize localization accuracy, small-defect preservation, and false-positive suppression.
   - **Luân (Direction B / Candidate C — Structural Efficiency & Adapter Compression)** operates on the **internal neural network parameterization** (`c11..c32` cross-scale adapters and `GCM` bottleneck blocks) while keeping the loss and scoring formula fixed to characterize the Pareto frontier between student capacity (`43.18M` $\to$ `28.52M` $\to$ `19.46M`), latency, and detection/localization quality.
2. **Why They Combine Cleanly in Stage 6 (`C1`):**
   - Because Direction A modifies the `AnomalyMapper` (inference/calibration module) and Direction B modifies `GCMAndDecoder` (architecture module), combining the best architecture (`B2` or `B3`) with the best calibrated mapper (`A3`) requires zero code conflict and directly tests whether calibrated reliability fusion recovers any minor localization drop introduced by model compression!
