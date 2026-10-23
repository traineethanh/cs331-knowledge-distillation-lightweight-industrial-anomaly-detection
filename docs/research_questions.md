# Formal Research Questions, Hypotheses, and Ablation Designs

## 1. Shared Baseline Research Question (RQ0 — Both Researchers)

- **RQ0 (Reproducibility & Protocol Integrity):**
  *Can we faithfully reproduce the CDA architecture (`Encoder` + `GCM` + `GSU Decoder` + Cross-Scale Alignment) on MVTec AD within Kaggle GPU constraints, and how much of the official repository's reported performance depends on test-set checkpoint selection (`best_img.pth` / `best_pix.pth` evaluated every 5 epochs) versus a leak-free fixed/last-epoch evaluation protocol?*
- **Hypothesis H0:**
  Fixing the software bugs in `WHUer-cloud/CDA` will yield strong Image-AUROC ($>95\%$) and Pixel-AUROC ($>96\%$) on representative texture and object categories, while comparing `best_test` vs. `last_epoch` will reveal a measurable optimistic bias ($0.3\%–1.5\%$ AUROC) in the official repository's test-set selection rule.

---

## 2. Researcher A (Thành) — Calibrated Cross-Scale Reliability Fusion

### Primary Research Question (RQ-A)
*When multi-scale same-scale ($a_1, a_2, a_3$) and cross-scale ($ac_1, ac_2, ac_3$) discrepancy maps exhibit mismatched background variance and local noise, does **Normal-Distribution Calibration** combined with **Cross-Scale Concordance Gating ($a_k \odot ac_k$)** and **Scale-Adaptive Smoothing** improve pixel-level anomaly localization (Pixel-AP, AUPRO, Pixel-AUROC) and image-level robustness without retraining the model?*

### Sub-Questions & Testable Hypotheses
1. **RQ-A1 (Normal Statistical Calibration):**
   - *Question:* Does standardizing each of the 6 discrepancy maps using normal training set statistics ($\mu_m^{\text{norm}}, \sigma_m^{\text{norm}}$) outperform CDA's unnormalized weighted sum (`2.0 * a + 0.1 * w * ac`)?
   - *Hypothesis H-A1:* Normal Z-score/quantile calibration will prevent high-variance shallow residuals ($a_1$ at $64\times64$) from dominating normal textured backgrounds (`carpet`, `tile`, `wood`), improving Pixel-AP and AUPRO.
2. **RQ-A2 (Cross-Scale Concordance / Reliability Gating):**
   - *Question:* Can multiplicative agreement between same-scale error $\hat{a}_k(h,w)$ and cross-scale error $\hat{ac}_k(h,w)$ suppress isolated single-scale false positives while retaining true small defects?
   - *Hypothesis H-A2:* Multiplicative reliability gating $\hat{a}_k (1 + \eta \tanh(\hat{a}_k \odot \hat{ac}_k))$ utilizes the `16.52M`-parameter cross-scale predictions far more effectively than CDA's $0.04$ linear weight, reducing false positives at low FPR (improving $\text{AUPRO}_{0.30}$ and $\text{AUPRO}_{0.05}$).
3. **RQ-A3 (Scale-Adaptive Smoothing & Top-$q\%$ Aggregation):**
   - *Question:* Does replacing uniform $\sigma=4$ Gaussian smoothing and single-pixel `max()` pooling with scale-adaptive smoothing ($\sigma_1=2, \sigma_2=3, \sigma_3=4$) and top-$q\%$ pixel tail pooling improve small-defect localization and image-level stability?
   - *Hypothesis H-A3:* Lighter smoothing on scale 1 ($\sigma_1=2$) preserves sharp boundaries on tiny defects (`screw`, `capsule`, `pill`), while top-$q\%$ ($q=0.5\%$) image scoring reduces sensitivity to single-pixel background spikes.

### Controlled Ablation Sequence (All Evaluated on Frozen `B0` Checkpoints)
| Variant ID | Normal Calibration | Cross-Scale Fusion Rule | Spatial Smoothing | Image Score Pooling |
| :--- | :--- | :--- | :--- | :--- |
| **`A0` (CDA Baseline)** | None (Raw `cos_dist`) | Linear: $2.0\sum a_k + 0.1(0.4ac_1+0.2ac_2+0.4ac_3)$ | Fixed $\sigma=4.0$ after sum | Single-pixel `max()` |
| **`A1` (`+NormCalib`)** | Z-score / Robust IQR on normal train split | Equal calibrated sum $\sum (\hat{a}_k + \alpha_k \hat{ac}_k)$ | Fixed $\sigma=4.0$ after sum | Single-pixel `max()` |
| **`A2` (`+RelGate`)** | Z-score / Robust IQR on normal train split | Concordance gate: $\sum \hat{a}_k(1 + \eta\tanh(\hat{a}_k \hat{ac}_k))$ | Fixed $\sigma=4.0$ after sum | Single-pixel `max()` |
| **`A3` (`+ScaleSmooth+TopQ`)** | Z-score / Robust IQR on normal train split | Concordance gate: $\sum \hat{a}_k(1 + \eta\tanh(\hat{a}_k \hat{ac}_k))$ | Scale-adaptive $\sigma=(2.0, 3.0, 4.0)$ | Top-$q\%$ tail mean ($q=0.5\%$) |

---

## 3. Researcher B (Luân) — Structural Bottleneck & Cross-Scale Adapter Compression

### Primary Research Question (RQ-B)
*Why do the six cross-scale channel-alignment convolutions (`c11..c32`) and bottleneck downsampler (`GCM.res`) consume **60.14% (`25.97M` / `43.18M`)** of CDA's trainable student parameters, and how does progressively compressing or pruning these modules affect the Pareto trade-off between model cost (parameters, FLOPs, latency, VRAM) and anomaly detection/localization performance?*

### Sub-Questions & Testable Hypotheses
1. **RQ-B1 (Auxiliary Regularizer vs. Inference Predictor Hypothesis):**
   - *Question:* Are the `16.52M`-parameter cross-scale alignment convolutions (`c11..c32`) actively needed during inference, or does their primary benefit come from regularizing the GSU decoder $(s_1, s_2, s_3)$ during training?
   - *Hypothesis H-B1:* Because CDA weights $ac_k$ by only $0.02–0.04$ vs. $2.0$ for $a_k$ at inference, pruning `c11..c32` at test time (`B1: CDA-AuxCSD`) will cut active student parameters by **38.25% (`43.18M` $\to$ `26.66M`)** and reduce inference latency by ~15–20% while changing Image/Pixel AUROC by $<0.2\%$.
2. **RQ-B2 (Depthwise-Separable Cross-Scale Adapter Factorization):**
   - *Question:* Can replacing dense $3\times3$ convolutions in `c11..c32` (`16.52M` params) with Depthwise-Separable Alignment blocks (`DWConv3x3 + BN + GELU + PWConv1x1`, `1.86M` params) preserve full cross-scale distillation during both training and inference?
   - *Hypothesis H-B2:* Because `s1, s2, s3` have already undergone spatial mixing inside `GSULayer`, factorizing `c11..c32` into depthwise-separable projections (`B2: CDA-SepAdapter`) reduces adapter parameters by **88.7% (`16.52M` $\to$ `1.86M`)** and total student parameters by **33.9% (`43.18M` $\to$ `28.52M`)** with negligible accuracy loss ($\le 0.3\%$ AUROC).
3. **RQ-B3 (Deep GCM + Adapter Compression Degradation Analysis):**
   - *Question:* When we further factorize the internal `ConvBlock`s of `GCM` (`local1`, `local2`, `res`) using depthwise-separable $3\times3$ stages (`B3: CDA-Lite`, reducing total student parameters by **54.9% to `19.46M`**), which performance metric degrades first: image-level detection, coarse pixel localization, or fine small-defect localization?
   - *Hypothesis H-B3:* Image-AUROC and large structural defect localization will remain robust under `B3`, whereas any capacity-induced degradation will appear first in Pixel-AP / AUPRO on fine-grained categories.

### Controlled Ablation Sequence (Fixed Loss & Fixed Evaluation Protocol)
| Variant ID | Cross-Scale Adapters (`c11..c32`) | `GCM` (`local1, local2, res`) | Student Params (Train / Infer) | Student FLOPs (`256x256`) | Param / FLOPs Reduction vs. `B0` | Requires Retraining? |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **`B0` (CDA Baseline)** | Dense $3\times3$ (`16.52M`) | Dense `ConvBlock` (`19.70M`) | `43.18M` / `43.18M` | `59.86 G` | `0.0%` / `0.0%` | Baseline run |
| **`B1` (`CDA-AuxCSD`)** | Active in Train, **Pruned in Infer** (`0M` infer) | Dense `ConvBlock` (`19.70M`) | `43.18M` / **`26.66M`** | **`15.17 G`** | **`-38.25%` / `-74.66%` (Infer)** | **No (Uses `B0` ckpt!)** |
| **`B2` (`CDA-SepAdapter`)** | Depthwise-Separable $3\times3+1\times1$ (**`1.88M`**) | Dense `ConvBlock` (`19.70M`) | **`28.54M` / `28.54M`** | **`20.27 G`** | **`-33.90%` / `-66.13%` (Both)** | Yes |
| **`B3` (`CDA-Lite`)** | Depthwise-Separable $3\times3+1\times1$ (**`1.88M`**) | Separable `ConvBlock` (**`12.38M`**) | **`21.22M` / `21.22M`** | **`18.13 G`** | **`-50.87%` / `-69.71%` (Both)** | Yes |

---

## 4. Combined System Research Question (RQ-C — Joint Stage 6)

- **RQ-C (Synergy Between Structural Compression and Calibrated Reliability Fusion):**
  *Does applying Thành's Calibrated Reliability Fusion (`A3`) on top of Luân's compressed `CDA-SepAdapter` (`B2`) or `CDA-Lite` (`B3`) compensate for any minor localization loss caused by parameter compression, yielding a model that is simultaneously smaller, faster, and more reliable in pixel localization than the unmodified `43.18M` CDA baseline?*
