# CDA Baseline Specification for Reproducible Execution

This specification defines the exact architectural, mathematical, and training configuration for our primary baseline (**CDA**, Hu et al., *Pattern Recognition* 2026), clearly separating settings across four provenance tiers:
1. **Paper-Reported Configuration**
2. **Official Repository Defaults (`WHUer-cloud/CDA`)**
3. **Our Reproducibility Adaptations (Bug Fixes & Leak-Free Protocol)**
4. **Temporary Smoke-Test Settings**

---

## 1. Four-Way Configuration Comparison Matrix

| Parameter / Component | 1. Paper-Reported | 2. Official Repo Default (`config.py` / `train.py`) | 3. Our Reproducibility Baseline (`configs/baseline/cda_mvtec.yaml`) | 4. Smoke-Test Config (`configs/experiments/smoke_test.yaml`) |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher Backbone** | Pretrained WideResNet-50-2 (frozen) | `wide_resnet50_2(weights=DEFAULT).eval()` (`24.86M` params) | **Identical** (`wide_resnet50_2`, frozen `.eval()`) | **Identical** (supports `pretrained: false` for offline unit tests) |
| **Teacher Layers Extracted** | Layers 1, 2, 3 | `layer1` $(256,64,64)$, `layer2` $(512,32,32)$, `layer3` $(1024,16,16)$ | **Identical** | **Identical** |
| **Bottleneck Module (`GCM`)** | Global-local Compression Module (`CLE` + `SGA`) | `GCM` (`19,704,579` params): `local1`, `local2`, `merge`, `Fusion` (RoPE), `res` | **Identical** (dynamic spatial size `h, w` instead of hard-coded `16, 16`) | **Identical** |
| **Student Decoder (`GSU`)** | 3-stage `PatchExpand2D` + `GSULayer` | `Decoder` (`6,959,616` params): `l3(1024)`, `l2(512)`, `l1(256)` | **Identical** (standard PyTorch device management instead of global import-time `device`) | **Identical** |
| **Cross-Scale Adapters** | All-to-all scale alignment | `c11..c32`: 6 dense $3\times3$ convs (`16,518,656` params) | **Identical** (`adapter_mode: "dense"`) | **Identical** |
| **Input Resolution** | $256 \times 256$ | `--img_size 256` (`Resize((256, 256))`) | $256 \times 256$ | $256 \times 256$ (or $128 \times 128$ for fast CPU shape test) |
| **Image Normalization** | ImageNet mean/std | `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]` | **Identical** | **Identical** |
| **Optimizer** | Adam | `Adam(lr=0.005, betas=(0.5, 0.999))` | **Identical** (`lr=0.005, betas=[0.5, 0.999]`) | **Identical** |
| **Batch Size** | 4 | `--bs 4`, `drop_last=True`, `shuffle=False` | `bs=4`, `drop_last=True`, configurable `shuffle: true/false` (default `false` for exact repo match) | `bs=2` |
| **Epochs** | 160 | `--epochs 160` | `160` | `2` (for pipeline verification only) |
| **Loss Function** | Weighted cosine distance ($\mathcal{L}_1, \mathcal{L}_2$) | `2.0 * loss1 + 0.1 * loss2` with `(0.4*l_kn + l_km)` in `train.py` | **Identical** (`cross_loss_mode: "official_train"`; also supports `"symmetric_metric"` for ablation) | **Identical** |
| **Anomaly Map Smoothing** | Gaussian filter $\sigma=4$ | `gaussian_filter(score, sigma=4)` | **Identical** ($\sigma=4.0$) | **Identical** |
| **Checkpoint Saving & Selection** | Unspecified in abstract | Saves only `dec.state_dict()` on test-set improvements (`best_img.pth`, `best_pix.pth`) | Saves full state (`model`, `optimizer`, `epoch`, `seed`, `config`) to `last.pth` (primary unbiased checkpoint) + `best_test.pth` (for official repo comparison) | Saves `last.pth` at epoch 2 |

---

## 2. Exact Tensor Shapes and Layer Specifications (`img_size = 256`)

1. **Input Batch:** `image` $\in \mathbb{R}^{B \times 3 \times 256 \times 256}$.
2. **Teacher Outputs (`Encoder`):**
   - $e_1 \in \mathbb{R}^{B \times 256 \times 64 \times 64}$
   - $e_2 \in \mathbb{R}^{B \times 512 \times 32 \times 32}$
   - $e_3 \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$
3. **GCM Internal Representations:**
   - `CLE`: $l_1 \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$, $l_2 \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$, $l \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$.
   - `SGA` (`Fusion`): $p_1, p_2, x_3 \in \mathbb{R}^{B \times 256 \times 1}$, 6 RoPE attention maps $\in \mathbb{R}^{B \times 256 \times 16 \times 16}$, concatenated $\in \mathbb{R}^{B \times 1536 \times 16 \times 16}$, `DoubleConv` output $g \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$.
   - `res`: $f = \text{res}(g + l) \in \mathbb{R}^{B \times 2048 \times 8 \times 8}$.
4. **Decoder & Cross-Scale Outputs (`GCMAndDecoder`):**
   - Same-scale: $s_3 \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$, $s_2 \in \mathbb{R}^{B \times 512 \times 32 \times 32}$, $s_1 \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
   - Cross-scale aligned to $e_1$: $l_{1m}, l_{1n} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
   - Cross-scale aligned to $e_2$: $l_{2m}, l_{2n} \in \mathbb{R}^{B \times 512 \times 32 \times 32}$.
   - Cross-scale aligned to $e_3$: $l_{3m}, l_{3n} \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$.

---

## 3. Acceptance Criteria for Baseline Reproduction

Before proceeding from Stage 4 (Baseline Reproduction) to Stage 5 (Improvement Experiments), the baseline must pass all 5 gates:
1. **Structural Gate:** Parameter counts match the official repository (`24,862,528` frozen teacher parameters; `43,182,851` trainable student parameters).
2. **Numerical Gate:** Loss decreases smoothly without `NaN`/`Inf` over 160 epochs; anomaly maps have shape $(N_{\text{test}}, 256, 256)$ with values bounded in $[0, \infty)$.
3. **Leak-Free Protocol Gate:** Zero test images or ground-truth masks are accessed during training or normal-statistic calibration; final evaluation reports `last.pth` (epoch 160) alongside `best_test.pth` (official repo style).
4. **Artifact Persistence Gate:** Every run exports `config.yaml`, `train_log.csv`, `metrics_summary.json`, `last.pth`, and qualitative anomaly heatmaps before Kaggle session termination.
5. **Performance Gate:** On the initial representative categories (`carpet`, `bottle`, `screw`), `B0` achieves strong detection and localization (e.g., Image-AUROC $> 94\%$, Pixel-AUROC $> 95\%$), providing a valid anchor for `A1..A3` and `B1..B3`.
