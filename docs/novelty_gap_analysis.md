# Novelty and Research-Gap Analysis

## 1. Purpose of This Assessment

In an undergraduate research project built on a 2026 baseline (**CDA**, Hu et al., *Pattern Recognition* 2026), the greatest scientific pitfall is **accidental rediscovery** — claiming a mechanism is novel when it is already present inside CDA or recent 2022–2026 literature. This document establishes strict boundaries between what is **already published**, what is **code-verified inside CDA**, and where **genuine, defensible research gaps** exist for Thành (Researcher A) and Luân (Researcher B).

---

## 2. What Cannot Be Claimed as Novel (Red-Line Claims)

| Prohibited Claim | Why It Is Invalid (Prior Art) |
| :--- | :--- |
| *"Using multi-scale feature pyramid matching for anomaly detection"* | Already established in **STFPM** (BMVC 2021) and **RD4AD** (CVPR 2022). |
| *"Transferring knowledge across different scales (Cross-Scale Distillation)"* | Already the central contribution of our baseline **CDA** (*Pattern Recognition* 2026, `c11..c32` in `model.py`). |
| *"Combining global attention and local convolutions in the bottleneck"* | Already implemented in **CDA's `GCM`** (`CLE` ConvBlocks + `SGA` RoPE SelfAttention in `model.py` and `Transformer.py`). |
| *"Using a Mamba-inspired / gated sequence decoder (`GSU`) for lightweight reconstruction"* | Already implemented in **MambaAD** (NeurIPS 2024) and **CDA's `GSU` Decoder** (`model.py` lines 42–66). |
| *"Simply reducing student channels uniformly and claiming a novel architecture"* | Generic width scaling is an engineering ablation, not an architectural insight — unless tied to a structural bottleneck analysis of CDA. |

---

## 3. Genuine Scientific & Architectural Gaps Identified in CDA

By auditing the mathematics and tensor-level parameter distribution of `WHUer-cloud/CDA`, we uncovered two concrete, previously unaddressed weaknesses in CDA:

### Gap 1 (Localization & Calibration Gap — Target for Researcher A / Thành)
**Observation in Official CDA (`metric.py` lines 18–37, 47):**
1. **Uncalibrated Raw Sum Across Scales and Pathways:**
   At test time, CDA computes 3 same-scale maps ($a_1, a_2, a_3$) and 3 cross-scale maps ($ac_1, ac_2, ac_3$), upsamples them to $256\times256$, and combines them with fixed constants:
   $$\mathcal{M}_{\text{raw}} = 2.0(a_1 + a_2 + a_3) + 0.1(0.4 ac_1 + 0.2 ac_2 + 0.4 ac_3)$$
   However, shallow $64\times64$ features ($e_1 \in \mathbb{R}^{256\times64\times64}$) exhibit higher baseline cosine distance and spatial variance on normal textured surfaces (e.g., `carpet`, `tile`, `wood`, `grid`) than deep $16\times16$ features ($e_3$). Because $a_1, a_2, a_3, ac_1, ac_2, ac_3$ are not standardized against their normal training distribution ($\mu_k^{\text{norm}}, \sigma_k^{\text{norm}}$), noisy shallow residuals dominate background regions.
2. **Under-Utilization of Cross-Scale Agreement:**
   CDA computes both same-scale discrepancy $a_k(h,w)$ and cross-scale discrepancy $ac_k(h,w)$, but merely adds $ac_k$ with a tiny weight ($0.1 \times 0.4 = 0.04$ vs. $2.0$ for $a_k$ — a **50:1 ratio**!). Because $0.04 \ll 2.0$, $ac_k$ has almost negligible arithmetic impact in the linear sum, wasting the `16.52M` parameters spent computing `l1m..l3n`! Instead of adding $ac_k$ with weight $0.04$, cross-scale agreement between $a_k$ and $ac_k$ (and across scales $k \in \{1,2,3\}$) can act as a **spatial reliability / consistency gate** that suppresses single-scale background false positives while preserving high-confidence localized defects.
3. **Single-Pixel `max()` Fragility (`metric.py` line 47):**
   CDA computes image-level anomaly score as $S_{\text{img}} = \max_{h,w} \mathcal{M}(h,w)$, which depends on a single pixel and is sensitive to isolated border or specular spikes.

### Gap 2 (Parameter & Computational Bottleneck Gap — Target for Researcher B / Luân)
**Observation in Official CDA (`model.py` lines 122–184):**
1. **The Cross-Scale Alignment Parameter Paradox (`16.52M` params = 38.25% of Student):**
   The authors designed an ultra-lightweight `GSU` Decoder (`6.96M` params total across `l3`, `l2`, `l1`), but then attached **six dense $3\times3$ convolutions (`c11, c12, c21, c22, c31, c32`)** to align $(s_1, s_2, s_3)$ across scales. Those six adapters alone contain **`16,518,656` parameters** — **$2.37\times$ the size of the entire GSU decoder**!
   - Specifically, `c11..c32` operate on already bilinearly interpolated feature maps where spatial context was already mixed by `GSULayer`. Using dense $3\times3$ kernels ($9 C_{\text{in}} C_{\text{out}}$ parameters) instead of **Depthwise-Separable ($3\times3$ DW + $1\times1$ PW)** or **Low-Rank / Bottleneck $1\times1$ adapters** inflates adapter parameters by **$8.9\times$ to $16\times$**!
2. **The `GCM.res` Downsampling Bottleneck (`9.45M` params = 21.89% of Student):**
   Inside `GCM`, `self.res = ConvBlock(1024, 3, [512, 1024, 2048], 2)` uses a dense $3\times3$ convolution ($512 \times 1024 \times 9 = 4.72\text{M}$) plus two $1\times1$ projections ($1024\times512 + 1024\times2048 = 2.62\text{M}$) and a dense $1\times1$ shortcut ($1024\times2048 = 2.10\text{M}$), totaling `9.45M` parameters just to downsample $16\times16 \to 8\times8$.
3. **Train-Only vs. Inference-Active Cross-Scale Branches:**
   Because cross-scale maps $ac_k$ have weight $\le 0.04$ in `metric.py`, an immediate scientific question arises: *Are `c11..c32` primarily regularizers during training, or are they needed during inference? And if kept at inference, can factorized cross-scale adapters + lightweight GCM compression cut total student parameters by 40%–55% while preserving or improving localization?*

---

## 4. Summary of Defensible Undergraduate Contributions

Rather than making sweeping claims of "new SOTA", our project frames two rigorous, falsifiable, and complementary scientific investigations on top of CDA:
1. **Direction A (Thành):** *Normal-Calibrated & Cross-Scale Reliability-Guided Anomaly Localization* — fixing CDA's 50:1 under-weighted linear fusion and single-pixel `max()` scoring using training-set quantile/z-score calibration, cross-scale consistency gating, and top-$q$ spatial pooling.
2. **Direction B (Luân):** *Structural Bottleneck Compression of Cross-Scale Adapters and GCM* — resolving CDA's `16.52M` cross-scale adapter paradox and `9.45M` `GCM.res` overhead via depthwise-separable / low-rank cross-scale alignment and inference-stage cross-scale pruning ablations.
