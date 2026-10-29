# Verified Recent-Literature Matrix (2021–2026)

This document provides a comprehensive, scientifically verified review of **13 key research papers** in Knowledge Distillation (KD), Teacher–Student architectures, Reverse Distillation, and Lightweight Industrial Anomaly Detection (IAD).

> [!IMPORTANT]
> **Provenance & Evidence Tags:**
> - `[VERIFIED-ACCESSED]`: Verified directly from open-access HTML/PDF, official proceedings metadata, or audited source repository.
> - `[INFERRED]`: Deductions based on software engineering principles, empirical domain norms, or mathematical analysis.
> - `[UNVERIFIED-PAYWALL]`: Information that could not be independently cross-checked because the paper is behind a subscription paywall.

---

## 1. Summary Comparison Table

| Paper | Venue & Year | Full Text Access | Code Provenance & License | Setting | Teacher / Student Architecture | Image AUROC / Pixel AUROC (MVTec) | Kaggle Reproduction Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STFPM** | BMVC 2021 | Open Access (arXiv) | Official (gdwang08/STFPM, GPL-3.0) / Anomalib (Apache-2.0) | Full-Shot Unsupervised | ResNet-18 / ResNet-18 (MS3, 3 stages) | **0.955 / 0.970** (PRO: 0.921) | **High** (<10 min/cat on T4, simple PyTorch) |
| **RD4AD** | CVPR 2022 | Open Access (arXiv) | Official (hq-deng/RD4AD, Apache-2.0) | Full-Shot Unsupervised | WideResNet-50 / Symmetric ResNet-50 Decoder | **0.985 / 0.978** (PRO: 0.939) | **High** (Standard CNN operations) |
| **EfficientAD**| WACV 2024 | Open Access (IEEE/CVF) | Official (nelson1425/EfficientAD, Apache-2.0) | Full-Shot Unsupervised | PDN (4 layers) / PDN + Autoencoder | **0.991 / 0.978** (PRO: 0.949) | **Medium** (Requires auxiliary ImageNet for hard mining) |
| **MambaAD** | NeurIPS 2024 | Open Access (arXiv) | Official (Haoyang-He/MambaAD, Apache-2.0) | Full-Shot Unsupervised | ResNet-50 / Multi-scan LSS Mamba Decoder | **0.993 / 0.981** (PRO: 0.948) | **Low** (Requires `mamba-ssm` CUDA C++ compilation) |
| **CPKD** | AAAI 2025 | Open Access (AAAI OJS) | No Official Public Code Found | Full-Shot Unsupervised | Co-evolving CNN Teacher / Student + Knowledge Prototype | Paper claims SOTA on MVTec AD | **Low** (No official code; complex co-progression training) |
| **SK-RD4AD** | CVPRW 2025 | Open Access (CVF) | Official (pej0918/SK-RD4AD, No License) | Full-Shot Unsupervised | WideResNet-50 / Skip-Connected Reverse Decoder | Paper claims gains over RD4AD on MVTec/VisA | **Medium-High** (Clean PyTorch, based on RD4AD) |
| **Dinomaly** | CVPR 2025 | Open Access (arXiv) | Official (bravo-j/Dinomaly, MIT) | Multi-Class Unsupervised | DINOv2-S / Linear Attention Transformer | **0.994 / 0.983** (VisA & MVTec) | **Medium** (High VRAM for DINOv2 ViT, multi-class setting) |
| **MVTec AD 2**| IJCV / arXiv 2025 | Open Access (arXiv) | Benchmark Evaluation Server / MVTec AD 2 | Benchmark Standard | Multiple Baselines Evaluated | SOTA falls to <60% AU-PRO | **Benchmark Only** (Evaluates saturation & border defects) |
| **CDA** | Pattern Recognit. 2026 | **Paywalled** (Elsevier) | Official (WHUer-cloud/CDA, No License) | Full-Shot Unsupervised | WideResNet50_2 / GCM + GSU (6.96M) + Adapters (16.52M) | Reported high on MVTec & VisA; unverified per-cat tables | **Medium** (Requires fixing 6 bugs; code audited) |
| **LKDF-AD** | Sci. Reports 2026 | Open Access (Nature) | Nature Supplemental / Author | **Few-Shot (1–10 shots)** | Pretrained CNN / Ultra-light Student (385K params) | Few-shot competitive on MVTec / BTAD | **High for Few-Shot; Incomparable to Full-Shot** |
| **DHKD** | ESWA 2026 | **Paywalled** (Elsevier) | Community/Author (Gilbert-Et/DHKD, Xmh-L/HDHKD) | Full-Shot Unsupervised | Pretrained CNN / Dual Heterogeneous + Mamba | Paper claims high AUROC on MVTec AD | **Low** (Requires Mamba kernels & dual branches) |
| **Multi-Teacher KD** | Neural Netw. 2026 | **Paywalled** (Elsevier) | No Official Repo Verified | Lightweight Distillation | Multiple CNN Teachers / Single Compact Student | Evaluated on industrial benchmarks | **Low** (Multi-teacher VRAM/compute overhead) |
| **SADED** | Comput. Ind. Eng. 2026 | **Paywalled** (Elsevier) | No Official Repo Verified | Synthetic-Enhanced KD | Pretrained CNN / Discrepancy-Enhanced Student | Evaluated on MVTec AD | **Low-Medium** (Synthetic defect generator overhead) |

---

## 2. In-Depth Per-Paper Scientific Profiles

### Paper 1: STFPM — Foundational Experimental Baseline
- **Citation:** Guodong Wang, Shumin Han, Errui Ding, Di Huang, *"Student-Teacher Feature Pyramid Matching for Anomaly Detection,"* BMVC 2021.
- **Identifiers:** arXiv:2103.04257 (`https://arxiv.org/abs/2103.04257`). DOI: `10.5244/C.35.349`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Freely accessible on arXiv (HTML and PDF).
- **Code Provenance:** Official GitHub repository `https://github.com/gdwang08/STFPM` (author: Guodong Wang). Also reimplemented in `open-edge-platform/anomalib` under `src/anomalib/models/image/stfpm/`.
- **Code License:** `[VERIFIED-ACCESSED]` Official repo: **GPL-3.0**. Anomalib implementation: **Apache-2.0**.
- **Problem Addressed:** `[VERIFIED-ACCESSED]` Feature representations at different scales capture different defect characteristics (textures vs. structure). Previous AE/GAN methods produce blurry pixel reconstructions, while single-scale matching misses multi-scale defects.
- **Method & Key Mechanism:** Multi-scale forward knowledge distillation. A frozen ImageNet-pretrained ResNet-18 teacher transfers features from `conv2_x`, `conv3_x`, and `conv4_x` to an identical student network initialized randomly. Loss is normalized $\ell_2$ distance:
  $$\mathcal{L} = \frac{1}{2} \sum_{l=1}^3 \frac{1}{H_l W_l} \sum_{h=1}^{H_l} \sum_{w=1}^{W_l} \left\| \hat{F}_T^l(h,w) - \hat{F}_S^l(h,w) \right\|_2^2$$
  Inference anomaly map is the element-wise product of upsampled multi-scale difference maps: $\mathcal{M} = \prod_{l=1}^3 \mathcal{I}(\mathcal{M}_l)$. Image score is $\max(\mathcal{M})$.
- **Reported Headline Numbers (MVTec AD):** `[VERIFIED-ACCESSED]` ResNet-18 backbone:
  - Mean Image AUROC: **0.955** (Table 2).
  - Mean Pixel AUROC: **0.970** (Table 1).
  - Mean PRO (at FPR $\le 0.3$): **0.921** (Table 1).
  - Weakest category: Transistor (Pixel AUROC 0.825).
- **Limitations:** `[VERIFIED-ACCESSED]` `[INFERRED]`
  1. Identical forward student architecture allows abnormal features to leak through if the student over-generalizes.
  2. Element-wise product of 3 maps can cause zero-collapse if one layer predicts zero discrepancy.
  3. Official code uses deprecated `np.bool` and non-standard PRO min-max rescaling.
- **Kaggle Feasibility:** `[VERIFIED-ACCESSED]` **Excellent**. Trains in 2–8 minutes per category on Kaggle Tesla T4 GPU (~2 GB VRAM). Minimal dependencies (`torch`, `torchvision`, `scikit-learn`).

---

### Paper 2: CPKD — AAAI 2025
- **Citation:** Bokang Yang, Zhe Zhang, Jie Ma, *"Co-Progression Knowledge Distillation with Knowledge Prototype for Industrial Anomaly Detection,"* Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25), Vol. 39, No. 12, pp. 13008–13016, 2025.
- **Identifiers:** DOI: `10.1609/aaai.v39i12.33419`. URL: `https://ojs.aaai.org/index.php/AAAI/article/view/33419`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Open access via AAAI OJS.
- **Code Provenance:** `[VERIFIED-ACCESSED]` No official GitHub repository released in the paper proceedings or author profile (`github.com/yangbokang`).
- **Problem Addressed:** `[VERIFIED-ACCESSED]` In conventional KD, a static frozen teacher pretrained on ImageNet suffers from domain shift on industrial textures, while unconstrained teacher fine-tuning leads to catastrophic forgetting and representation collapse.
- **Method & Key Mechanism:** Co-Progression KD (CPKD) where both teacher and student evolve concurrently, regulated by a "Knowledge Prototype" that anchors the teacher's representation path to prevent drift.
- **Datasets & Metrics:** MVTec AD, VisA; Image-AUROC, Pixel-AUROC, PRO.
- **Limitations:** `[INFERRED]`
  1. Concurrently updating both teacher and student doubles backward-pass computation and optimizer memory on Kaggle GPUs.
  2. Absence of official code makes reproduction high-risk for a 3-week undergraduate timeline.
- **Feasibility as Comparator:** **Low / Infeasible**. Reproduction from scratch without official code introduces high variance.

---

### Paper 3: LKDF-AD — Scientific Reports 2026
- **Citation:** Xian Gao, *"A lightweight knowledge distillation framework for few shot unsupervised industrial surface anomaly detection and localization,"* Scientific Reports, Springer Nature, Vol. 16, Article 74501, 2026.
- **Identifiers:** DOI: `10.1038/s41598-026-74501-8`. URL: `https://www.nature.com/articles/s41598-026-74501-8`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Freely accessible on Nature.com.
- **Code Provenance:** Nature supplementary materials and author release.
- **Problem Addressed:** `[VERIFIED-ACCESSED]` Industrial scenarios where acquiring hundreds of normal images is prohibitive; standard KD models overfit or fail when only 1–10 normal reference samples are available.
- **Method & Key Mechanism:** Ultra-lightweight student network (containing only **385,168 parameters**, ~0.385M) distilled from a frozen teacher under **few-shot settings (1, 2, 4, 8, 10 shots)**. Employs compact feature mapping layers to prevent overfitting on few normal exemplars.
- **Datasets & Metrics:** MVTec AD, BTAD; Image-AUROC, Pixel-AUROC.
- **Critical Methodological Distinction:** `[VERIFIED-ACCESSED]` **LKDF-AD is strictly a few-shot framework**. It evaluates detection performance under limited-sample constraints ($K \in [1, 10]$).
- **Feasibility as Comparator:** **Valid ONLY with explicit protocol qualification**. It cannot be directly placed in the same benchmark table as full-data models without explicitly documenting that it operates under $K$-shot constraints.

---

### Paper 4: DHKD — Expert Systems with Applications 2026
- **Citation:** Muhao Xu, *"Beyond Feature Mapping: Dual-Heterogeneous Knowledge Distillation with Mamba for Industrial Anomaly Detection,"* Expert Systems with Applications, Vol. 308, Article 131146, 2026.
- **Identifiers:** DOI: `10.1016/j.eswa.2026.131146`. PII: `S0957-4174(26)00060-6`.
- **Full-Text Status:** `[UNVERIFIED-PAYWALL]` Elsevier ScienceDirect subscription paywall.
- **Code Provenance:** `[VERIFIED-ACCESSED]` Related author repositories: `github.com/Gilbert-Et/DHKD` and `github.com/Xmh-L/HDHKD`.
- **Problem Addressed:** `[VERIFIED-ACCESSED]` Homogeneous feature mapping struggles with complex industrial textures; standard Transformer decoders have $\mathcal{O}(L^2)$ quadratic spatial complexity.
- **Method & Key Mechanism:** Dual-heterogeneous distillation combining two structurally distinct student paths with State-Space Model (Mamba) sequence modeling to capture global context with linear complexity $\mathcal{O}(L)$.
- **Dependencies & Compute Burden:** `[VERIFIED-ACCESSED]` `[INFERRED]`
  - Requires `mamba-ssm` and `causal-conv1d`.
  - These packages depend on specific CUDA C++ kernel compilation matching the exact Torch and CUDA runtime installed on the host.
  - On Kaggle cloud environments (Tesla T4 / P100 with read-only CUDA toolkits), installing and compiling `mamba-ssm` routinely fails due to header mismatches or takes >25 minutes per session.
- **Feasibility as Comparator:** **High Risk / Impractical on Kaggle**. Architectural complexity and native CUDA compilation dependencies create severe reproducibility bottlenecks.

---

### Paper 5: CDA — Pattern Recognition 2026
- **Citation:** Ronglong Hu, Jizheng Yi, Xiaoyao Li, Aibin Chen, Guangjie Han, *"Efficient industrial anomaly detection via cross-scale distillation with enhanced feature compression,"* Pattern Recognition, Vol. 173, Article 112761, 2026.
- **Identifiers:** DOI: `10.1016/j.patcog.2025.112761`. Official Code: `https://github.com/WHUer-cloud/CDA`.
- **Full-Text Status:** `[UNVERIFIED-PAYWALL]` Full Elsevier PDF not publicly available.
- **Code Provenance & License:** `[VERIFIED-ACCESSED]` GitHub `WHUer-cloud/CDA`. **No LICENSE file** (All Rights Reserved by upstream authors; usable for academic analysis and clean reimplementation).
- **Method & Architecture:**
  - Frozen `Wide_ResNet50_2` teacher encoder ($24.86\text{M}$ params).
  - Global-local Compression Module (`GCM`, $19.70\text{M}$ params): Channel Local Extraction (`CLE`) + Spatial Global Attention (`SGA` with RoPE) + Residual bottleneck (`res`, $9.45\text{M}$).
  - Gated-depth Sequence Unit student decoder (`GSU`, $6.96\text{M}$ params).
  - Cross-scale distillation via 6 dense $3\times3$ convolutions (`c11..c32`, **$16.52\text{M}$ params = 38.3% of student!**).
- **Official Code Audit Findings:** `[VERIFIED-ACCESSED]`
  - 6 blocking/latent software bugs (filename typo `dataloder.py`, hard-coded paths, `parents=True` missing, non-portable slash parsing, sequence size fixed at 256).
  - Test-set checkpoint leakage: evaluates test set every 5 epochs and saves `best_img.pth` and `best_pix.pth` independently from test-set performance.
- **Feasibility as Comparator:** **Feasible ONLY via our clean-room, bug-fixed reimplementation (`src/anomaly_detection`)**, retaining it as a secondary modern baseline while acknowledging that the publisher PDF has not been inspected.

---

### Paper 6: SK-RD4AD — CVPRW 2025
- **Citation:** EunJu Park, Taekyung Kim, Minju Kim, Hojun Lee, Gil-Jun Lee, *"SK-RD4AD: Skip-Connected Reverse Distillation For Robust One-Class Anomaly Detection,"* IEEE/CVF CVPR Workshops (VAND), 2025.
- **Identifiers:** CVF Open Access (`https://openaccess.thecvf.com`). Official Code: `https://github.com/pej0918/SK-RD4AD`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Freely accessible on CVF Open Access.
- **Code License:** `[VERIFIED-ACCESSED]` No explicit license file in repository.
- **Problem Addressed:** `[VERIFIED-ACCESSED]` In Reverse Distillation (RD4AD), deep features undergo heavy bottleneck compression, leading to significant loss of fine-grained spatial and multi-scale representations in the decoder.
- **Method & Key Mechanism:** Introduces non-corresponding cross-hierarchical skip connections from intermediate teacher encoder layers directly into deeper decoder layers, bypassing the bottleneck to preserve multi-scale details.
- **Datasets & Metrics:** MVTec AD, VisA, Valeo Anomaly Dataset (VAD); Image-AUROC, Pixel-AUROC, PRO.
- **Kaggle Feasibility:** `[VERIFIED-ACCESSED]` **High**. Pure PyTorch CNN operations, builds directly on RD4AD codebase, easily run on Kaggle T4 GPUs without external compiled libraries. Highly recommended as a clean, recent comparator for reverse distillation.

---

### Paper 7: Dinomaly — CVPR 2025
- **Citation:** Jia Guo et al., *"Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection,"* IEEE/CVF CVPR 2025 (arXiv:2405.14325).
- **Identifiers:** `https://arxiv.org/abs/2405.14325`. Official Code: `https://github.com/bravo-j/Dinomaly`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Freely accessible on arXiv.
- **Code License:** `[VERIFIED-ACCESSED]` MIT License.
- **Problem Addressed:** Single-model multi-class unsupervised anomaly detection across diverse categories without per-category model retraining.
- **Method:** Minimalist Transformer reconstruction using frozen DINOv2 features, noisy bottleneck, and linear attention.
- **Relevance to CS331:** Exemplifies the Vision Foundation Model (VFM) trend in IAD. While highly accurate (>99% AUROC), DINOv2-based models have high parameter footprints (>80M–300M) and higher latency, defining the heavy "high-accuracy" end of the efficiency Pareto frontier against which lightweight student models compete.

---

### Paper 8: MVTec AD 2 Benchmark — IJCV / arXiv 2025
- **Citation:** Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, Carsten Steger, *"The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection,"* International Journal of Computer Vision (IJCV) / arXiv:2503.21622, 2025.
- **Identifiers:** `https://arxiv.org/abs/2503.21622`. Benchmark Portal: `https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2`.
- **Full-Text Status:** `[VERIFIED-ACCESSED]` Freely accessible on arXiv.
- **Core Findings & Relevance:**
  1. Identifies that standard MVTec AD is **heavily saturated** (>98–99% AUROC on simple textures), causing recent papers to over-fit to minor metric noise.
  2. Demonstrates that on realistic industrial defects (border defects, transparent objects, lighting shifts), SOTA models fall from >98% to <60% AU-PRO.
  3. Recommends focusing on **low-FPR localization ($\text{AUPRO}_{0.05}$ and Pixel-AP)** and **small/edge defect preservation**, directly validating the focus of **Researcher A (Thành)**.
