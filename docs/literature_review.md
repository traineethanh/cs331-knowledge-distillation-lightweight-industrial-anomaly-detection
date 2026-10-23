# Focused Literature Review (2021–2026)

This literature review examines **10 core research papers** (plus the MVTec AD / MVTec AD 2 benchmark papers) spanning foundational Teacher–Student distillation, reverse distillation, state-space/sequence decoders, multi-scale/cross-scale transfer, and lightweight industrial anomaly detection published through 2026.

> [!IMPORTANT]
> **Provenance & Evidence Tags:**
> - `[PAPER-SUPPORTED]`: Directly supported by verified paper publication/abstract/proceedings.
> - `[CODE-VERIFIED]`: Verified by inspecting official source code.
> - `[TECH-INTERPRETATION]`: Our technical synthesis and comparative analysis.
> - `[OPEN-VERIFICATION]`: Requires full-text access or empirical testing to confirm.

---

## 1. Detailed Per-Paper Analysis

### [1] CDA — Primary Baseline (Hu et al., *Pattern Recognition* 2026)
- **Verified Citation:** Ronglong Hu, Jizheng Yi, Xiaoyao Li, Aibin Chen, Guangjie Han. *"Efficient industrial anomaly detection via cross-scale distillation with enhanced feature compression."* Pattern Recognition, Vol. 173, Article 112761, 2026. DOI: `https://doi.org/10.1016/j.patcog.2025.112761`. Official Code: `https://github.com/WHUer-cloud/CDA`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Same-scale distillation ignores cross-scale semantic consistency; existing reverse distillation bottlenecks lose global spatial context and rely on heavy ResNet decoders.
- **Architecture & Training Objective:** `[CODE-VERIFIED]` Frozen `Wide_ResNet50_2` teacher (`layer1..3`) $\to$ `GCM` (`CLE` strided ConvBlocks + `SGA` 6-pair RoPE attention + `res` bottleneck) $\to$ 3-stage `GSU` (`PatchExpand2D` + `GSULayer`) student decoder $\to$ 6 dense $3\times3$ cross-scale alignment convolutions (`c11..c32`). Trained with $\mathcal{L} = 2.0\mathcal{L}_{\text{same}} + 0.1\mathcal{L}_{\text{cross}}$ using pointwise cosine distance.
- **Key Mechanism:** `[CODE-VERIFIED]` Dual global-local compression (`GCM`), Mamba-inspired gated sequence decoding (`GSU`), and all-to-all cross-scale feature alignment.
- **Dataset & Evaluation Metrics:** `[PAPER-SUPPORTED]` `[CODE-VERIFIED]` MVTec AD (15 classes) and VisA (12 classes); Image-level AUROC and Pixel-level AUROC (`metric.py`).
- **Limitations:** `[CODE-VERIFIED]` `[TECH-INTERPRETATION]`
  1. The 6 cross-scale $3\times3$ alignment layers (`c11..c32`) consume `16.52M` parameters (38.3% of the trainable model, $2.37\times$ larger than the `6.96M` GSU decoder), and `GCM.res` consumes another `9.45M` parameters.
  2. Inference fusion uses uncalibrated static weights $2.0(a_1+a_2+a_3) + 0.1(0.4ac_1 + 0.2ac_2 + 0.4ac_3)$ with fixed $\sigma=4$ Gaussian blur and single-pixel `max()` image scoring.
- **Overlap with Candidate Contributions:** Any candidate that merely claims "cross-scale distillation", "global-local compression", or "lightweight sequence decoder" overlaps directly with CDA. Our improvements must specifically target CDA's **uncalibrated static anomaly-map fusion** (Direction A) and **over-parameterized cross-scale alignment/bottleneck blocks** (Direction B).

---

### [2] STFPM — Foundational Multi-Scale Forward KD (Wang et al., *BMVC* 2021)
- **Verified Citation:** Guodong Wang, Shumin Han, Errui Ding, Di Huang. *"Student-Teacher Feature Pyramid Matching for Anomaly Detection."* BMVC 2021. DOI: `10.5244/C.35.349`. URL: `https://arxiv.org/abs/2103.04257`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Single-scale feature matching misses defects of varying sizes; pixel-level reconstruction methods (Autoencoders/GANs) produce blurry reconstructions and high false positives.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Identical Teacher and Student CNN backbones (e.g., ResNet-18). Teacher is ImageNet-pretrained and frozen; Student is randomly initialized and trained to minimize $\ell_2$-normalized feature distance (equivalent to cosine distance) at `layer1`, `layer2`, `layer3`.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Multi-scale feature pyramid matching during training and element-wise multiplication (or summation) of upsampled multi-scale discrepancy maps at inference.
- **Dataset & Metrics:** `[PAPER-SUPPORTED]` MVTec AD; Image-AUROC, Pixel-AUROC, PRO.
- **Limitations:** `[PAPER-SUPPORTED]` `[TECH-INTERPRETATION]` Because Teacher and Student share the exact same forward architecture and both see the raw image directly, the Student can over-generalize to anomalies; furthermore, only same-scale features are matched.
- **Relationship to CDA:** Foundational multi-scale baseline; CDA replaces STFPM's forward same-scale pyramid with a reverse bottleneck (`GCM`), lightweight `GSU` decoder, and cross-scale alignment.

---

### [3] RD4AD — Reverse Distillation Foundation (Deng & Li, *CVPR* 2022)
- **Verified Citation:** Hanqiu Deng, Xingyu Li. *"Anomaly Detection via Reverse Distillation from One-Class Embedding."* IEEE/CVF CVPR 2022, pp. 9737–9746. DOI: `https://doi.org/10.1109/CVPR52688.2022.00951`. URL: `https://arxiv.org/abs/2201.10703`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Forward Teacher–Student homogeneity allows abnormal features to pass through the student network.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Frozen `Wide_ResNet50_2` teacher encoder $\to$ Multi-scale Feature Fusion (MFF) + One-Class Embedding (OCE) bottleneck (together termed OCBE) $\to$ symmetric WideResNet-50 decoder using transposed convolutions. Minimizes multi-scale cosine distance $\sum_{k=1}^3 D_{\cos}(e_k, d_k)$.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Asymmetric Encoder–Bottleneck–Decoder flow restricts the student input to a compact $(2048, 8, 8)$ normal embedding.
- **Dataset & Metrics:** `[PAPER-SUPPORTED]` MVTec AD; Image-AUROC (98.5%), Pixel-AUROC (97.8%), AUPRO (93.9%).
- **Limitations:** `[PAPER-SUPPORTED]` `[TECH-INTERPRETATION]` OCBE uses only local convolutions (no global spatial attention), the symmetric WideResNet-50 decoder is parameter-heavy, and distillation is strictly same-scale.
- **Relationship to CDA:** `[CODE-VERIFIED]` CDA is a direct architectural descendant of RD4AD (`model.py` line 121 explicitly comments `# ---------- OCBE + Decoder ----------`), upgrading MFF+OCE to `GCM`, the ResNet decoder to `GSU`, and same-scale loss to cross-scale loss.

---

### [4] DeSTSeg — Supervised Synthetic Anomaly Fusion (Zhang et al., *CVPR* 2023)
- **Verified Citation:** Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, Ting Chen. *"DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection."* IEEE/CVF CVPR 2023, pp. 3914–3923. DOI: `https://doi.org/10.1109/CVPR52729.2023.00381`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Heuristic equal-weighted multi-scale map summation is suboptimal, and training purely on clean normal images leaves student behavior on anomalies unconstrained.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Denoising Student–Teacher trained with synthetic Perlin-noise anomalies + a trainable segmentation head that fuses multi-scale Teacher–Student discrepancy maps using focal + $\ell_1$ loss against synthetic masks.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Supervised multi-scale discrepancy fusion via synthetic anomaly masks.
- **Limitations:** `[TECH-INTERPRETATION]` Performance depends heavily on whether synthetic anomaly textures resemble real test defects; adds extra training stages and segmentation network parameters.
- **Relationship to CDA & Overlap Check:** DeSTSeg uses *supervised synthetic-mask training* to learn multi-scale fusion. Therefore, for Researcher A (Thành), our anomaly-map calibration/fusion method should remain **strictly unsupervised (normal-statistics & cross-scale consistency driven)** rather than claiming supervised synthetic segmentation fusion.

---

### [5] EfficientAD — Millisecond-Latency Distillation (Batzner et al., *WACV* 2024)
- **Verified Citation:** Kilian Batzner, Lars Heckler, Rebecca König. *"EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies."* IEEE/CVF WACV 2024, pp. 128–138. DOI: `https://doi.org/10.1109/WACV57701.2024.00020`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` High inference latency of deep backbones and inability of patch-level Teacher–Student networks to capture global logical anomalies.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Ultra-lightweight Patch Description Network (PDN, 4 conv layers) as Teacher/Student + an Autoencoder branch for global logical anomalies, trained with hard-feature mining loss on natural image auxiliary data (ImageNet) and quantile-based map normalization.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Shallow PDN + hard-feature loss + quantile calibration of local (Student–Teacher) and global (Autoencoder–Student) anomaly maps computed on normal validation images.
- **Limitations:** `[PAPER-SUPPORTED]` Requires auxiliary ImageNet dataset during training for hard-feature mining and a separate pre-distilled PDN teacher.
- **Relationship to CDA & Overlap Check:** EfficientAD demonstrates the value of **normal-distribution quantile/statistical calibration** when combining heterogeneous anomaly maps (local vs. global). Critically, CDA lacks any normal-distribution scale calibration between its same-scale maps ($a_1, a_2, a_3$) and cross-scale maps ($ac_1, ac_2, ac_3$).

---

### [6] MambaAD — State-Space Decoder for IAD (He et al., *NeurIPS* 2024)
- **Verified Citation:** Haoyang He, Yuhu Bai, Jiangning Zhang, Qingdong He, Hongxu Chen, Zhenye Gan, Chengjie Wang, Xiangtai Li, Guanzhong Tian, Lei Xie. *"MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection."* NeurIPS 2024. URL: `https://arxiv.org/abs/2404.06564`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` CNN decoders lack long-range context while Transformer decoders suffer from $\mathcal{O}(L^2)$ complexity.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Pretrained encoder + Locality-Enhanced State Space (LSS) decoder combining multi-directional Hilbert/raster Mamba scans with multi-kernel depthwise convolutions; trained with multi-scale cosine distance loss.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Linear-complexity gated state-space sequence blocks with 1D depthwise convolutions for multi-scale feature reconstruction.
- **Relationship to CDA & Overlap Check:** `[CODE-VERIFIED]` In `CDA/model.py` (commit `604dc01`), the `GSU` block was originally commented `# ---------- Mamba ----------`, as `GSU` uses Mamba's gated `in_proj -> Conv1d(k=4, groups=d) -> SiLU gate -> out_proj` structure (replacing the SSM scan with `DepthwiseMLP`). This means **claiming a Mamba/gated-sequence decoder is NOT novel relative to MambaAD (2024) and CDA (2026)**.

---

### [7] CPKD — Co-Progression KD with Knowledge Prototype (Yang et al., *AAAI* 2025)
- **Verified Citation:** Bokang Yang, Zhe Zhang, Jie Ma. *"Co-Progression Knowledge Distillation with Knowledge Prototype for Industrial Anomaly Detection."* AAAI 2025, Vol. 39, No. 12, pp. 13008–13016. DOI: `https://doi.org/10.1609/aaai.v39i12.33419`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Static frozen teachers pretrained on ImageNet suffer from domain shift on industrial surfaces, while unconstrained co-training causes catastrophic forgetting.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Bidirectional Teacher–Student co-progression regulated by a Knowledge Prototype mechanism that anchors the teacher's representation path.
- **Relationship to CDA:** Highlights that keeping the teacher completely frozen (`Wide_ResNet50_2` in `.eval()`) leaves shallow teacher features ($e_1$) susceptible to normal industrial texture noise.

---

### [8] Cross-Architecture Distillation (IEEE *ICIPCA* 2025)
- **Verified Citation:** *"Exploring Cross-Architecture Distillation for Industrial Anomaly Detection."* IEEE 3rd International Conference on Image Processing and Computer Applications (ICIPCA), 2025. DOI: `https://doi.org/10.1109/ICIPCA65645.2025.11138658`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Homogeneous CNN-to-CNN distillation misses global semantic priors from Vision Foundation Models (VFMs) while full VFMs are too slow for edge deployment.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Distils a high-capacity `DINOv2-Small` ViT teacher into a lightweight `ResNet-18` student using ResNet block adapters and QKV cross-attention feature migration. Evaluated on VisA.
- **Relationship to CDA & Overlap Check:** Demonstrates adapter-based feature migration between heterogeneous modules.

---

### [9] Multi-Teacher KD for Lightweight AD (Yousefimehr et al., *Neural Networks* 2026)
- **Verified Citation:** Behnam Yousefimehr, Mehdi Ghatee, Roozbeh Razavi-Far. *"Multi-Teacher Knowledge Distillation Framework for Lightweight Anomaly Detection."* Neural Networks, Vol. 195, Article 108267, 2026. DOI: `https://doi.org/10.1016/j.neunet.2025.108267`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` A single teacher architecture cannot capture all normal feature variations when distilling into a compact lightweight student.
- **Architecture & Training Objective:** `[PAPER-SUPPORTED]` Aggregates knowledge from multiple teachers with a resampling strategy to train a compact student network for resource-constrained environments.
- **Relationship to CDA & Overlap Check:** Confirms that generic "lightweight student via KD" is already well-explored; our efficiency direction (Researcher B) must target the **specific structural parameter bottleneck inside Cross-Scale Reverse Distillation (`c11..c32` and `GCM.res`)** rather than generic student shrinking.

---

### [10] SADED — Synthetic Anomalies & Discrepancy Enhancement (Lin et al., *Comput. Ind. Eng.* 2026)
- **Verified Citation:** Tai Lin, Quanquan Li, Yan Song, Guoxiang Tong. *"Synthetic anomalies oriented knowledge distillation with discrepancy enhancements for industrial anomaly detection."* Computers & Industrial Engineering, Vol. 212, Article 111709, 2026. DOI: `https://doi.org/10.1016/j.cie.2025.111709`.
- **Problem Addressed:** `[PAPER-SUPPORTED]` Training only on normal samples yields weak Teacher–Student discrepancy boundaries on subtle defects.
- **Key Mechanism:** `[PAPER-SUPPORTED]` Synthetic anomaly injection paired with discrepancy enhancement losses during distillation.

---

### [11] Benchmark Foundations: MVTec AD (CVPR 2019) & MVTec AD 2 (2025)
- **MVTec AD:** Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. *"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."* CVPR 2019, pp. 9592–9600. (`https://www.mvtec.com/research-teaching/datasets/mvtec-ad`).
- **MVTec AD 2:** Lars Heckler-Kram et al. *"The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection."* 2025. (`https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2`). Highlights that tiny border defects, high normal variance, and strict low-FPR localization ($\text{AUPRO}_{0.05}$, Pixel-AP) are the true differentiators where saturated Image/Pixel AUROC metrics fail.

---

## 2. Literature Comparison Matrix

| Work (Year, Venue) | Paradigm | Teacher Backbone | Bottleneck / Compression | Student Architecture | Cross-Scale Transfer? | Anomaly Map Fusion at Inference | Primary Limitation Targeted by Our Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STFPM** (BMVC 2021) | Forward KD | ResNet-18 | None (Direct input) | ResNet-18 (3 stages) | No (Same-scale only) | Product/sum of 3 upsampled maps | Over-generalization; no cross-scale or reliability calibration |
| **RD4AD** (CVPR 2022) | Reverse KD | WideResNet50_2 | OCBE (Local MFF + OCE ConvBlock) | Symmetric WideResNet50 Decoder | No (Same-scale only) | Equal sum $a_1+a_2+a_3$ + Gaussian $\sigma=4$ | Local-only bottleneck; heavy ResNet decoder; static sum |
| **DeSTSeg** (CVPR 2023) | Denoising KD + Seg | ResNet-18 | None | Denoising Encoder-Decoder + SegNet | Via SegNet | Supervised SegNet trained on synthetic masks | Depends on synthetic anomaly distribution |
| **EfficientAD** (WACV 2024) | Patch KD + AE | PDN (4-layer CNN) | AE bottleneck (global branch) | PDN + Autoencoder | Local + Global | Quantile-calibrated average on validation set | Requires external ImageNet data during training |
| **MambaAD** (NeurIPS 2024) | Reverse KD | ResNet / WideResNet | MFF + OCE | Multi-scan Mamba (HSS/LSS) Decoder | No (Same-scale only) | Sum of multi-scale cosine maps + Gaussian blur | Complex 8-way Hilbert/raster SSM scanning |
| **CPKD** (AAAI 2025) | Co-Progression KD | Co-evolving CNN | Knowledge Prototype | Co-evolving Student | No | Multi-scale discrepancy sum | Doubles training memory by updating teacher |
| **Cross-Arch KD** (ICIPCA 2025) | Heterogeneous KD | DINOv2-Small (ViT) | ResNet Block Adapters | ResNet-18 + Cross-Attention | Multi-level QKV | Multi-scale discrepancy aggregation | ViT teacher latency & adapter tuning complexity |
| **Multi-Teacher KD** (Neural Netw. 2026) | Multi-Teacher KD | Multiple Teachers | Resampling fusion | Compact CNN Student | Teacher ensemble | Aggregated student-teacher error | Multiple teachers increase training memory/cost |
| **SADED** (CIE 2026) | Synthetic-Guided KD | Pretrained CNN | Discrepancy Enhancement | Student CNN | Multi-scale | Enhanced discrepancy fusion | Relies on synthetic defect generator tuning |
| **CDA** (Pattern Recognit. 2026) **[BASELINE]** | **Cross-Scale Reverse KD** | **WideResNet50_2** (`24.86M`) | **GCM (`CLE` + `SGA` RoPE + `res`)** (`19.70M`) | **GSU Decoder (`6.96M`) + Dense $3\times3$ Adapters (`16.52M`)** | **Yes (6 dense $3\times3$ convs `c11..c32`)** | **Static sum $2.0\sum a_k + 0.1\sum w_k ac_k$, $\sigma=4$, `max()`** | **1. `16.52M` params in dense $3\times3$ adapters (`c11..c32`) + `9.45M` in `GCM.res`<br/>2. Uncalibrated static fusion & outlier-sensitive `max()` scoring** |
