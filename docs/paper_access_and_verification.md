# Paper Access and Verification Record

## 1. Bibliographic Identity of the Primary Paper

- **Title:** Efficient industrial anomaly detection via cross-scale distillation with enhanced feature compression
- **Authors:** Ronglong Hu, Jizheng Yi, Xiaoyao Li, Aibin Chen, Guangjie Han
- **Journal:** *Pattern Recognition* (Elsevier), Volume 173, Article 112761, 2026
- **DOI:** `https://doi.org/10.1016/j.patcog.2025.112761`
- **ScienceDirect PII:** `S0031320325014244` (`https://www.sciencedirect.com/science/article/pii/S0031320325014244`)
- **DBLP Record:** `https://dblp.org/rec/journals/pr/HuYLCH26`
- **Official Author Repository:** `https://github.com/WHUer-cloud/CDA` (GitHub user: `WHUer-cloud`, commit author: `LiyAohUi <19374154819@163.com>`, latest commit `67224ca04d3aff03c97bdaad04aafb0b3a1ad74e` dated 2025-10-19)

---

## 2. Materials Actually Accessed and Verified

In accordance with **Work Package 1** operational rules (*"Do not bypass access controls or assume that an abstract is equivalent to the full text... Record which materials were actually accessible"*), we explicitly document the exact provenance of every piece of information used in this repository:

| Source Material | Access Status | What Was Verified |
| :--- | :--- | :--- |
| **Publisher Page & Indexing Metadata** (ScienceDirect `S0031320325014244`, DBLP, Crossref) | **Accessible (Public Metadata/Abstract)** | Full title, author list, journal volume/article number, DOI, publication year (2026), core architectural module names (CDA, GCM, CLE, SGA, GSU), and high-level problem statement. |
| **Full-Text Publisher PDF** (`10.1016/j.patcog.2025.112761`) | **Restricted (Subscription Paywall / Not in Local Workspace)** | A workspace-wide and local filesystem scan (`Desktop`, `Downloads`, sibling folders) confirmed that the copyrighted Elsevier PDF is not stored locally, and no legitimate open-access preprint (arXiv/SSRN) is publicly posted. |
| **Official Author Source Code** (`https://github.com/WHUer-cloud/CDA`) | **Fully Accessible & Audited (Cloned & Inspected Including Full Git History)** | Complete PyTorch implementation (`model.py`, `Transformer.py`, `train.py`, `test.py`, `dataloder.py`, `metric.py`, `config.py`, `README.md`) across all 9 commits (`0aae557` to `67224ca`). |
| **Foundational Reference Paper (STFPM)** (`https://arxiv.org/abs/2103.04257`) | **Accessible (Open Access)** | Student–Teacher Feature Pyramid Matching formulation, multi-scale cosine distance loss, anomaly map aggregation, and evaluation protocol. |
| **Direct Architectural Predecessor (RD4AD)** (`CVPR 2022`, Deng & Li) | **Accessible (Open Access)** | Reverse Distillation from One-Class Bottleneck Embedding (OCBE), WideResNet50_2 teacher encoder (`layer1`, `layer2`, `layer3`), and MFF/OCE bottleneck structure upon which CDA's `GCM` directly builds (explicitly noted by `# ---------- OCBE + Decoder ----------` in `CDA/model.py`). |

---

## 3. Provenance Classification: Paper-Reported vs. Code-Verified vs. Unresolved

To maintain strict scientific integrity, all technical specifications in our documentation are tagged under three categories:

### 3.1 Verified from Paper Metadata + Official Code (`[VERIFIED-BOTH]`)
1. **Overall Paradigm:** Reverse knowledge distillation using a frozen pretrained CNN teacher encoder (`Wide_ResNet50_2`), a Global-local Compression Module (`GCM`), a lightweight Gated-depth Sequence Unit (`GSU`) student decoder, and cross-scale feature distillation.
2. **Core Module Mapping:**
   - **Teacher Encoder:** Frozen `wide_resnet50_2` (`Wide_ResNet50_2_Weights.DEFAULT`) extracting multi-scale feature maps $e_1 \in \mathbb{R}^{B \times 256 \times 64 \times 64}$, $e_2 \in \mathbb{R}^{B \times 512 \times 32 \times 32}$, $e_3 \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$ for $256 \times 256$ inputs.
   - **Global-local Compression Module (GCM):** Combines **Channel-oriented Local Extraction (CLE)** (`local1`, `local2`, `merge` in `model.py`) and **Spatial-guided Global Attention (SGA)** (`Fusion` with 2D-flattened RoPE multi-scale attention and `DoubleConv` in `Transformer.py`), followed by residual bottleneck compression (`res` in `model.py`) to produce $f \in \mathbb{R}^{B \times 2048 \times 8 \times 8}$.
   - **Lightweight Student Decoder with GSU:** 3-stage decoder (`l3`, `l2`, `l1`), each pairing `PatchExpand2D` (bilinear $2\times$ upsampling + linear channel halving + LayerNorm) with `GSULayer` (`RMSNorm` + `GSU` containing `in_proj`, causal-padded depthwise `Conv1d(k=4)`, `DepthwiseMLP` via `GroupedLinear`, SiLU gating, and `out_proj`).
   - **Cross-Scale Distillation:** Aligning decoded student features $(s_1, s_2, s_3)$ across scales via bilinear interpolation and $3 \times 3$ alignment convolutions (`c11`, `c12`, `c21`, `c22`, `c31`, `c32`) to match teacher features $(e_1, e_2, e_3)$ both within the same scale and across scales.

### 3.2 Verified Directly from Official Source Code (`[VERIFIED-CODE]`)
1. **Default Training Hyperparameters (`config.py`, `train.py`):**
   - Input resolution: $256 \times 256$, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - Optimizer: `Adam(dec.parameters(), lr=0.005, betas=(0.5, 0.999))`.
   - Batch size: `bs = 4`.
   - Epochs: `epochs = 160`.
   - Evaluation interval: `test_interval = 5` epochs.
2. **Exact Loss Weighting & Anomaly Scoring Formula (`train.py`, `metric.py`):**
   - Same-scale cosine distance loss $\mathcal{L}_1$ weighted by $2.0$ and cross-scale cosine distance loss $\mathcal{L}_2$ weighted by $0.1$, with scale weights $(0.4, 0.2, 0.4)$.
   - Gaussian filter smoothing with $\sigma = 4$ (`scipy.ndimage.gaussian_filter`) on the $256 \times 256$ fused anomaly map.
   - Image-level anomaly score computed as the maximum pixel value (`s.max()`) of the smoothed anomaly map.

### 3.3 Unresolved Without Full-Text Paper PDF (`[UNRESOLVED-PDF-REQUIRED]`)
Because the Elsevier full-text PDF is paywalled and not present in the workspace, the following items cannot be independently cross-checked against the paper's prose/tables until the team provides the PDF:
1. **Exact Per-Category Numerical Tables in the Paper:** The exact Image-AUROC, Pixel-AUROC, and PRO/AUPRO numbers reported in the paper's tables for MVTec AD and VisA (as well as which competing methods are listed in each table).
2. **Paper Equations vs. Code Parentheses Discrepancy in $\mathcal{L}_2$:**
   - In `metric.py` (line 21–32), the cross-scale anomaly score weights both cross-scale terms at scale $k$ equally: $0.4 \cdot (\text{cos\_dist}(e_1, l_{1n}) + \text{cos\_dist}(e_1, l_{1m}))$.
   - In `train.py` (line 43–45), the code writes `(0.4 * cos_dist(e1, l1n) + cos_dist(e1, l1m)).mean()`, which multiplies only $l_{1n}$ by $0.4$ while leaving $l_{1m}$ with weight $1.0$.
   - Whether the printed equation in the paper applies $\alpha_k \in \{0.4, 0.2, 0.4\}$ to the sum $(l_{kn} + l_{km})$ (as in `metric.py`) or only to $l_{kn}$ (as in `train.py`) requires checking the paper's printed loss equation.
3. **Paper Terminology Mapping for Sub-Equations:** Whether the paper uses specific Greek symbols (e.g., $\lambda_1, \lambda_2$ or $\alpha, \beta$) for the loss weights `2.0`, `0.1`, `0.4`, `0.2`, `0.4`.
4. **Reported Hardware & PRO Metric:** Which GPU model the authors used to measure inference speed in the paper, and whether the paper also reports pixel-level AUPRO alongside Image-AUROC and Pixel-AUROC (note: `metric.py` in the official repository only implements `roc_auc_score` for image and pixel levels).

---

## 4. Action Item for the Research Team (Thành & Luân)

If institutional access (e.g., via university library proxy) allows downloading the PDF of `10.1016/j.patcog.2025.112761`:
1. Keep the PDF in a local non-committed folder (do **not** commit copyrighted publisher PDFs to GitHub).
2. Cross-check the printed equation for $\mathcal{L}_2$ and Table 1/2 per-category MVTec AD numbers against `docs/paper_equations.md` and `docs/official_code_audit.md`.
