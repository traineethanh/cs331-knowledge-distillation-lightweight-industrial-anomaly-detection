# Mathematical Formulation and Equations of CDA

> [!NOTE]
> All equations below are reconstructed directly from the verified official implementation (`model.py`, `Transformer.py`, `train.py`, `metric.py` in `https://github.com/WHUer-cloud/CDA`) and aligned with the architectural components described in Hu et al., *Pattern Recognition* 2026 (`10.1016/j.patcog.2025.112761`). Where an implementation detail in `train.py` differs from `metric.py` or requires cross-checking against the paywalled PDF, it is explicitly marked.

---

## 1. Multi-Scale Teacher Feature Extraction (`model.py: Encoder`)

Given an RGB input image $I \in \mathbb{R}^{B \times 3 \times H \times W}$ with default resolution $H = W = 256$, normalized by ImageNet channel statistics $\mu = [0.485, 0.456, 0.406]^\top$ and $\sigma = [0.229, 0.224, 0.225]^\top$, the frozen pretrained `Wide_ResNet50_2` teacher encoder $\mathcal{E}_{\text{teacher}}$ extracts three hierarchical feature maps:

$$x_0 = \text{MaxPool}\big(\text{ReLU}(\text{BN}_1(\text{Conv}_{7\times7, s=2}(I)))\big) \in \mathbb{R}^{B \times 64 \times 64 \times 64}$$

$$e_1 = \text{Layer}_1(x_0) \in \mathbb{R}^{B \times 256 \times 64 \times 64}, \quad e_2 = \text{Layer}_2(e_1) \in \mathbb{R}^{B \times 512 \times 32 \times 32}, \quad e_3 = \text{Layer}_3(e_2) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$

- **Dimensions:** For $k \in \{1, 2, 3\}$, $e_k \in \mathbb{R}^{B \times C_k \times H_k \times W_k}$ with $(C_1, H_1, W_1) = (256, 64, 64)$, $(C_2, H_2, W_2) = (512, 32, 32)$, $(C_3, H_3, W_3) = (1024, 16, 16)$.

---

## 2. Global-local Compression Module (GCM) (`model.py: GCM`, `Transformer.py: Fusion`)

### 2.1 Residual Convolutional Block (`model.py: ConvBlock`)
For input $X \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$, filter triplet $[f_1, f_2, f_3]$, kernel size $k=3$, and stride $s$:

$$\text{Stage}(X) = \phi\Big(\text{BN}_3\big(\text{Conv}_{1\times1}^{f_2 \to f_3}\big(\phi(\text{BN}_2(\text{Conv}_{3\times3, p=1}^{f_1 \to f_2}(\phi(\text{BN}_1(\text{Conv}_{1\times1, s}^{C_{\text{in}} \to f_1}(X))))))\big)\big)\Big)$$

$$\text{Short}(X) = \phi\big(\text{BN}_s(\text{Conv}_{1\times1, s}^{C_{\text{in}} \to f_3}(X))\big)$$

$$\text{ConvBlock}(X) = \text{Stage}(X) + \text{Short}(X) \in \mathbb{R}^{B \times f_3 \times (H_{\text{in}}/s) \times (W_{\text{in}}/s)}$$

where $\phi(\cdot) = \text{GELU}(\cdot)$.

### 2.2 Channel-oriented Local Extraction (CLE) (`model.py: GCM.local1, local2, merge`)
CLE aligns $e_1$ and $e_2$ to the spatial resolution of $e_3$ ($16 \times 16$) via strided `ConvBlock` modules and merges them with $e_3$:

$$l_1 = \text{ConvBlock}_{256 \to [256, 512, 1024], s=4}(e_1) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$

$$l_2 = \text{ConvBlock}_{512 \to [512, 512, 1024], s=2}(e_2) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$

$$l = \text{Conv}_{1\times1}^{3072 \to 1024}\big(\text{Concat}([l_1, l_2, e_3], \text{dim}=1)\big) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$

### 2.3 Spatial-guided Global Attention (SGA) (`Transformer.py: Fusion, SelfAttention`)
1. **Spatial Resizing and Channel Compression to Spatial Tokens:**
   $$\tilde{p}_1 = \text{Interp}_{16\times16}(e_1) \in \mathbb{R}^{B \times 256 \times 16 \times 16}, \quad \tilde{p}_2 = \text{Interp}_{16\times16}(e_2) \in \mathbb{R}^{B \times 512 \times 16 \times 16}$$
   $$t_1 = \text{Reshape}_{B \times 256 \times 1}\big(\text{Conv}_{1\times1}^{256 \to 1}(\tilde{p}_1)\big), \quad t_2 = \text{Reshape}_{B \times 256 \times 1}\big(\text{Conv}_{1\times1}^{512 \to 1}(\tilde{p}_2)\big), \quad t_3 = \text{Reshape}_{B \times 256 \times 1}\big(\text{Conv}_{1\times1}^{1024 \to 1}(e_3)\big)$$
   Here $L = 16 \times 16 = 256$ represents the flattened spatial grid positions, and each token initially has scalar channel dimension $1$.

2. **Rotary Position Embedding (RoPE) Multi-Head Cross/Self-Attention (`SelfAttention`):**
   Given token pair $(t_i, t_j)$ with $t_i, t_j \in \mathbb{R}^{B \times L \times 1}$ ($L=256$) and hidden dimension $D = 256$, heads $H_a = 8$ ($d_h = D / H_a = 32$):
   $$\tilde{Q} = t_i W_Q \in \mathbb{R}^{B \times L \times D}, \quad \tilde{K} = t_j W_K \in \mathbb{R}^{B \times L \times D}, \quad V = t_j W_V \in \mathbb{R}^{B \times L \times D}$$
   For position index $p \in \{0, \dots, L-1\}$ and frequency index $m \in \{0, \dots, D/2 - 1\}$ with $\theta_m = 10000^{-2m/D}$:
   $$\text{RoPE}(u)_{p, 2m} = u_{p, 2m}\cos(p\theta_m) - u_{p, 2m+1}\sin(p\theta_m)$$
   $$\text{RoPE}(u)_{p, 2m+1} = u_{p, 2m+1}\cos(p\theta_m) + u_{p, 2m}\sin(p\theta_m)$$
   After applying $Q = \text{RoPE}(\tilde{Q})$ and $K = \text{RoPE}(\tilde{K})$ and reshaping into $H_a = 8$ heads of shape $(B, H_a, L, d_h)$:
   $$A_{i,j} = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{\tau}}\right) V \in \mathbb{R}^{B \times H_a \times L \times d_h}$$
   *(Implementation note verified in `Transformer.py` line 147: the code sets `scaling = K.shape[2] = 256` ($\tau = L = 256$) rather than $d_h = 32$.)*
   After merging heads, applying output projection $W_O \in \mathbb{R}^{D \times D}$, and reshaping to spatial grid $(B, 256, 16, 16)$:
   $$M_{i,j} = \text{Reshape}_{B \times 256 \times 16 \times 16}\big(\text{ConcatHeads}(A_{i,j}) W_O\big)$$

3. **Multi-Scale Pair Aggregation and DoubleConv:**
   Computing all 6 upper-triangular scale interactions $\mathcal{P} = \{(1,1), (1,2), (1,3), (2,2), (2,3), (3,3)\}$:
   $$g = \text{DoubleConv}_{1536 \to 512 \to 1024}\Big(\text{Concat}_{(i,j)\in\mathcal{P}}\big(M_{i,j},\, \text{dim}=1\big)\Big) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$

### 2.4 Bottleneck Compression (`model.py: GCM.res`)
$$f = \text{ConvBlock}_{1024 \to [512, 1024, 2048], s=2}(g + l) \in \mathbb{R}^{B \times 2048 \times 8 \times 8}$$

---

## 3. Lightweight Student Decoder and GSU (`model.py: Decoder, PatchExpand2D, GSULayer, GSU`)

### 3.1 PatchExpand2D (`model.py: PatchExpand2D`)
Given 2D feature map $X \in \mathbb{R}^{B \times C \times H \times W}$ and channel reduction factor $\text{scale}=2$:
$$\tilde{X} = \text{Interp}_{\text{scale}=2, \text{bilinear}}(X) \in \mathbb{R}^{B \times C \times 2H \times 2W}$$
$$U = \text{LayerNorm}\big(\text{FlattenSpatial}(\tilde{X})^\top W_{\text{expand}}\big) \in \mathbb{R}^{B \times (4HW) \times (C/2)}$$

### 3.2 RMSNorm and Gated-depth Sequence Unit (`model.py: RMSNorm, GSU, GSULayer`)
For sequence input $U \in \mathbb{R}^{B \times L \times d}$:
1. **RMSNorm:**
   $$\text{RMSNorm}(U) = \frac{U}{\sqrt{\frac{1}{d}\sum_{c=1}^{d} U_{\dots, c}^2 + \epsilon}} \odot \gamma, \quad \epsilon = 10^{-5}, \; \gamma \in \mathbb{R}^d$$
2. **GSU Forward Pass** on normalized input $\hat{U} = \text{RMSNorm}_1(U)$:
   $$[X_0, Z] = \hat{U} W_{\text{in}} \in \mathbb{R}^{B \times L \times 2d}, \quad X_0, Z \in \mathbb{R}^{B \times L \times d}$$
   $$X_1 = \text{DWConv1D}_{k=4, \text{pad}=3}(X_0^\top)[:, :, :L]^\top \in \mathbb{R}^{B \times L \times d}$$
   $$X_2 = \text{SiLU}\big(\text{GroupedLinear}_{d \to d, g}(X_1)\big) \in \mathbb{R}^{B \times L \times d}, \quad g = \max\{k \le \lfloor\sqrt{d}\rfloor : k \mid d\}$$
   $$\text{GSU}(\hat{U}) = \big(\text{RMSNorm}_2(X_2) \odot \text{SiLU}(Z)\big) W_{\text{out}} \in \mathbb{R}^{B \times L \times d}$$
3. **Residual Layer Output:**
   $$\text{GSULayer}(U) = U + \text{GSU}(\text{RMSNorm}_1(U)) \in \mathbb{R}^{B \times L \times d}$$

### 3.3 Hierarchical Decoder Reconstruction (`model.py: Decoder`)
Starting from $f \in \mathbb{R}^{B \times 2048 \times 8 \times 8}$:
$$s_3 = \text{To2D}\big(\text{GSULayer}_{1024}(\text{PatchExpand2D}_{2048 \to 1024}(f))\big) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}$$
$$s_2 = \text{To2D}\big(\text{GSULayer}_{512}(\text{PatchExpand2D}_{1024 \to 512}(s_3))\big) \in \mathbb{R}^{B \times 512 \times 32 \times 32}$$
$$s_1 = \text{To2D}\big(\text{GSULayer}_{256}(\text{PatchExpand2D}_{512 \to 256}(s_2))\big) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

---

## 4. Cross-Scale Alignment Projections (`model.py: GCMAndDecoder`)

To align decoded student features across scales, six $3 \times 3$ convolutions (`padding=1, stride=1`) follow bilinear interpolation (`align_corners=True`):

$$\begin{aligned}
l_{1m} &= \text{Conv}_{3\times3}^{512 \to 256}\big(\text{Interp}_{64\times64}(s_2)\big) \in \mathbb{R}^{B \times 256 \times 64 \times 64}, & l_{1n} &= \text{Conv}_{3\times3}^{1024 \to 256}\big(\text{Interp}_{64\times64}(s_3)\big) \in \mathbb{R}^{B \times 256 \times 64 \times 64} \\
l_{2m} &= \text{Conv}_{3\times3}^{256 \to 512}\big(\text{Interp}_{32\times32}(s_1)\big) \in \mathbb{R}^{B \times 512 \times 32 \times 32}, & l_{2n} &= \text{Conv}_{3\times3}^{1024 \to 512}\big(\text{Interp}_{32\times32}(s_3)\big) \in \mathbb{R}^{B \times 512 \times 32 \times 32} \\
l_{3m} &= \text{Conv}_{3\times3}^{256 \to 1024}\big(\text{Interp}_{16\times16}(s_1)\big) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}, & l_{3n} &= \text{Conv}_{3\times3}^{512 \to 1024}\big(\text{Interp}_{16\times16}(s_2)\big) \in \mathbb{R}^{B \times 1024 \times 16 \times 16}
\end{aligned}$$

---

## 5. Loss Terms and Variable Specification (`train.py`, `metric.py`)

### 5.1 Pointwise Channel-wise Cosine Distance (`metric.py: cos_dist`)
For any two aligned feature tensors $F^{(1)}, F^{(2)} \in \mathbb{R}^{B \times C_k \times H_k \times W_k}$:

$$D_{\cos}\big(F^{(1)}, F^{(2)}\big)_{b, h, w} = 1 - \frac{\sum_{c=1}^{C_k} F^{(1)}_{b,c,h,w} F^{(2)}_{b,c,h,w}}{\max\left(\|F^{(1)}_{b,:,h,w}\|_2 \cdot \|F^{(2)}_{b,:,h,w}\|_2,\, \epsilon\right)} \in \mathbb{R}^{B \times H_k \times W_k}$$

- **Implementation:** `metric.py` line 6 (`cos_dist(f1, f2)` using `1 - torch.nn.functional.cosine_similarity(f1, f2, dim=1)`).

### 5.2 Loss Term 1: Same-Scale Distillation Loss ($\mathcal{L}_1$)
1. **Equation:**
   $$\mathcal{L}_1 = \sum_{k=1}^{3} \frac{1}{B H_k W_k} \sum_{b=1}^{B} \sum_{h=1}^{H_k} \sum_{w=1}^{W_k} D_{\cos}(e_k, s_k)_{b,h,w}$$
2. **Variables & Shapes:**
   - $e_k \in \mathbb{R}^{B \times C_k \times H_k \times W_k}$: Frozen teacher feature at scale $k \in \{1, 2, 3\}$.
   - $s_k \in \mathbb{R}^{B \times C_k \times H_k \times W_k}$: Student decoder feature at matching scale $k$ (`d1, d2, d3` in `train.py`).
   - $D_{\cos}(e_k, s_k) \in \mathbb{R}^{B \times H_k \times W_k}$: Spatial cosine distance map at scale $k$.
   - $\mathcal{L}_1 \in \mathbb{R}$: Scalar loss averaged over batch and spatial coordinates per scale.
3. **Purpose:** Enforces direct feature reconstruction at each pyramid scale $64\times64$, $32\times32$, and $16\times16$ on normal training images.
4. **Implementation File & Line:** `train.py`, line 42:
   ```python
   loss1 = cos_dist(e1, d1).mean() + cos_dist(e2, d2).mean() + cos_dist(e3, d3).mean()
   ```

### 5.3 Loss Term 2: Cross-Scale Distillation Loss ($\mathcal{L}_2$)
1. **Equation (As Implemented in `train.py` lines 43–45 vs. `metric.py` lines 21–32):**
   - **Code Formula A (`train.py` lines 43–45 — Actual Training Code):**
     $$\mathcal{L}_2^{\text{train}} = \text{mean}\big(0.4\, D_{\cos}(e_1, l_{1n}) + D_{\cos}(e_1, l_{1m})\big) + \text{mean}\big(0.2\, D_{\cos}(e_2, l_{2n}) + D_{\cos}(e_2, l_{2m})\big) + \text{mean}\big(0.4\, D_{\cos}(e_3, l_{3n}) + D_{\cos}(e_3, l_{3m})\big)$$
   - **Code Formula B (`metric.py` lines 21–23, 32 — Symmetric Scale-Weighted Form Used in Anomaly Scoring):**
     $$\mathcal{L}_2^{\text{sym}} = 0.4\,\text{mean}\big(D_{\cos}(e_1, l_{1n}) + D_{\cos}(e_1, l_{1m})\big) + 0.2\,\text{mean}\big(D_{\cos}(e_2, l_{2n}) + D_{\cos}(e_2, l_{2m})\big) + 0.4\,\text{mean}\big(D_{\cos}(e_3, l_{3n}) + D_{\cos}(e_3, l_{3m})\big)$$
   > `[UNRESOLVED-PDF-REQUIRED]` Notice that in `train.py`, `0.4`, `0.2`, `0.4` multiply only the `l_kn` term inside each scale, whereas in `metric.py`, `0.4`, `0.2`, `0.4` multiply the entire scale sum `ac_k = cos_dist(e_k, l_kn) + cos_dist(e_k, l_km)`. Our baseline implementation supports both modes via configuration (`cross_loss_mode: "official_train"` vs. `"symmetric_metric"`), defaulting to `"official_train"` for 100% code-faithful baseline reproduction.
2. **Variables & Shapes:**
   - $l_{km}, l_{kn} \in \mathbb{R}^{B \times C_k \times H_k \times W_k}$: Cross-scale projected student features aligned to teacher scale $k$.
   - Scale weights $\alpha = (0.4, 0.2, 0.4)$: Assigns higher emphasis to the finest scale ($k=1, 64\times64$) and deepest semantic scale ($k=3, 16\times16$).
3. **Purpose:** Regularizes the student decoder so that features decoded at each stage preserve information required by both shallower and deeper teacher scales.
4. **Implementation File & Line:** `train.py`, lines 43–45.

### 5.4 Total Training Objective ($\mathcal{L}_{\text{total}}$)
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 = 2.0 \, \mathcal{L}_1 + 0.1 \, \mathcal{L}_2$$
- **Implementation File & Line:** `train.py`, line 46 (`loss = 2.0 * loss1 + 0.1 * loss2`).

---

## 6. Inference Anomaly Map and Scoring Equations (`metric.py`)

For a test image $I \in \mathbb{R}^{1 \times 3 \times H \times W}$ ($H = W = 256$):

1. **Same-Scale Discrepancy Maps ($k \in \{1, 2, 3\}$):**
   $$a_k = D_{\cos}(e_k, s_k) \in \mathbb{R}^{1 \times H_k \times W_k}, \quad \tilde{a}_k = \text{Interp}_{H \times W, \text{bilinear}}(a_k) \in \mathbb{R}^{1 \times H \times W}$$
2. **Cross-Scale Discrepancy Maps ($k \in \{1, 2, 3\}$):**
   $$ac_k = D_{\cos}(e_k, l_{kn}) + D_{\cos}(e_k, l_{km}) \in \mathbb{R}^{1 \times H_k \times W_k}, \quad \tilde{ac}_k = \text{Interp}_{H \times W, \text{bilinear}}(ac_k) \in \mathbb{R}^{1 \times H \times W}$$
3. **Weighted Multi-Scale Map Fusion (`metric.py` line 32):**
   $$\mathcal{M}_{\text{raw}} = 2.0 \big(\tilde{a}_1 + \tilde{a}_2 + \tilde{a}_3\big) + 0.1 \big(0.4\,\tilde{ac}_1 + 0.2\,\tilde{ac}_2 + 0.4\,\tilde{ac}_3\big) \in \mathbb{R}^{H \times W}$$
4. **Gaussian Spatial Smoothing (`metric.py` line 36):**
   $$\mathcal{M}(h, w) = \big(\mathcal{M}_{\text{raw}} * G_{\sigma=4}\big)(h, w) \in \mathbb{R}^{H \times W}$$
5. **Image-Level Anomaly Score (`metric.py` line 47):**
   $$S_{\text{img}} = \max_{1 \le h \le H,\, 1 \le w \le W} \mathcal{M}(h, w) \in \mathbb{R}$$
