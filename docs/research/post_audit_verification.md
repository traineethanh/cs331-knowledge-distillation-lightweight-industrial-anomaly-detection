# Post-Audit Verification — DSKD Baseline & Research Decisions

**Project:** Knowledge Distillation-Based Lightweight Industrial Anomaly Detection (`CS331`)  
**Repository:** `cs331-knowledge-distillation-lightweight-industrial-anomaly-detection`  
**Branch / HEAD:** `chore/repository-reset` @ `41af7b015a69fe7480c8fd1212e6e22840ae22d0`  
**Role:** Independent Senior ML Research Auditor  
**Verification Date:** `2026-11-14`  
**Baseline Reference:** *Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection* (`papers/2402.00448v1.pdf`, Yao & Gao, Feb 1, 2024)  
**Final Verdict:** **`READY AFTER REQUIRED CORRECTIONS`** *(Implementation must remain paused until the human approves resolution of the remaining architectural, equation, and accounting mismatches identified below.)*

---

## 1. Executive Summary

An independent post-audit verification was conducted to challenge—rather than defend—the conclusions in `docs/research/final_technical_audit.md` and the 10 companion research documents in `docs/research/`. Every architectural claim, mathematical identity, complexity figure, experimental assumption, and Git state claim was re-checked directly against `papers/2402.00448v1.pdf`, the existing repository code in `src/anomaly_detection/`, executable PyTorch reconstructions at `256x256` (`batch_size=1`), and Git CLI outputs.

**Five consequential audit conclusions require correction before implementation begins:**

1. **`B0-Clean` Architectural Equivalence Cannot Be Established (Verdict: `D`):** Because `papers/2402.00448v1.pdf` provides no official source code and omits critical micro-architectural specifications (DFE internal channel widths, whether $F_3$'s `1x1 conv` changes channels, DFE output dimension $256$ vs. $512$, `ConvTranspose2d` vs. bilinear upsampling in $S_d$, and whether `layer4` is present), `B0-Clean` (`12.180M` or `12.098M` actual params vs. `39.30M` reported in Table 6) cannot be proven architecturally equivalent to `B0-Paper`.
2. **Multiple Parameter and FLOPs Accounting Errors in `final_technical_audit.md`:**
   - **Decoder $S_d$ undercounted:** Previous tables copied `S_e`'s parameter count (`2.783M`) for `S_d`. An inverted 3-stage `ResNet18` decoder with symmetric block widths actually has **`3.326M` params (`3,325,824`, bilinear + Conv3x3)** or **`3.244M` (`3,243,904`, ConvTranspose2d)** (`+0.543M` higher at $\alpha=1.0$; likewise `1.872M` vs `1.566M` at $\alpha=0.75$ and `0.832M` vs `0.697M` at $\alpha=0.50$).
   - **`MACs` vs. `FlopCounterMode` FLOPs confused (`2x` factor):** Previous tables reported `~MACs` (`6.379G` for `B0-Clean`) under the label `FLOPs`, whereas `src/anomaly_detection/utils/profiling.py` (`measure_student_flops`, lines 28–41) and `experiment_design.md` specify `torch.utils.flop_counter.FlopCounterMode` (`1 MAC = 2 FLOPs`). Measured `FlopCounterMode` FLOPs for 3-stage `B0-Clean` are **`13.451G–13.485G` FLOPs (`6.726G–6.742G` MACs)** with bilinear `S_d`, or **`12.411G` FLOPs (`6.206G` MACs)** with deconv `S_d`. Meanwhile, the paper's `6.17G` in `Table 6` aligns numerically with **MACs**, not `FlopCounterMode` FLOPs.
   - **Two conflicting `Lite-DFE` designs & false `PW_skip` omission claim:** `method_b_proposal.md` (`d_b = 64*alpha`, concatenated width `3*d_b = 192*alpha`, **`36,736` params / `0.037M`** at $\alpha=0.5, \rho=0.5$) *already included* `PW_skip` (`96 -> 128`). `final_technical_audit.md` silently changed `Lite-DFE` to `768*alpha` concatenation (`dsmsa_expand`, **`104,096` params / `0.104M`** at $\alpha=0.5, \rho=0.5$) and erroneously claimed Agent 4 omitted `PW_skip`. Both variants exist in the docs and must be explicitly disambiguated.
   - **BN buffers & unimplemented measurements:** BatchNorm running buffers (`11.8k–16.3k` elements) were omitted; runtime GPU/CPU latency, Peak VRAM, and `Method A` `FlopCounterMode` overhead (`+0.01G` claimed) are **`NOT VERIFIED`** in code.
3. **Numerical Overstatement in `Eq. (3)` Audit Corrected:** Under `N3` ($\hat{u} = u / (\|u\|_2^2 + \epsilon)$), exact unregularized $\epsilon = 0$ is Kelvin inversion ($\to \infty$ as $\|u\|_2 \to 0$), **not** the implemented regularized case $\epsilon > 0$. With any $\epsilon > 0$, $\hat{u} \to 0$ continuously as $\|u\|_2 \to 0$, and gradients remain **mathematically bounded** ($O(1/\epsilon)$ peak at $\|u\|_2 = \sqrt{\epsilon}$, where $\|\hat{u}\|_2 = \frac{1}{2\sqrt{\epsilon}}$), though still dangerously stiff in FP16/FP32 if $\epsilon$ is tiny (e.g., $10^{-12}$).
4. **Experimental Budget & Pilot Claims Need Revision:** A 50-epoch pilot (`S0_eq3_norm`, `S0_input_norm`) can filter catastrophic divergence/collapse, **not** prove faithful 200-epoch baseline reproduction (`100%` train/good vs `80/20` split, unknown paper `batch_size`). Furthermore, the `~18.2h` / `~19.5h` Kaggle GPU estimate relies on unbenchmarked CPU `AUPRO` loops (`500` thresholds across region masks) and `39.3M` VRAM formulas rather than actual timing runs (`Requires revision`).
5. **Git Audit & Residual Doc Contradictions:** Zero tracked repository files were modified (`git diff --stat` and `git diff --name-status` are both empty against HEAD `41af7b015`). All 11 files in `docs/research/` are **untracked (`??`)**. While prior edits fixed real issues (e.g., swapped Method A/B labels, 4 wrong paper citations), residual contradictions remain between `research_decision.md` (body vs. checklist lines 78–82), `kaggle_implementation_plan.md` (lines 488, 521–528, 538), `method_b_proposal.md`, and `final_technical_audit.md`. **Recommendation:** Retain existing research docs (do **not** revert) and issue targeted corrections after human sign-off.

---

## 2. `B0-Clean` Architecture Verdict

### 2.1. Formal Classification Verdict

> **VERDICT: `D. An implementation whose equivalence cannot yet be established.`**

Matching intermediate spatial shapes (`64x64`, `32x32`, `16x16`) and channel tuples (`64, 128, 256`) does **not** prove architectural equivalence when the reference paper (`papers/2402.00448v1.pdf`) withholds code and omits critical layer definitions.

### 2.2. Component-by-Component Comparison: `B0-Paper` vs. `B0-Clean`

| Component | Original Paper Specification (`papers/2402.00448v1.pdf`) | Proposed `B0-Clean` Implementation (Research Docs) | Observed Equivalence Status / Ambiguity |
| :--- | :--- | :--- | :--- |
| **Feature Extraction Stages ($K$)** | Sec. 3.1 (p. 5): *"Specifically, we select the first three blocks (`conv2_x, conv3_x, conv4_x`) in ResNet18"* ($K=3$); outputs $c_k \in \{64, 128, 256\}$ at $64^2, 32^2, 16^2$ for $256\times 256$ inputs (Sec. 4.1.2, p. 7). | Uses `torchvision.models.resnet18` truncated at `stem + layer1..layer3` (drops `layer4` and `fc`). | **Truncated match for active graph**, but paper's `39.3M` params / `105MB` memory strongly implies `layer4` (and possibly full ResNet18) remained in the paper's instantiated objects (`Sec. 4.2.4, Table 6, p. 10`). |
| **Teacher ($T$)** | Sec. 3.1 (p. 5): ImageNet pre-trained ResNet18, frozen parameters. Algo. 1/2 take `T(I)` multi-scale features. | `torchvision.models.resnet18(IMAGENET1K_V1)`, `stem + layer1..3` = **`2,782,784` params** (`4,495` BN buffers), `3.664G` FlopCounter FLOPs (`1.832G` MACs). Full `stem..layer4` = `11,176,512` params (+`fc` = `11,689,512`). | **Width/stage ambiguity:** Active `T` stages match `stem..layer3`, but whether paper kept unused `layer4` (`+8.394M` params) cannot be confirmed without code. |
| **Student Encoder ($S_e$)** | Sec. 3.1 (p. 5) & Fig. 3 (p. 3): Same ResNet18 backbone as $T$, randomly initialized, trainable ($\theta_e$). | Same `stem + layer1..3` topology = **`2,782,784` params** (`4,495` buffers), `3.664G` FlopCounter FLOPs (`1.832G` MACs). | **Matches $T$ truncation choice**, subject to the same `layer4` and `dfe` gradient-detachment question below. |
| **Deep Feature Embedding (`DFE`)** | Fig. 4 (p. 5) shows: $F_1 \to 2\times \text{conv }3\times 3$ (with bn+relu), $F_2 \to 1\times \text{conv }3\times 3$ (with bn+relu), **$F_3 \to 1\times \text{conv }1\times 1$ (with bn+relu)** $\to \text{concat} \to$ Residual Block (`conv3x3`, `conv3x3` + `conv1x1` shortcut) $\to F_{\text{emb}}$. Algo. 2 line 7 prints `feature_embedding(\hat{F}_e)` vs. Algo. 1 narrative. **Zero channel widths are specified anywhere.** | `final_technical_audit.md` assumes `DenseDFE` **without** the $F_3$ `1x1 conv` (`3,222,272` params) expanding channels `64->128->256` and `128->256` (`768 -> 256` residual block). Adding `F_3` `conv1x1(256->256)` yields `3,288,320` params; RD4AD-style `out=512` yields `6,959,360`–`7,025,408` params; non-expanding `64, 128, 256 -> 448` yields `1,959,936`–`2,025,984` params. | **Cannot establish equivalence:** (1) $F_3$ `1x1 conv` in Fig. 4 was omitted in `3.222M` `DenseDFE`; (2) concat width (`448` vs `768`) and bottleneck $F_{\text{emb}}$ width (`256` vs RD4AD's `512`) are unstated in the paper; (3) `dfe([f.detach()])` in `A2` conflicts with Algo. 1 if paper allowed gradient flow from $\ell_d$ through `DFE` into $S_e$. |
| **Student Decoder ($S_d$)** | Sec. 1 (p. 2): *"completely reversed from $S_e$"*. Sec. 3.1 (p. 5): *"completely transposed backbone, where downsampling operations are replaced by upsampling operations."* No block channel transition (`convTranspose` vs `bilinear+conv`) or layer block counts are printed. | `final_technical_audit.md` claims `2.783M` params (`S_d == S_e`). An actual 3-stage reversed ResNet18 (`256->256` in `rev_layer3`, `256->128` in `rev_layer2`, `128->64` in `rev_layer1`, 2 BasicBlocks/stage) has **`3,325,824` params (`4.362G` FLOPs)** with `bilinear + Conv3x3`, or **`3,243,904` params (`3.288G` FLOPs)** with `ConvTranspose2d(2x2, s=2)`. (Note: omitting stem deconv back to $256\times 256$, which stops at $k=1: 64\times 64$.) | **Cannot establish equivalence:** Neither upsampling operator nor reversed stage channel counts (`rev_layer3` `256->256` vs `512->256` like RD4AD) is defined in the PDF, and `B0-Clean` docs miscounted `S_d` by `-0.543M` (bilinear) or `-0.461M` (deconv). |
| **Projection Layers ($\mathcal{P}_{e,d}^k$)** | Absent (`0` params; $S_e, S_d$ output $\{64, 128, 256\}$ directly). | Absent in `B0-Clean` (`0` params). *(Added only in Method B `Slim-DSKD`: `Conv1x1 + BN (+ act)`.)* | **Equivalent (`0` params in baseline).** |
| **Total Parameters** | **`39.30M`** (`Table 6, p. 10`), `105 MB` memory. | Claimed **`11.571M`** (`final_technical_audit.md` line 252). Actual 3-stage `B0-Clean` PyTorch count: **`12.180M`** (`f3c=True, exp256, bilinear`), **`12.098M`** (`deconv`), or **`12.114M`** (`no-f3c, bilinear`). | **Large gap (`~27.1M–27.7M`, `-69%` to `-70.6%`)**, analyzed below. |
| **FLOPs (`256x256`, `bs=1`)** | **`6.170G`** (`Table 6, p. 10`). | Claimed **`6.379G`** (`final_technical_audit.md` line 252). Actual PyTorch `FlopCounterMode` FLOPs: **`13.485G`** (`6.742G` MACs, bilinear `f3c=True`) or **`12.411G`** (`6.206G` MACs, deconv `f3c=True`). | **Unit mismatch + `S_d` mismatch:** Paper `6.170G` matches **MACs** (e.g., `deconv f3c=True` = `6.206G` MACs, within `0.6%`), NOT `2x` `FlopCounterMode` FLOPs. |

### 2.3. Exact Mathematical Explanation of `39.30M` vs. `11.571M` / `12.180M`

A single `torchvision.models.resnet18` has:
- `stem (conv1 + bn1)` + `layer1..layer3`: **`2,782,784` (`2.783M`)** params
- `layer4` (`256 -> 512`, two BasicBlocks): **`8,393,728` (`8.394M`)** params (`75.1%` of `stem..layer4`!)
- `stem..layer4` (without `fc`): **`11,176,512` (`11.177M`)** params (or `11,689,512` with `fc`)

Why the `~27.1M`–`27.7M` difference occurs—and why **truncating `layer4` alone does NOT completely explain `39.30M`**:
1. Retaining `layer4` in three ResNet18-scale backbones (`T`, `S_e`, and a 4-stage `S_d`) accounts for **`3 x 8.394M = 25.18M`** (actually `S_d` with a symmetric 4th stage `512->512` + `512->256` downsampling inverse would differ slightly; e.g., $T_{\text{l1..4}} + S_{e,\text{l1..4}} + S_{d,\text{4-stage}} \approx 33.5\text{M}–35.7\text{M}$).
2. Crucially, `3 x 11.177M = 33.530M`, leaving `39.30M - 33.53M = 5.77M` for `DFE` (exactly what `method_b_proposal.md` line 232 back-solved: `39.300 - 3*11.177 = 5.769M`). However, an actual `DenseDFE` outputting `512` channels (`RD4AD`-style `F_emb` feeding `rev_layer4`) has `4.878M` (`448->512`) to `7.025M` (`768->512`), yielding **`~38.41M`–`40.56M`** total—squarely brackets **`39.30M`**!
3. Even more tellingly: why did `B0-Paper` have **`39.30M` params** (`3.24x` `12.1M`) yet only **`6.17G` FLOPs** (virtually identical to 3-stage **`6.21G` MACs**)? Because `layer4` operates at $8\times 8$ spatial resolution (`1.074G` FLOPs / **`0.537G` MACs** per ResNet18), and if `T/S_e/S_d` instantiated full 4-stage backbones (`+25.18M` params) in `count_parameters` while inference only routed 3 stages (or included `layer4` `~1.6G` MACs alongside a `448`-ch DFE `~0.53G` MACs), params jump `>3x` while MACs stay near `6.17G`! Thus, without `B0-Paper`'s code, **`B0-Clean` is a clean 3-stage PyTorch reconstruction whose architectural equivalence to `B0-Paper` remains unestablished (`D`).**

---

## 3. `Eq. (3)–(6)` Verification

### 3.1. Direct Formula Audit (`papers/2402.00448v1.pdf`, pp. 5–7)

Let $u = F_t^k(I_m)_{ij} \in \mathbb{R}^{c_k}$ and $v = F_s^k(I_m)_{ij} \in \mathbb{R}^{c_k}$ ($s \in \{e, d\}$) with Euclidean norms $r_u = \|u\|_2$, $r_v = \|v\|_2$, and cosine similarity $\cos\theta = \frac{u^\top v}{r_u r_v}$ (for $r_u, r_v > 0$). Under the regularized implementations tested in the audit (`N1`, `N2`, `N3`, with $\epsilon > 0$):

1. **`N1` (Unit-$\ell_2$ Normalization):**
   $$\hat{u}_{N1} = \frac{u}{\|u\|_2 + \epsilon} \quad (\text{or } \frac{u}{\max(\|u\|_2, \epsilon)}), \qquad \hat{v}_{N1} = \frac{v}{\|v\|_2 + \epsilon}$$
   - **Eq. (4) & Eq. (5):** When $r_u, r_v \gg \epsilon$, $\|\hat{u}_{N1}\|_2 \approx 1$ and $\|\hat{v}_{N1}\|_2 \approx 1$, so:
     $$\ell_2^{(N1)} = \frac{1}{2}\|\hat{u}_{N1} - \hat{v}_{N1}\|_2^2 \approx 1 - \cos\theta \equiv \ell_{\cos}^{(N1)}, \qquad M^{(N1)} = \lambda \ell_2^{(N1)} + \ell_{\cos}^{(N1)} \approx (1+\lambda)(1-\cos\theta) \in [0, 2(1+\lambda)]$$
   - **Behavior as $\|v\|_2 \to 0$ with $\epsilon > 0$:** With `+ \epsilon` in denominator, $\hat{v}_{N1} \to 0$ continuously (`0 / \epsilon = 0`), so $\ell_2^{(N1)} \to \frac{1}{2}\|\hat{u}_{N1}\|_2^2 \le \frac{1}{2}$. For $\ell_{\cos}$, denominator $(\|\hat{u}\|_2 + \epsilon)(\|\hat{v}\|_2 + \epsilon)$ (or `F.cosine_similarity(..., eps=1e-8)`) prevents division by zero (`1 - 0 = 1`).
   - **Gradient bound for $\epsilon > 0$:** Differentiating $\hat{v}_{N1} = v / (r_v + \epsilon)$ gives Jacobian $\frac{1}{r_v+\epsilon}I - \frac{v v^\top}{r_v(r_v+\epsilon)^2}$. Because $\frac{v v^\top}{r_v}$ is homogeneous of degree $1$ with spectral norm $r_v \to 0$, **the Jacobian norm is bounded by $\frac{1}{r_v + \epsilon} \le \frac{1}{\epsilon}$** (or for $\max(r_v, \epsilon)$, identically $I/\epsilon$ inside $r_v < \epsilon$). Hence $\|\nabla_v \ell_2^{(N1)}\|_2$ is **strictly finite ($O(1/\epsilon)$), NOT infinite**. However, setting $\epsilon = 10^{-12}$ still allows $O(10^{12})$ gradient spikes if $r_v \approx 10^{-12}$, whereas $\epsilon \in [10^{-8}, 10^{-6}]$ keeps gradients well-behaved in `FP32`/`AMP`.

2. **`N2` (Raw-$\ell_2$ on Unnormalized $F$ + Normalized Cosine, MKD-style):**
   $$\ell_2^{(N2)} = \frac{1}{2}\|u - v\|_2^2, \qquad \ell_{\cos}^{(N2)} = 1 - \frac{u^\top v}{\max(\|u\|_2 \|v\|_2, \epsilon)}, \qquad M^{(N2)} = \lambda \ell_2^{(N2)} + \ell_{\cos}^{(N2)}$$
   - **Why `N2` is textually plausible:** Explains why `Eq. (5)` retains $\|\hat{F}_t\|\|\hat{F}_e\|$ in its denominator and why `Eq. (6)` uses $\lambda = 0.1$ (balancing raw squared feature scale vs bounded cosine `[0, 2]`).
   - **Behavior as $\|v\|_2 \to 0$ with $\epsilon > 0$:** $\ell_2^{(N2)} \to \frac{1}{2}\|u\|_2^2$ with **exact smooth gradient $\nabla_v \ell_2^{(N2)} = v - u \to -u$** (zero singularity regardless of $\epsilon$). The cosine term retains standard $O(1/\epsilon)$ bound inside the $\epsilon$-ball when `eps` stabilizes `r_v`.

3. **`N3` (Literal `Eq. 3` with $\epsilon > 0$ Safeguard):**
   $$\hat{u}_{N3} = \frac{u}{\|u\|_2^2 + \epsilon}, \qquad \hat{v}_{N3} = \frac{v}{\|v\|_2^2 + \epsilon}$$
   - **Crucial correction to `final_technical_audit.md` (lines 45, 59, 94):** Previous audit prose conflated exact unregularized $\epsilon = 0$ (Kelvin spherical inversion: $\|\hat{v}\|_2 = 1/r_v \to \infty$, $\|\nabla_v \ell_2\|_2 \propto r_v^{-3} \to \infty$) with numerical implementation $\epsilon > 0$:
     - **With $\epsilon = 0$ (math singularity):** $r_v \to 0 \implies \|\hat{v}\|_2 \to \infty$ and $\nabla_v \ell_2 \to \infty$.
     - **With ANY $\epsilon > 0$ (actual PyTorch implementation):** As $r_v \to 0$, $r_v^2 + \epsilon \to \epsilon > 0$, so **$\hat{v}_{N3} \to 0$**, $\|\hat{v}_{N3}\|_2 = \frac{r_v}{r_v^2 + \epsilon}$ is **strictly bounded** on $[0, \infty)$! Its global maximum occurs at $r_v = \sqrt{\epsilon}$, where $\|\hat{v}_{N3}\|_{\max} = \frac{1}{2\sqrt{\epsilon}}$ (`50.0` for $\epsilon = 10^{-4}$; `5e5` for $\epsilon = 10^{-12}$), and $\nabla_v \hat{v}_{N3} = \frac{1}{r_v^2+\epsilon}I - \frac{2 v v^\top}{(r_v^2+\epsilon)^2}$ has operator norm **bounded by $\frac{1}{\epsilon}$** at $r_v = 0$ (and $\ell_2^{(N3)}$ reaches $\approx \frac{1}{8\epsilon}$ peak, **not $\infty$**!).
     - Despite finite bounds for $\epsilon > 0$, `N3` still severely distorts anomaly weighting: large activations ($r_v = 10$) are crushed (`0.1`) while near-zero activations ($r_v \approx \sqrt{\epsilon}$) explode by $\frac{1}{2\sqrt{\epsilon}}$, reversing spatial emphasis (`Eq. 4` $\not\equiv$ `Eq. 5`) and causing extreme conditioning issues unless $\epsilon$ is artificially tuned ($\sim 10^{-4}$–$10^{-1}$).

### 3.2. Separation of Feature Normalization (`Eq. 3`) vs. Input Image Normalization (`Sec. 4.1.2`)

- **`Eq. (3)` (p. 5) + Algo. 1 (lines 7–9, p. 4) + Algo. 2 (lines 5–8, p. 6):** Applied **channel-wise per spatial location $(i,j)$** to extracted CNN feature tensors $F_t^k, F_e^k, F_d^k$.
- **Sec. 4.1.2 (p. 7):** *"The input images are firstly resized, $256 \times 256$ for MVTec AD and MVTec 3D-AD... The resized images are then normalized by $\ell_2$ function."* Applied to **RGB input images** before entering frozen `ResNet18` $T$.
- **Consequence:** These are two independent axes (`3 x 2` conceptual design space if crossed, though currently planned as separate 1D ablations `S0_eq3_norm` and `S0_input_norm`).

### 3.3. Unresolved Preprocessing & Loss Decisions

1. **Choice among `N1` / `N2` / `N3`:** Remains unresolved pending `S0_eq3_norm`; `research_decision.md` contradicts itself (line 26 orders testing `N1/N2/N3`; line 78 locks `N1` unconditionally).
2. **Exact meaning of Sec. 4.1.2 *"normalized by $\ell_2$ function"*:** Could mean per-image tensor $\ell_2$ normalization ($I / \|I\|_2$), per-pixel 3-channel $\ell_2$ normalization, or author shorthand for `Eq. (3)` alongside standard ImageNet Z-score (`src/anomaly_detection/data/mvtec.py`, lines 39–55). `S0_input_norm` compares ImageNet Z-score vs. `P_img_l2`, but `P_img_l2`'s tensor axis (whole image vs. channel-wise vs. spatial) is not locked in code/spec!
3. **Exact input to `DFE` (`Algo. 1` vs `Algo. 2` line 7 + gradient detachment):** `Algo. 2` line 7 writes `F_emb <- feature_embedding(\hat{F}_e)` (normalized features), whereas Sec. 3.2/Fig. 4 and `experiment_design.md` / `final_technical_audit.md` pass `raw_f_e` detached (`[f.detach() for f in raw_f_e]`). Whether baseline `B0-Clean` uses raw vs normalized features into `DFE` and detached vs non-detached $S_e$ gradients into `DFE` must be explicitly verified in `G0`.

---

## 4. Parameter / FLOPs Verification

All counts below were independently executed and verified in PyTorch (`torchvision.models.resnet18`, `256x256` input, `batch_size=1`, `torch.utils.flop_counter.FlopCounterMode`, zero bias on Conv before BN).

### 4.1. Independently Verified Component Breakdown

| Component (`ResNet18` 3-stage, `256x256`, `bs=1`) | Total Params | Trainable Params | BN Buffers | `FlopCounter` FLOPs | `MACs` (`FLOPs / 2`) | Source / Audit Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Teacher $T$ (`stem + layer1..3`, frozen)** | `2,782,784` | `0` | `4,495` | `3.664G` (`3,663,724,544`) | `1.832G` | `torchvision.models.resnet18` (`layer4`: `8,393,728` params, `5,125` buf, `1.074G` FLOPs; `fc`: `513,000` params) |
| **Student Enc $S_e$ ($\alpha=1.00$)** | `2,782,784` | `2,782,784` | `4,495` | `3.664G` | `1.832G` | Matches `2.783M`; prior table wrote `1.816G` MACs |
| **Student Enc $S_e$ ($\alpha=0.75$)** | `1,567,920` | `1,567,920` | `3,375` | `2.119G` | `1.059G` | Prior table wrote `1.566M` (`48,96,192`) / `0.961G` |
| **Student Enc $S_e$ ($\alpha=0.50$)** | `699,168` | `699,168` | `2,255` | `0.993G` | `0.496G` | Prior table wrote `0.697M` / `0.371G` or `0.456G` |
| **Student Dec $S_d$ ($\alpha=1.00$, Bilinear+3x3)** | **`3,325,824`** | **`3,325,824`** | `3,982` | **`4.362G`** | **`2.181G`** | **Audit error:** prior tables wrote `2.783M` / `1.816G` (`-0.543M` undercount!) |
| **Student Dec $S_d$ ($\alpha=1.00$, ConvT 2x2)** | **`3,243,904`** | **`3,243,904`** | `3,982` | **`3.288G`** | **`1.644G`** | Alternative paper upsampling interpretation |
| **Student Dec $S_d$ ($\alpha=0.75$, Bilinear / ConvT)** | `1,871,520` / `1,825,440` | same | `2,990` | `2.454G` / `1.850G` | `1.227G` / `0.925G` | Prior table wrote `1.566M` (`-0.306M` undercount!) |
| **Student Dec $S_d$ ($\alpha=0.50$, Bilinear / ConvT)** | `832,448` / `811,968` | same | `1,998` | `1.091G` / `0.822G` | `0.545G` / `0.411G` | Prior table wrote `0.697M` (`-0.135M` undercount!) |
| **`DenseDFE` (`no F3 conv1x1`, `768->256`)** | `3,222,272` | `3,222,272` | `2,822` | `1.762G` | `0.881G` | Used by `final_technical_audit.md` (`3.222M`) |
| **`DenseDFE` (`+F3 1x1 conv`, `768->256`, Fig. 4)** | `3,288,320` | `3,288,320` | `3,335` | `1.795G` | `0.898G` | Matches Fig. 4's `1x1 conv` box on $F_3$ |
| **`DenseDFE` (`+F3 1x1`, `768->512` RD4AD-width)** | `7,025,408` | `7,025,408` | `4,871` | `3.708G` | `1.854G` | Explains `39.30M` when paired with 4-stage `T/Se/Sd` |
| **`Lite-DFE` (`dsmsa_db`, $\alpha=1.0, \rho=0.5$)** | `141,056` | `141,056` | `2,442` | `0.072G` | `0.036G` | `method_b_proposal.md` (`3*d_b=192`); **`0.037M` at $\alpha=0.5, \rho=0.5$ (`36,736`) ALREADY included `PW_skip`** |
| **`Lite-DFE` (`dsmsa_db`, $\alpha=0.5, \rho=1.0 / 0.5$)** | `51,904` / `36,736` | `51,904` / `36,736` | `1,482` / `1,226` | `0.026G` / `0.019G` | `0.013G` / `0.009G` | `51,904` vs `137,696` gap was hidden in prior audit! |
| **`Lite-DFE` (`dsmsa_expand`, $\alpha=1.0, \rho=0.5$)** | `408,896` | `408,896` | `3,466` | `0.221G` | `0.111G` | `final_technical_audit.md` variant (`768*alpha` concat) |
| **`Lite-DFE` (`dsmsa_expand`, $\alpha=0.5, \rho=1.0 / 0.5$)** | `137,696` / `104,096` | `137,696` / `104,096` | `1,994` / `1,738` | `0.073G` / `0.056G` | `0.037G` / `0.028G` | Note: `final_technical_audit.md` line 212 wrote `141,120` (`64` off from `137,696` due to mistaking `PW` bias/dims) |
| **Proj Heads $\mathcal{P}_{e,d}^{1..3}$ (2 branches, $\alpha=0.75 / 0.50$)** | `130,816` / `87,808` | same | `1,798` / `1,798` | `0.151G` / `0.101G` | `0.076G` / `0.050G` | Half (`65,408` / `43,904` params; `0.076G` / `0.050G` FLOPs) for 1 branch (`P_d` at standalone inference) |

### 4.2. Full Reconstructed Model Variants (`256x256`, `batch_size=1`)

| Model Variant | Total Params | Trainable Params ($S_e + \text{DFE} + S_d + \mathcal{P}$) | Inference Params (Active Forward) | BN Buffers | `FlopCounter` FLOPs (Train / Inf) | `MACs` (Train / Inf) | vs. Claimed in `final_technical_audit.md` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B0-Clean` (`no-f3c`, Bilinear $S_d$)** | **`12.114M`** (`12,113,664`) | **`9.331M`** (`9,330,880`) | `12.114M` | `15,794` | `13.451G` / `13.451G` | `6.726G` / `6.726G` | Audit claimed `11.571M` / `8.788M` / `6.379G` |
| **`B0-Clean` (`+F3 1x1`, Bilinear $S_d$)** | **`12.180M`** (`12,179,712`) | **`9.397M`** (`9,396,928`) | `12.180M` | `16,307` | **`13.485G` / `13.485G`** | `6.742G` / `6.742G` | Most faithful to Fig. 4 + bilinear $S_d$ |
| **`B0-Clean` (`+F3 1x1`, ConvT $S_d$)** | **`12.098M`** (`12,097,792`) | **`9.315M`** (`9,315,008`) | `12.098M` | `16,307` | `12.411G` / `12.411G` | `6.206G` / `6.206G` | `6.206G` MACs matches paper `6.170G` (`+0.6%`) |
| **`B1` (`Lite-DFE`, `dsmsa_db`, $\alpha=1.0, \rho=0.5$, bi)** | `9.032M` (`9,032,448`) | `6.250M` (`6,249,664`) | `9.032M` | `15,414` | `11.761G` / `11.761G` | `5.881G` / `5.881G` | — |
| **`B1` (`Lite-DFE`, `dsmsa_ex`, $\alpha=1.0, \rho=0.5$, bi)** | `9.300M` (`9,300,288`) | `6.518M` (`6,517,504`) | `9.300M` | `16,438` | `11.911G` / `11.911G` | `5.955G` / `5.955G` | Audit claimed `8.758M` / `5.975M` / `4.949G` |
| **`B3-0.75` (`dsmsa_db`, $\rho=0.5$, bi)** | `6.433M` (`6,433,488`) | `3.651M` (`3,650,704`) | `6.368M` *(skip $\mathcal{P}_e$)* | `14,492` | `8.428G` / `8.352G` | `4.214G` / `4.176G` | — |
| **`B3-0.75` (`dsmsa_ex`, $\rho=0.5$, bi)** | `6.584M` (`6,584,448`) | `3.802M` (`3,801,664`) | `6.519M` *(skip $\mathcal{P}_e$)* | `15,260` | `8.512G` / `8.436G` | `4.256G` / `4.218G` | Audit claimed `6.240M` / `3.457M` / `3.838G` |
| **`B3-0.50` (`dsmsa_db`, $\rho=1.0$, bi)** | **`4.454M`** (`4,454,112`) | **`1.671M`** (`1,671,328`) | `4.410M` *(skip $\mathcal{P}_e$)* | `12,028` | **`5.874G` / `5.824G`** | `2.937G` / `2.912G` | **`-82.2%` trainable**, **`-56.4%` FLOPs** vs `B0-Clean (f3c, bi)` |
| **`B3-0.50` (`dsmsa_ex`, $\rho=1.0$, bi)** | **`4.540M`** (`4,539,904`) | **`1.757M`** (`1,757,120`) | `4.496M` *(skip $\mathcal{P}_e$)* | `12,540` | **`5.921G` / `5.871G`** | `2.960G` / `2.935G` | Audit claimed `4.369M` / `1.586M` / `2.658G` |
| **`C1-0.50` (`dsmsa_db`, $\rho=0.5$, bi, `FC-Lite`)** | `4.439M` (`4,438,944`) | `1.656M` (`1,656,160`) | `4.439M` *(needs $\mathcal{P}_{e,d}$)* | `11,772` | `5.867G` / `5.867G` | `2.933G` / `2.933G` | Only `36.7k` `Lite-DFE` |
| **`C1-0.50` (`dsmsa_ex`, $\rho=0.5$, bi, `FC-Lite`)** | `4.506M` (`4,506,304`) | `1.724M` (`1,723,520`) | `4.506M` *(needs $\mathcal{P}_{e,d}$)* | `12,284` | `5.904G` / `5.904G` | `2.952G` / `2.952G` | Note: `C1` with `deconv` `S_d` is `5.598G`–`5.636G` |

**Unverified Claims (Explicitly Marked `NOT VERIFIED`):**
- **`Method A` (`FC-DSKD`) inference/training overhead (`+0.16%` / `+0.01G` FLOPs):** **`NOT VERIFIED`** — `Method A` losses and `DSKDAnomalyScorer` (`AvgPool3x3`, Gaussian filter $\sigma=4$, bilinear upsampling, `sqrt` fusion) are not yet implemented and have never been profiled with `FlopCounterMode`.
- **Actual GPU/CPU Latency (`ms/img`) and Peak GPU VRAM (`MB`):** **`NOT VERIFIED`** — No DSKD model code exists in `src/anomaly_detection/` to benchmark on GPU or CPU. (`kaggle_implementation_plan.md` lines 488–499 derived VRAM from `39.3M` params rather than `12.18M`, making its static weight/Adam breakdown over-estimated by `3x`.)
- **Existing `profiling.py` bug:** `src/anomaly_detection/utils/profiling.py` (`measure_student_flops`, lines 32–34) is hardcoded for `(256, 512, 1024)` student feature tensors and cannot profile `DSKDModel` without adding a full-image profiler.

---

## 5. Experiment Protocol Verification

| Experimental Claim / Protocol Element | Verification Status | Justification & Conditions Observed in Repo / PDF |
| :--- | :---: | :--- |
| **`Core-5` Subset + `All-15` Two-Stage Protocol & Anti-Overclaiming Rule** (`experiment_design.md`, Sec. 2.1–2.4) | **`Technically justified`** | Selecting `carpet, tile, bottle, screw, toothbrush` directly targets paper-stated failure modes (`tile, toothbrush` cited in `2402.00448v1.pdf` Sec. 4.2.1, p. 8) plus tiny-sample stress (`toothbrush`: 48 train / 12 val), while keeping Core-5 subset means (`98.88 / 98.18 / 94.20`) strictly separate from Full-15 means (`98.5 / 97.6 / 93.5`). |
| **`val/good` Calibration Protocol & Zero Test Leakage** (`mvtec.py` `80/20` `seed=42`, $\sigma_{\min}=10^{-3}$) | **`Technically justified`** (for leak prevention) / **`Reasonable but unverified`** (for `toothbrush` stability) | `MVTecTrainDataset` (`mvtec.py`, lines 112–150) enforces one-class purity and deterministic `80/20` split (`Zero Train-Test` and `Train-Val` overlap). However, `12` val images in `toothbrush` yield `3,072` spatial patches at $k=3$ that are *spatially correlated within only 12 images*, so small-sample image-level variance of $(\mu_s^k, \sigma_s^k)$ remains unverified until `S0_split` and Core-5 calibration run. |
| **200-Epoch Baseline (`Adam, lr=0.001, beta=(0.5, 0.999), lambda=0.1`)** (`experiment_design.md`, Sec. 3.3; PDF Sec. 4.1.2, p. 7) | **`Technically justified`** *(hyperparams)* / **`Reasonable but unverified`** *(split & batch size)* | Optimizer, betas, learning rate, $\lambda=0.1$, and 200 epochs match `2402.00448v1.pdf` Sec. 4.1.2 (p. 7) verbatim. However, `batch_size = 16` and training on `80%` (`train/good`) are unverified design choices because the paper omits `batch_size` and trains on `100%` `train/good`; `S0_split` is planned but currently missing from Kaggle GPU Phase 1 (`kaggle_implementation_plan.md` line 521). |
| **50-Epoch Sanity Checks (`S0_eq3_norm`, `S0_input_norm`)** | **`Requires revision`** *(interpretation/scope)* | **Why a 50-epoch pilot CANNOT prove full paper reproduction:** At 50/200 epochs on only 2 categories (`carpet`, `bottle`) with an `80/20` split, optimization is only 25% complete, slow convergence of Euclidean amplitude matching vs cosine direction has not settled, hard categories (`screw`, `transistor`, `toothbrush`, `tile`) are untested, and the unspecified paper `batch_size` + `S_d`/`DFE` micro-architecture remain uncalibrated. A 50-epoch pilot only detects catastrophic failure (`NaN`, gradient blow-up, representational collapse, or broken ImageNet `BN` input scale)—it **cannot** establish baseline equivalence (`B0-Clean` vs `B0-Paper` requires full 200-epoch Stage 1/2 evaluation). |
| **GPU Time Budget (`~18.2h` / `~19.5h`, `~6.0–7.0 min/cat`, CPU `AUPRO ~45–75s`)** (`kaggle_implementation_plan.md`, Sec. 4.2–4.3) | **`Requires revision`** | Based entirely on unmeasured heuristics (`0.09–0.12s/iter`, `45–75s` CPU `compute_aupro`). In `src/anomaly_detection/evaluation/metrics.py` (lines 109–133), `compute_aupro` runs Python loops with a per-region list comprehension across up to `~500` thresholds *twice* (`max_fpr=0.30` and `0.05`) per category, whose true wall-clock cost on `256x256` masks is `NOT VERIFIED` (`run` benchmark needed on 1 mock category first). Moreover, Phase 1–6 counts in Sec. 4.3 (`15` runs in Phase 3 instead of `20` for `A1a, A1, A2, A1+A2`; missing `S0_split`; `C1` single vs `{0.75, 0.50}`) do not match `experiment_design.md` Sec. 4.3! |

---

## 6. Git Change Audit

### 6.1. Exact Repository & Branch State

- **Branch:** `chore/repository-reset` (tracking `origin/chore/repository-reset`, up to date)
- **HEAD Commit:** `41af7b015a69fe7480c8fd1212e6e22840ae22d0` (`chore: reset repository to minimal foundation for Dual-Student KD (arXiv:2402.00448v1)`)
- **`git diff --stat` & `git diff --name-status` vs. `HEAD`:** **Empty** — **zero tracked repository files** (`README.md`, `configs/`, `src/`, `tests/`, `scripts/`, `docs/experiment_protocol.md`, `docs/research_plan.md`, `archive/`) have been modified.
- **`git status -uall`:** Only `docs/research/` contains untracked (`??`) files (**11 files prior to this report**, now 12 with `post_audit_verification.md`):

| File (`docs/research/`) | Git Status | Last Write Time (Local) | Provenance & Purpose | Consistency Check & Residual Errors Found |
| :--- | :---: | :---: | :--- | :--- |
| `dskd_paper_analysis.md` | `??` Untracked | `8:31:38 PM` | Round-1 Paper Analyst output (untouched by prior audit). | Assumes $\ell_2 \equiv \ell_{\cos}$ and `39.3M` baseline (superseded by audit/post-audit). |
| `literature_review.md` | `??` Untracked | `8:37:17 PM` | Round-1 Literature output (untouched; titles match PDFs). | Consistent with verified PDF headers. |
| `final_technical_audit.md` | `??` Untracked | `9:09:50 PM` | Newly authored prior audit report. | Contains `S_d` param/FLOPs undercount, MACs/FLOPs `2x` confusion, `dsmsa_db` vs `dsmsa_expand` conflation, `N3` $\infty$-gradient overstatement, and `READY FOR IMPLEMENTATION` overreach. |
| `research_gap_analysis.md` | `??` Untracked | `9:16:44 PM` | Modified after Round 1 to fix 4 fabricated paper titles/authors. | Valid citation fixes retained. |
| `experiment_design.md` | `??` Untracked | `9:16:48 PM` | Modified to fix swapped `Method A/B` labels, split `S0_eq3_norm` / `S0_input_norm` / `S0_split`, and lock `A/B/C1` matrix. | Still carries `8.788M` trainable baseline and `3.457M`/`1.586M`/`-58.3%` `Method B` complexity numbers from `final_technical_audit.md`. |
| `kaggle_implementation_plan.md` | `??` Untracked | `9:22:55 PM` | Modified to fix swapped `Method A/B` labels and upgrade `CHECKPOINT_SCHEMA_V2`. | **Contradictions remain:** Sec. 4.1 (p. 488) still computes VRAM from `39.3M` params; Sec. 4.3 Phase 1–6 still uses stale `A_Cfg*` names, omits `S0_split`, counts `15` vs `20` Method A models; line 538 still refers to *"biến thể B2 (hiệu chuẩn đa nhánh trên val/good)"*! |
| `method_comparison.md` | `??` Untracked | `9:22:58 PM` | Modified to align `Method A/B` definitions and audit fixes. | Inherits `final_technical_audit.md` complexity numbers. |
| `research_decision.md` | `??` Untracked | `9:23:01 PM` | Modified to align audit conditions & marks status **`READY FOR IMPLEMENTATION`**. | **Internal self-contradiction:** Line 26 mandates testing `S0_eq3_norm` (`N1/N2/N3`), whereas **Line 78** says unconditionally *"install unit-L2 in Eq. (3)"*; Line 55 mandates 3-way projection activation ablation, whereas **Line 82** hardcodes `ReLU(BN(Conv1x1))` and quotes stale `82.7%` params / `54.4%` FLOPs (vs `82.0%` / `58.3%` in line 57)! |
| `project_timeline.md` | `??` Untracked | `9:23:05 PM` | Modified to align timeline/hours (`~19.5h` / `~18.2h`) and `A/B` labels. | Inherits unverified timing/count discrepancies from `kaggle_implementation_plan.md`. |
| `method_a_proposal.md` | `??` Untracked | `9:23:07 PM` | Modified to lock `A1/A2/A3` fixes (`DFE` input detach, `Synergistic Fusion`, $\rho_{\text{tail}}=0.001$). | Retains claim `+0.16%` FLOPs without code profiling (`NOT VERIFIED`). |
| `method_b_proposal.md` | `??` Untracked | `9:23:09 PM` | Modified to add audit baseline column `B0-Clean` (`11.571M`). | **Self-contradiction inside lines 232–240 vs footnote line 241:** Body uses `dsmsa_db` (`36,736` / `0.037M` at $\alpha=0.5, \rho=0.5$ — NOT `0.141M`), yet table row `B3-0.50` copied `0.141M` from `dsmsa_expand` (`final_technical_audit.md`), and table uses wrong `S_d = S_e` (`2.783M` / `0.697M`) and `MACs` labeled `FLOPs`. |

### 6.2. Git Audit Verdict (Retain vs. Revert)

- **No tracked files were harmed.** There is nothing in Git tracked history to revert (`HEAD == 41af7b015`).
- **Recommendation: Retain the untracked research files (`do NOT delete/revert`)**, because prior edits resolved critical structural blockers (Method A/B label swap, citation errors, `DFE` detachment policy, `CHECKPOINT_SCHEMA_V2`), **then apply a focused doc synchronization pass** after human decisions below to eliminate the stale checklists and complexity numbers before coding.

---

## 7. Remaining Critical Issues (Must Be Resolved Before Implementation)

1. **`Lite-DFE` Design Split (`dsmsa_db` vs. `dsmsa_expand`) Must Be Chosen:**
   - Option 1 (`method_b_proposal.md`, `dsmsa_db`): Pointwise reduction per branch to `d_b = 64*alpha` $\implies$ `3*d_b = 192*alpha` concat $\implies$ `141k` ($\alpha=1.0$), **`51.9k` ($\alpha=0.5, \rho=1.0$)**, **`36.7k` ($\alpha=0.5, \rho=0.5$)**.
   - Option 2 (`final_technical_audit.md`, `dsmsa_expand`): Expand branches to `256*alpha` $\implies$ `768*alpha` concat $\implies$ `409k` ($\alpha=1.0$), **`137.7k` ($\alpha=0.5, \rho=1.0$)**, **`104.1k` ($\alpha=0.5, \rho=0.5$)**.
   - **Action needed:** Pick one canonical `Lite-DFE` architecture (`dsmsa_db` saves `~86k` more params and `~47M` FLOPs at $\alpha=0.5, \rho=1.0$; or test both in `B1`).
2. **`B0-Clean` Micro-Architecture Convention Must Be Locked in Config:**
   - Decide explicitly: (a) keep or omit Fig. 4's $1\times 1$ conv on $F_3$ (`+66k` params, `+0.034G` FLOPs), (b) `S_d` upsampling block (`Bilinear + Conv3x3` = `3.326M` params / `4.362G` FLOPs vs. `ConvTranspose2d` = `3.244M` params / `3.288G` FLOPs), (c) whether `B0-Clean` uses `f.detach()` before `DFE` by default or if `detach` is strictly treated as part of `A2` vs `B0` baseline, and (d) whether `DFE` takes raw `F_e` (Fig. 4/Sec. 3.2) or `l2_normalized` `\hat{F}_e` (`Algo. 2` line 7).
3. **Standardize Complexity Reporting Metric (`FlopCounterMode` FLOPs vs. MACs):**
   - Replace every stale `11.571M / 8.788M / 6.379G` table entry across `research_decision.md`, `experiment_design.md`, `method_b_proposal.md`, `method_comparison.md`, and `final_technical_audit.md` with the verified PyTorch counts (`12.180M` / `9.397M` trainable / **`13.485G` `FlopCounterMode` FLOPs (`6.742G` MACs)** for `f3c=True, bilinear`, or `12.098M` / `9.315M` / **`12.411G` FLOPs (`6.206G` MACs)** for `f3c=True, deconv`), reporting both `FlopCounterMode FLOPs` and `MACs` explicitly (`1 MAC = 2 FLOPs`).
4. **Clarify `S0_input_norm` (`P_img_l2`) Exact Definition:**
   - Lock the exact mathematical definition of `P_img_l2` (full tensor $\ell_2$: `x / ||x||_2` vs. per-pixel RGB $\ell_2$) before writing dataloader transforms, and consider running `S0_eq3_norm` / `S0_input_norm` / `S0_split` for sufficient epochs (or full 200 epochs on 2 categories—at `~6 min/cat`, even 200-epoch `S0` runs cost only `~0.1h/run`).
5. **Harmonize Contradictions in `research_decision.md` & `kaggle_implementation_plan.md`:**
   - Remove the Round-2 hardcoded `Eq. (3) = Unit-L2` and `ReLU(BN(Conv1x1))` text in `research_decision.md` Section 5 (lines 78, 82), fix the Phase 1–6 model count/naming table (`S0_split`, `20` models for `A`, `C1-0.75/0.50`, line 538 stale `B2` note), and plan a 1-category CPU benchmark of `compute_aupro` before trusting the `~19.5h` schedule.

---

## 8. Final Implementation Readiness Verdict

> **FINAL VERDICT: `READY AFTER REQUIRED CORRECTIONS`**

The core scientific hypotheses (`Method A: FC-DSKD` and `Method B: Lite-DSKD`), data-leakage safeguards (`Zero Test Leakage`, `MVTecTrainDataset` purity & `80/20` split), metric implementations (`evaluation/metrics.py`), and checkpoint/resume architecture (`CHECKPOINT_SCHEMA_V2`) are **sound and feasible** within the team's Kaggle GPU budget. However, code implementation **must NOT** start under the current contradictory `research_decision.md` checklist and miscounted `B0-Clean` / `Lite-DFE` specifications. Once the human approves the architectural conventions and syncs the 5 items in Section 7, the team is cleared to begin Phase 2 implementation (`src/anomaly_detection/`).
