# Official CDA Implementation Audit Report

**Audited Repository:** `https://github.com/WHUer-cloud/CDA`  
**Latest Commit Audited:** `67224ca04d3aff03c97bdaad04aafb0b3a1ad74e` (Author: `LiyAohUi <19374154819@163.com>`, Date: 2025-10-19)  
**Total Files in Official Repo:** 8 (`README.md`, `config.py`, `dataloder.py`, `metric.py`, `model.py`, `test.py`, `train.py`, `Transformer.py`)

---

## 1. Executive Summary of the Code Audit

We cloned and audited the complete official CDA repository (`WHUer-cloud/CDA`), including its full 9-commit Git history.

- **Completeness of Core Model (`model.py`, `Transformer.py`):** **High.** All major modules reported in the paper — Teacher `Encoder` (`Wide_ResNet50_2`), `GCM` (`CLE` + `SGA`/`Fusion` + `res`), `Decoder` (`PatchExpand2D` + `GSULayer`/`GSU`), and cross-scale alignment convolutions (`c11..c32`) — are fully implemented and match the paper's structural description.
- **Out-of-the-Box Reproducibility:** **Broken without fixes.** Running `python train.py --obj_id -1` in a clean environment immediately fails due to multiple software bugs (a misspelled filename `dataloder.py` vs. `from data_loader import ...`, missing `parents=True` in `Path.mkdir()`, hard-coded dataset paths, and import-time global CUDA device binding).
- **Experimental Protocol Rigor:** **Requires scientific hardening.** The official `train.py` selects `best_img.pth` and `best_pix.pth` by evaluating on the test set every 5 epochs (test-set leakage) and does not support resuming interrupted training sessions (critical for Kaggle GPU time limits).
- **Licensing Status:** The official repository contains **no `LICENSE` file**. Under GitHub terms of service, code without an explicit license defaults to all-rights-reserved by the authors, permitting viewing and forking on GitHub and academic inspection, **not** wholesale unlicensed copying into a third-party MIT/Apache codebase. Therefore, our repository implements a **clean-room, modular, bug-fixed PyTorch reconstruction** with full academic citation and attribution to Hu et al. (2026) and `WHUer-cloud/CDA`.

---

## 2. Checklist Verification Against Work Package 2 Criteria

| Audit Criterion | Status in Official Repo (`WHUer-cloud/CDA`) | Detailed Finding & Impact on Our Project |
| :--- | :--- | :--- |
| **1. Implements all major modules described in the paper?** | **Yes** | `Encoder`, `GCM` (`local1`, `local2`, `merge`, `Fusion`/`SelfAttention` with RoPE, `res`), `Decoder` (`PatchExpand2D`, `GSULayer`, `GSU`, `DepthwiseMLP`, `GroupedLinear`, `RMSNorm`), and `GCMAndDecoder` (`c11..c32`) are all present. |
| **2. Default config matches reported experimental setup?** | **Yes (for MVTec AD)** | `config.py` defaults to `bs=4`, `lr=0.005`, `epochs=160`, `img_size=256`, `test_interval=5`, and lists all 15 MVTec AD categories (plus commented lists for BTAD and VisA). |
| **3. Pretrained weights available and legally accessible?** | **Teacher: Yes / Student: No** | Teacher uses standard Torchvision `Wide_ResNet50_2_Weights.DEFAULT` (freely accessible under BSD-3-Clause). Pretrained student checkpoints are not hosted in the repo; models must be trained from scratch per category. |
| **4. Dependencies compatible with Kaggle environment?** | **Mostly Yes (No `requirements.txt` provided)** | Official repo has no `requirements.txt`. Imports require `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `Pillow`, `tqdm`, `tensorboard`, and `thop` (imported in `train.py` line 10 though unused after commit `537f928`). All except `thop` are pre-installed on Kaggle GPUs. |
| **5. Dataset paths hard-coded?** | **Yes (Bug)** | `dataloder.py` line 42 hard-codes `self.gt_root = "./datasets/mvtec/" + self.obj_name + "/ground_truth/"`, ignoring `--data_path`! This fails on Kaggle where MVTec AD resides at `/kaggle/input/mvtec-ad/`. |
| **6. Training/evaluation reproducible from a clean environment?** | **No (4 Blocking Bugs)** | See Section 3 below for the exact tracebacks and fixes. |
| **7. Checkpoint saving and resuming supported?** | **PartialSave / No Resume** | Saves `dec.state_dict()` only (`epoch{epoch}.pth`, `best_img.pth`, `best_pix.pth`). Does **not** save optimizer state, epoch index, RNG states, or best metric trackers; cannot resume an interrupted Kaggle run. |
| **8. Evaluation code follows stated benchmark protocol?** | **Partial (Leakage + Missing Metrics)** | Computes Image-AUROC and Pixel-AUROC via `sklearn.metrics.roc_auc_score`, but (a) selects best checkpoints on the test set every 5 epochs, (b) allows `best_img` and `best_pix` to come from different epochs, and (c) omits PRO/AUPRO and Average Precision (AP). |
| **9. Undocumented assumptions affecting results?** | **Yes (5 Assumptions)** | See Section 4 below. |

---

## 3. Detailed Inventory of Bugs and Issues in Official Code

### Bug 1 (Blocker): Filename Typo `dataloder.py` vs. `import data_loader`
- **Location:** `train.py` line 6 (`from data_loader import TrainDataset`) and `test.py` line 3 (`from data_loader import TestDataset`), while the file in the repository is named `dataloder.py`.
- **Effect:** Immediate `ModuleNotFoundError: No module named 'data_loader'` when running `train.py` or `test.py`.

### Bug 2 (Blocker): Missing `parents=True` in Checkpoint Directory Creation
- **Location:** `train.py` lines 16, 19–20:
  ```python
  os.makedirs(args.checkpoint_path, exist_ok=True)
  ck_dir = Path(args.checkpoint_path) / "WRes50" / run_name
  ck_dir.mkdir(exist_ok=True)
  ```
- **Effect:** Because `./checkpoints/WRes50` does not exist on a fresh clone, `ck_dir.mkdir(exist_ok=True)` raises `FileNotFoundError: [WinError 3] / [Errno 2] No such file or directory`.

### Bug 3 (Blocker on Kaggle / Custom Data Paths / Windows): Hard-Coded `gt_root` and Slash Splitting in `dataloder.py`
- **Location:** `dataloder.py` lines 42, 64–65:
  ```python
  self.gt_root = "./" + "datasets/mvtec/" + self.obj_name + "/ground_truth/"
  ...
  label = img_path.split("/")[-2]
  gt_path = self.gt_root + label + "/" + img_path.split("/")[-1][:3] + "_mask.png"
  ```
- **Effect:**
  1. Ignores `root_dir` (`--data_path`) for ground-truth masks, failing whenever data is not located at `./datasets/mvtec/`.
  2. On Windows, `glob.glob` returns paths with `\` separators, causing `img_path.split("/")[-2]` to raise `IndexError` or return the wrong string.
  3. Taking `[:3]` assumes all image stems have exactly 3 characters (true for MVTec AD `000.png`, false for VisA or custom datasets; `Path(img_path).stem + "_mask.png"` is the portable solution).

### Bug 4 (Latent Crash): Hard-Coded `256*256` in `metric.py`
- **Location:** `metric.py` line 47:
  ```python
  img_scores = np.array([s.max() for s in scores.reshape(-1, 256*256)])
  ```
- **Effect:** If `--img_size` is changed from `256` (e.g., `128` for a smoke test or `224`), `compute_auroc` either crashes or groups pixels across different images! Moreover, `scores` was already a list of 2D arrays `(H, W)` before line 45 flattened and concatenated them.

### Bug 5 (Latent Crash): Hard-Coded `16x16` Sequence Reshape in `Transformer.py`
- **Location:** `Transformer.py` lines 22–24 (`view(b, 1, 256)`) and line 163 (`return x.view(b, 256, 16, 16)`), plus `if __name__ == "__main__": net = Fusion(2048, 2048)` (which passes 2 arguments to `Fusion.__init__(self)` that takes 0 arguments).
- **Effect:** `Fusion` only works when `e3` has spatial size $16 \times 16$ (i.e., input image size $256 \times 256$). In our clean implementation, we dynamically read `h, w = x3.shape[2], x3.shape[3]` and `seq_len = h * w` so the model supports arbitrary input resolutions divisible by 32 (e.g., $128 \times 128$ or $256 \times 256$) while remaining 100% identical at $256 \times 256$.

### Bug 6 (Device Management Anti-Pattern): Import-Time Global `device` in `model.py`
- **Location:** `model.py` lines 7, 13, 16, 45–49, 91.
- **Effect:** `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` is bound at module import time and passed to some parameters (`RMSNorm`, `GSU`, `PatchExpand2D`) while omitted on others (`GroupedLinear`, `ConvBlock`, `GCM`). Calling `.to(other_device)` on `GCMAndDecoder` fails if `RMSNorm.forward` forces `x = x.to(device)` using the global variable.

---

## 4. Undocumented Assumptions & Scientific Implications

1. **Unshuffled Training DataLoader with `drop_last=True` (`train.py` line 28):**
   - `DataLoader(ds, batch_size=args.bs, shuffle=False, drop_last=True)` means:
     - Every epoch processes the exact same deterministic mini-batches in alphabetical filename order (`000.png`, `001.png`, ...), affecting `BatchNorm2d` running statistics in `GCM`.
     - Up to 3 normal training images at the end of the sorted list are permanently dropped if `len(ds) % 4 != 0`.
2. **Teacher `BatchNorm2d` Frozen in `.eval()` Mode, Student `BatchNorm2d` in `.train()` Mode:**
   - `encoder = Encoder().to(device).eval()` freezes both weights and running statistics of `Wide_ResNet50_2`.
3. **Asymmetric Weighting in Cross-Scale Loss (`train.py` lines 43–45) vs. Anomaly Score (`metric.py` line 32):**
   - Documented in detail in [paper_equations.md](file:///c:/Users/Administrator/Desktop/workspace/CS331-Anomaly-Detection/cs331-knowledge-distillation-lightweight-industrial-anomaly-detection/docs/paper_equations.md).
4. **Attention Scaling by Sequence Length $L=256$ Instead of Head Dimension $d_h=32$ (`Transformer.py` line 147):**
   - `scaling = K.shape[2]` divides logits by $\sqrt{256} = 16$ instead of $\sqrt{32} \approx 5.66$, making the softmax distribution softer (closer to uniform spatial averaging).
5. **Independent Test-Set Selection of `best_img` and `best_pix` (`train.py` lines 64–69):**
   - `best_img` and `best_pix` can come from different epochs (e.g., epoch 85 for `best_img` and epoch 140 for `best_pix`) evaluated directly on the test set.

---

## 5. Exact Parameter and Complexity Profile of Official CDA

Verified using PyTorch 2.11.0 on the official `model.py`:

| Component | Module Name in Official Code | Parameters | % of Trainable (`GCMAndDecoder`) | Output Tensor Shape (at $256\times256$ input) |
| :--- | :--- | ---: | ---: | :--- |
| **Teacher Encoder (Frozen)** | `Encoder` (`Wide_ResNet50_2` layers 1–3) | `24,862,528` | — (Frozen) | `e1:(256,64,64), e2:(512,32,32), e3:(1024,16,16)` |
| **GCM — Global Branch (SGA)** | `GCM.fusion` (`Fusion` in `Transformer.py`) | `1,384,451` | `3.21%` | `g: (1024, 16, 16)` |
| **GCM — Local Branch (CLE)** | `GCM.local1 + local2 + merge` | `8,866,048` | `20.53%` | `l: (1024, 16, 16)` |
| **GCM — Bottleneck ResBlock** | `GCM.res` (`ConvBlock 1024->2048, s=2`) | `9,454,080` | `21.89%` | `f: (2048, 8, 8)` |
| *Subtotal: GCM* | `GCM` | *`19,704,579`* | *`45.63%`* | `f: (2048, 8, 8)` |
| **Student Decoder Stage 3** | `Decoder.l3` (`PatchExpand2D + GSULayer(1024)`) | `5,289,984` | `12.25%` | `s3: (1024, 16, 16)` |
| **Student Decoder Stage 2** | `Decoder.l2` (`PatchExpand2D + GSULayer(512)`) | `1,334,272` | `3.09%` | `s2: (512, 32, 32)` |
| **Student Decoder Stage 1** | `Decoder.l1` (`PatchExpand2D + GSULayer(256)`) | `335,360` | `0.78%` | `s1: (256, 64, 64)` |
| *Subtotal: GSU Decoder* | `Decoder` | *`6,959,616`* | *`16.12%`* | `s1, s2, s3` |
| **Cross-Scale Adapters** | `c11, c12, c21, c22, c31, c32` ($3\times3$ convs) | `16,518,656` | `38.25%` | `l1m, l1n, l2m, l2n, l3m, l3n` |
| **Total Trainable Student (`GCMAndDecoder`)** | `GCMAndDecoder` | **`43,182,851`** | **`100.00%`** | 9 feature maps |
| **Total Pipeline (Teacher + Student)** | `Encoder + GCMAndDecoder` | **`68,045,379`** | — | Anomaly map `(B, 256, 256)` |
