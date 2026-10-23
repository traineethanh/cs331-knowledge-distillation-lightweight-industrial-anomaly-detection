# Kaggle GPU Execution Plan and Resource Management Strategy

## 1. Kaggle Platform Constraints and Budget

- **Typical Kaggle GPU Hardware:** NVIDIA Tesla T4 (`16 GB` VRAM) or NVIDIA Tesla P100 (`16 GB` VRAM).
- **Session Limits:** Maximum `12 hours` per continuous interactive/background notebook run; `30 hours/week` total GPU quota per account (combined across Thành and Luân = up to `60 hours/week`).
- **Ephemeral Scratch Disk:** `/kaggle/working/` (`~20 GB` limit); wiped if a session times out without saving outputs or uploading artifacts.
- **Read-Only Dataset Mount:** `/kaggle/input/mvtec-ad/` (MVTec AD dataset mounted directly from Kaggle Datasets at zero working-disk cost).

---

## 2. Memory and Runtime Estimates for CDA on Kaggle GPU

Based on our architectural profile of CDA (`24.86M` frozen teacher + `43.18M` trainable student at $256 \times 256$, `batch_size = 4`):
- **Model Weights in FP32:**
  - Teacher (`Encoder`): `24.86M * 4 bytes` $\approx 99.5\text{ MB}$.
  - Student (`GCMAndDecoder`): `43.18M * 4 bytes` $\approx 172.7\text{ MB}$ (plus Adam moments $2\times \approx 345.4\text{ MB}$ during training).
- **Activation & Gradient Peak VRAM (`bs = 4`, $256 \times 256$):**
  - Estimated peak VRAM during training: **`~2.4 GB – 3.2 GB`** (fits comfortably within $16\text{ GB}$ T4/P100 VRAM with $>12\text{ GB}$ headroom).
- **Training Wall-Clock Time Estimate (per MVTec AD category, ~240 normal images, 160 epochs = 9,600 iterations at `bs = 4`):**
  - Estimated time on Tesla T4 / P100: **`~12 – 18 minutes` per category** (when test evaluation runs only at final epoch or every 20 epochs instead of every 5 epochs!).
  - > [!TIP]
    > **Crucial Kaggle Optimization:** In the official `train.py`, evaluating on the test set every `test_interval = 5` epochs runs `32` full test evaluations (with CPU `scipy.ndimage.gaussian_filter` per test image) per category, doubling wall-clock time! Setting `eval_interval: 20` (or evaluating at epoch 160) cuts training time per category by **~35–45%**.

---

## 3. Six-Stage Execution Workflow on Kaggle

### Stage 1 — Environment Check (`notebooks/01_environment_check.ipynb`)
- Run `python scripts/check_environment.py`.
- Verify CUDA availability, GPU name (`Tesla T4` or `P100`), VRAM capacity, PyTorch/Torchvision versions, free disk space in `/kaggle/working/`, and dataset mount path `/kaggle/input/mvtec-ad/`.

### Stage 2 — Data Smoke Test (`notebooks/02_data_smoke_test.ipynb`)
- Run `python scripts/validate_dataset.py --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw`.
- Load 1 normal training sample (`label=0`) and 1 anomalous test sample + ground-truth mask (`label=1`).
- Verify tensor dimensions `(3, 256, 256)` and `(1, 256, 256)`, ImageNet normalization ranges, and confirm zero overlap between train and test file paths.

### Stage 3 — Baseline Smoke Test (`notebooks/03_cda_baseline_smoke_test.ipynb`, Run `R00`)
- Run `python scripts/train.py --config configs/experiments/smoke_test.yaml --categories carpet bottle`.
- Executes 2 epochs on 1 texture (`carpet`) and 1 object (`bottle`), saves `last.pth`, tests `--resume` from `last.pth`, runs `scripts/evaluate.py`, and generates a visualization panel via `scripts/visualize_predictions.py`.
- **Gate:** Must complete in $<3$ minutes with zero errors before launching 160-epoch runs.

### Stage 4 — Reproducibility Baseline Runs (`R01` $\to$ `R02`)
1. **Core Representative Subset (`R01` — 3 Categories, ~45 mins total):**
   - Train `B0` for 160 epochs on:
     - `carpet` (texture category with complex background weave)
     - `bottle` (rigid object category with structural defects)
     - `screw` (object category with small/fine-grained surface defects and rotation variance)
2. **Full Benchmark Expansion (`R02` — Remaining 12 Categories, batched into 3 Kaggle runs of 4 categories each, ~1 hour per batch):**
   - Batch 1: `tile`, `leather`, `wood`, `grid` (remaining 4 textures)
   - Batch 2: `capsule`, `pill`, `toothbrush`, `transistor`
   - Batch 3: `cable`, `hazelnut`, `metal_nut`, `zipper`
3. **Checkpoint Archival:** Immediately export `/kaggle/working/outputs/B0/` as a private **Kaggle Dataset (`cs331-cda-b0-checkpoints`)** so all Stage 5 experiments mount `B0` checkpoints read-only in `/kaggle/input/`!

### Stage 5 — Controlled Improvement Experiments (`A1`–`A3` and `B1`–`B3`)
- **Thành's Branch (`A1`, `A2`, `A3`):**
  - Mount `cs331-cda-b0-checkpoints` in Kaggle.
  - Run `scripts/evaluate.py` with `configs/research_a/default.yaml` across `A0`, `A1`, `A2`, `A3`.
  - Because `A1`–`A3` require **zero retraining**, all 15 categories across all 4 variants (`60` evaluations) finish in **`< 25 minutes`** on a single Kaggle GPU session!
- **Luân's Branch (`B1`, `B2`, `B3`):**
  - Run `B1` (`CDA-AuxCSD`, inference-only pruning of `c11..c32`) directly on `B0` checkpoints (`< 5 minutes`, zero retraining).
  - Train `B2` (`CDA-SepAdapter`, `28.52M` params) and `B3` (`CDA-Lite`, `19.46M` params) on the 3 core categories (`carpet`, `bottle`, `screw`) first (`R01` scope), then expand to more categories as GPU quota permits.
  - Run `scripts/benchmark_latency.py` comparing `B0`, `B1`, `B2`, `B3` in the **same Kaggle GPU session**.

### Stage 6 — Combined Model (`C1`)
- Evaluate `C1` (Luân's trained `B2`/`B3` checkpoints + Thành's `A3` Calibrated Reliability Fusion mapper) with zero additional training cost.

---

## 4. Checkpointing, Resume, and Session-Expiry Protection

To ensure no GPU work is ever lost to a Kaggle timeout:
1. **Full-State Checkpointing (`src/anomaly_detection/training/trainer.py`):**
   Every `save_interval` epochs (default `10`) and at the final epoch (`160`), `trainer.py` saves an atomic checkpoint dictionary:
   ```python
   {
       "epoch": epoch,
       "model_state_dict": student.state_dict(),
       "optimizer_state_dict": optimizer.state_dict(),
       "best_metrics": best_metrics,
       "config": config_dict,
       "git_commit": git_hash,
   }
   ```
2. **Seamless Resume (`--resume path/to/last.pth`):**
   If a Kaggle session stops at epoch 110, passing `--resume outputs/.../last.pth` restores the model weights, Adam momentum buffers (`exp_avg`, `exp_avg_sq`), and epoch counter (`111..160`).
3. **Disk Footprint Control:**
   Each student checkpoint is `~173 MB` (`B0`) or `~78 MB` (`B3`). To stay well below Kaggle's `20 GB` `/kaggle/working/` limit, our trainer overwrites rolling `last.pth` rather than keeping intermediate `epoch5.pth, epoch10.pth...` files on disk.
