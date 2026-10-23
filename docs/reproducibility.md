# Reproducibility Specification and Checklist

## 1. Deterministic Seeding and Environment Control

All training, calibration, evaluation, and latency profiling scripts call `set_seed(seed=42)` (`src/anomaly_detection/utils/seed.py`), which sets:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## 2. Exact Commands to Reproduce Each Stage

### 2.1 Environment & Unit Test Verification (No Dataset Required)
```bash
# 1. Check Python, PyTorch, CUDA, and package versions
python scripts/check_environment.py

# 2. Run the unit test suite (synthetic tensors & mock dataset)
python -m unittest discover -s tests -v
```

### 2.2 Dataset Validation (Requires MVTec AD)
```bash
python scripts/validate_dataset.py --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw
```

### 2.3 Stage 3 Smoke Test (`R00` — 2 Epochs on 1 Texture + 1 Object)
```bash
python scripts/train.py --config configs/experiments/smoke_test.yaml --data-root /kaggle/input/mvtec-ad --categories carpet bottle
```

### 2.4 Stage 4 Baseline Reproduction (`B0` — 160 Epochs)
```bash
# Train baseline on core triad
python scripts/train.py --config configs/baseline/cda_mvtec.yaml --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw

# Resume an interrupted run from checkpoint
python scripts/train.py --config configs/baseline/cda_mvtec.yaml --data-root /kaggle/input/mvtec-ad --categories screw --resume outputs/baseline/screw/checkpoints/last.pth
```

### 2.5 Stage 5A — Thành's Calibrated Reliability Fusion (`A0` $\to$ `A3`, Inference-Only on `B0`)
```bash
python scripts/evaluate.py --config configs/research_a/default.yaml --data-root /kaggle/input/mvtec-ad --checkpoint-dir outputs/baseline --categories carpet bottle screw --ablation-all
```

### 2.6 Stage 5B — Luân's Structural Adapter & GCM Compression (`B1` $\to$ `B3`)
```bash
# B1: Inference-only cross-scale pruning on B0 checkpoint
python scripts/evaluate.py --config configs/research_b/default.yaml --variant B1_aux_csd --data-root /kaggle/input/mvtec-ad --checkpoint-dir outputs/baseline --categories carpet bottle screw

# B2: Train Depthwise-Separable Cross-Scale Adapter variant (28.52M params)
python scripts/train.py --config configs/research_b/default.yaml --variant B2_sep_adapter --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw

# B3: Train Factorized GCM + Separable Adapter variant (19.46M params)
python scripts/train.py --config configs/research_b/default.yaml --variant B3_cda_lite --data-root /kaggle/input/mvtec-ad --categories carpet bottle screw

# Benchmark parameter counts, MACs/FLOPs, latency (ms), and peak VRAM across B0, B1, B2, B3
python scripts/benchmark_latency.py --variants B0 B1_aux_csd B2_sep_adapter B3_cda_lite --img-size 256 --warmup 50 --iters 200
```

---

## 3. Artifact Management & Anti-Bloat Policy

1. **Never Committed to Git (`.gitignore` Enforced):**
   - Raw datasets (`datasets/`, `mvtec/`, `visa/`, `*.tar.xz`, `*.zip`)
   - Model checkpoints (`*.pth`, `*.pt`, `*.ckpt`, `checkpoints/`)
   - Large experiment outputs (`outputs/*`, `logs/`, `runs/`, `*.tfevents*`)
   - Credentials (`kaggle.json`, `.env`, `*.pem`, `*.key`)
2. **Always Committed to Git:**
   - Source code (`src/`), scripts (`scripts/`), unit tests (`tests/`), YAML configs (`configs/`), documentation (`docs/`), and lightweight summary CSV/JSON tables in `experiments/` once verified.
