"""Unified Kaggle GPU Training & Ablation Pipeline for CS331 Anomaly Detection.

Supports running in a single command:
1. Baseline (`STFPM-Paper` & `STFPM-Smoothed`)
2. Method 1 — Researcher A (`SCG-STFPM` Ablations: `A1_calibrated`, `A2_concordance_gate`, `A3_full_scg`)
3. Method 2 — Researcher B (`Slim-STFPM` Ablations: alpha=0.75/0.5, Pointwise/DW-Sep, L2 vs Focal-KD Loss)
4. Unified Method (`C1_unified_slim_scg`: Slim-STFPM + Focal-KD + SCG-STFPM Calibration & Gating)

Optimized for Kaggle GPU (Tesla T4 / P100, 16GB VRAM):
- Optional Automatic Mixed Precision (`--amp` via `torch.amp.autocast`)
- Pin memory and non-blocking tensor transfers
- Smart checkpoint reuse: Method 1 (`A1..A3`) reuses the trained Baseline checkpoint without
  redundant retraining; Unified `C1` reuses the trained Slim-STFPM checkpoint.
- Automatically exports publication-ready `.csv` and `.md` Markdown tables for the CS331 report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTestDataset, MVTecTrainDataset
from anomaly_detection.evaluation import evaluate_predictions
from anomaly_detection.models.baseline.stfpm import STFPM, STFPMLoss, compute_stfpm_anomaly_map
from anomaly_detection.models.research_a.scg_stfpm import SCGNormalStats, SCGSTFPMMapper
from anomaly_detection.models.research_b.slim_stfpm import (
    FocalSTFPMLoss,
    SlimSTFPM,
    count_conv_bn_flops,
)
from anomaly_detection.utils import set_seed


def measure_pipeline_latency(
    model: nn.Module,
    mapper: SCGSTFPMMapper,
    img_size: int = 256,
    device: torch.device = torch.device("cpu"),
    warmup: int = 10,
    iters: int = 30,
) -> Dict[str, float]:
    """Measures end-to-end forward + anomaly map generation latency with CUDA sync."""
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            t_feats, s_feats = model(dummy)
            _ = mapper.compute_anomaly_map(t_feats, s_feats)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            t_feats, s_feats = model(dummy)
            _ = mapper.compute_anomaly_map(t_feats, s_feats)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    arr = np.asarray(timings, dtype=np.float64)
    return {
        "latency_ms": round(float(arr.mean()), 2),
        "latency_std_ms": round(float(arr.std()), 2),
        "fps": round(float(1000.0 / max(arr.mean(), 1e-4)), 1),
    }


def train_student_checkpoint(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    ckpt_dir: Path,
    category: str,
    variant_key: str,
    epochs: int = 100,
    lr: float = 0.4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    use_amp: bool = False,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """Trains a Teacher-Student model (STFPM or SlimSTFPM) or loads cached checkpoint if present."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pth"
    best_ckpt_path = ckpt_dir / "best_val.pth"

    if last_ckpt_path.exists() and not force_retrain:
        print(f"  [Cache Hit] Loading existing trained checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt["student_state_dict"])
        return {
            "checkpoint_path": str(last_ckpt_path),
            "peak_vram_mb": float(ckpt.get("peak_vram_mb", 0.0)),
            "train_duration_sec": float(ckpt.get("total_duration_sec", 0.0)),
            "final_train_loss": float(ckpt.get("final_train_loss", 0.0)),
        }

    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    peak_vram_mb = 0.0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []
    start_time = time.time()

    print(f"  [Train] Variant={variant_key} | Category={category} | Epochs={epochs} | AMP={amp_enabled}")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()  # Keeps teacher in eval mode permanently
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled and scaler is not None:
                with torch.amp.autocast("cuda"):
                    t_feats, s_feats = model(images)
                    loss = loss_fn(t_feats, s_feats)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                t_feats, s_feats = model(images)
                loss = loss_fn(t_feats, s_feats)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validation loss on 20% holdout normal split
        val_loss = avg_train_loss
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device, non_blocking=True)
                    t_feats, s_feats = model(images)
                    v_loss += float(loss_fn(t_feats, s_feats).item())
                    v_batches += 1
            val_loss = v_loss / max(v_batches, 1)

        if device.type == "cuda":
            cur_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_mb = max(peak_vram_mb, cur_vram)

        ep_sec = time.time() - ep_t0
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "epoch_time_sec": round(ep_sec, 2),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "category": category,
                    "variant_key": variant_key,
                    "epoch": epoch,
                    "student_state_dict": model.student.state_dict(),
                    "best_val_loss": best_val_loss,
                    "seed": seed,
                },
                best_ckpt_path,
            )

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"    Epoch [{epoch:03d}/{epochs:03d}] Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} ({ep_sec:.2f}s)"
            )

    total_duration = time.time() - start_time
    final_train_loss = history[-1]["train_loss"] if history else 0.0

    torch.save(
        {
            "category": category,
            "variant_key": variant_key,
            "epoch": epochs,
            "student_state_dict": model.student.state_dict(),
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "total_duration_sec": round(total_duration, 2),
            "history": history,
            "seed": seed,
        },
        last_ckpt_path,
    )

    return {
        "checkpoint_path": str(last_ckpt_path),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "train_duration_sec": round(total_duration, 2),
        "final_train_loss": final_train_loss,
    }


@torch.no_grad()
def evaluate_model_with_mapper(
    model: nn.Module,
    mapper: SCGSTFPMMapper,
    test_loader: DataLoader,
    img_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates a trained model + mapper on MVTecTestDataset and measures hardware latency."""
    model.eval()
    all_maps: List[np.ndarray] = []
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_masks: List[np.ndarray] = []

    for batch in test_loader:
        image = batch["image"].to(device, non_blocking=True)
        gt_mask = batch["gt_mask"].squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        label = int(batch["label"].item())

        t_feats, s_feats = model(image)
        anom_map, img_score = mapper.compute_anomaly_map(t_feats, s_feats)

        all_maps.append(anom_map.squeeze(0).cpu().numpy())
        all_scores.append(float(img_score.cpu().item()))
        all_labels.append(label)
        all_masks.append(gt_mask)

    maps_arr = np.stack(all_maps, axis=0)
    scores_arr = np.asarray(all_scores, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    masks_arr = np.stack(all_masks, axis=0)

    metrics = evaluate_predictions(
        anomaly_maps=maps_arr,
        image_labels=labels_arr,
        gt_masks=masks_arr,
        image_scores=scores_arr,
        include_aupro=True,
    )
    latency = measure_pipeline_latency(model, mapper, img_size=img_size, device=device)
    metrics.update(latency)
    return metrics


def get_experiment_definitions(suite: str) -> List[Dict[str, Any]]:
    """Returns ordered list of experiment configurations for the requested suite."""
    baseline_exps = [
        {
            "exp_id": "B0_STFPM_Paper",
            "group": "Baseline",
            "ckpt_key": "stfpm_base",
            "arch": "stfpm",
            "width_mult": 1.0,
            "proj_type": "identity",
            "focal_gamma": 0.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 0.0,
            "description": "Foundational STFPM (Paper Eq. 2-3, no blur, max score)",
        },
        {
            "exp_id": "B0_STFPM_Smoothed",
            "group": "Baseline",
            "ckpt_key": "stfpm_base",
            "arch": "stfpm",
            "width_mult": 1.0,
            "proj_type": "identity",
            "focal_gamma": 0.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 4.0,
            "description": "Community STFPM-Smoothed (sigma=4.0, max score)",
        },
    ]

    method1_exps = [
        {
            "exp_id": "A1_ScaleCalibrated",
            "group": "Method 1 (SCG-STFPM)",
            "ckpt_key": "stfpm_base",
            "arch": "stfpm",
            "width_mult": 1.0,
            "proj_type": "identity",
            "focal_gamma": 0.0,
            "mapper_mode": "A1_calibrated",
            "baseline_sigma": 4.0,
            "description": "Scale-wise Normal Z-Score Calibration (train/good)",
        },
        {
            "exp_id": "A2_ConcordanceGate",
            "group": "Method 1 (SCG-STFPM)",
            "ckpt_key": "stfpm_base",
            "arch": "stfpm",
            "width_mult": 1.0,
            "proj_type": "identity",
            "focal_gamma": 0.0,
            "mapper_mode": "A2_concordance_gate",
            "baseline_sigma": 4.0,
            "description": "A1 + Inter-Scale Semantic Concordance Gate (eta=0.5)",
        },
        {
            "exp_id": "A3_Full_SCG_STFPM",
            "group": "Method 1 (SCG-STFPM)",
            "ckpt_key": "stfpm_base",
            "arch": "stfpm",
            "width_mult": 1.0,
            "proj_type": "identity",
            "focal_gamma": 0.0,
            "mapper_mode": "A3_full_scg",
            "baseline_sigma": 4.0,
            "description": "Full SCG-STFPM (A2 + Adaptive Blur [1.5,2.5,4.0] + Top-0.5%)",
        },
    ]

    method2_exps = [
        {
            "exp_id": "M2_Slim075_PW_L2",
            "group": "Method 2 (Slim-STFPM)",
            "ckpt_key": "slim075_pw_l2",
            "arch": "slim",
            "width_mult": 0.75,
            "proj_type": "pointwise",
            "focal_gamma": 0.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 4.0,
            "description": "Slim-STFPM (alpha=0.75, 1x1 PW Align, L2 Loss)",
        },
        {
            "exp_id": "M2_Slim050_PW_L2",
            "group": "Method 2 (Slim-STFPM)",
            "ckpt_key": "slim050_pw_l2",
            "arch": "slim",
            "width_mult": 0.5,
            "proj_type": "pointwise",
            "focal_gamma": 0.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 4.0,
            "description": "Slim-STFPM (alpha=0.50, 1x1 PW Align, L2 Loss)",
        },
        {
            "exp_id": "M2_Slim050_DWSep_L2",
            "group": "Method 2 (Slim-STFPM)",
            "ckpt_key": "slim050_dwsep_l2",
            "arch": "slim",
            "width_mult": 0.5,
            "proj_type": "dw_sep",
            "focal_gamma": 0.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 4.0,
            "description": "Slim-STFPM (alpha=0.50, 3x3 DW-Sep Align, L2 Loss)",
        },
        {
            "exp_id": "M2_Slim050_DWSep_Focal",
            "group": "Method 2 (Slim-STFPM)",
            "ckpt_key": "slim050_dwsep_focal",
            "arch": "slim",
            "width_mult": 0.5,
            "proj_type": "dw_sep",
            "focal_gamma": 2.0,
            "mapper_mode": "A0_baseline",
            "baseline_sigma": 4.0,
            "description": "Full Slim-STFPM (alpha=0.50, DW-Sep Align + Focal-KD gamma=2.0)",
        },
    ]

    unified_exps = [
        {
            "exp_id": "C1_Unified_Slim_SCG",
            "group": "Unified (Method 1 + 2)",
            "ckpt_key": "slim050_dwsep_focal",
            "arch": "slim",
            "width_mult": 0.5,
            "proj_type": "dw_sep",
            "focal_gamma": 2.0,
            "mapper_mode": "A3_full_scg",
            "baseline_sigma": 4.0,
            "description": "Unified Slim-STFPM (0.74M, Focal-KD) + SCG-STFPM (A3_full_scg)",
        },
    ]

    if suite == "baseline":
        return baseline_exps
    if suite == "method1":
        return baseline_exps + method1_exps
    if suite == "method2":
        return baseline_exps + method2_exps
    if suite == "unified":
        return baseline_exps + [method1_exps[-1], method2_exps[-1]] + unified_exps
    if suite == "all":
        return baseline_exps + method1_exps + method2_exps + unified_exps
    raise ValueError(f"Unknown suite '{suite}'.")


def write_summary_reports(
    records: List[Dict[str, Any]],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Exports CSV and Markdown summary tables (including Mean across categories)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_ablation.csv"
    md_path = out_dir / "summary_ablation.md"

    fieldnames = [
        "category",
        "exp_id",
        "group",
        "student_params_m",
        "param_reduction_pct",
        "student_gflops",
        "flop_reduction_pct",
        "auroc_img",
        "ap_img",
        "auroc_pix",
        "ap_pix",
        "aupro_30",
        "aupro_05",
        "latency_ms",
        "fps",
        "peak_vram_mb",
        "description",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Also compute mean across categories per exp_id if >1 category
    categories = sorted(list({r["category"] for r in records}))
    exp_ids = []
    for r in records:
        if r["exp_id"] not in exp_ids:
            exp_ids.append(r["exp_id"])

    lines = [
        "# KẾT QUẢ THỰC NGHIỆM & ABLATION STUDY TRÊN KAGGLE GPU",
        "",
        "## 1. Bảng Chi tiết Theo Danh mục (Per-Category Results)",
        "",
        "| Category | Variant | Params (M) | Δ Params | GFLOPs | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO@0.30 | AUPRO@0.05 | Latency (ms) | FPS |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        lines.append(
            f"| `{r['category']}` | **{r['exp_id']}** | {r['student_params_m']:.3f}M | "
            f"-{r['param_reduction_pct']:.1f}% | {r['student_gflops']:.3f} | "
            f"{r['auroc_img']:.4f} | {r['ap_img']:.4f} | {r['auroc_pix']:.4f} | "
            f"{r['ap_pix']:.4f} | {r['aupro_30']:.4f} | **{r['aupro_05']:.4f}** | "
            f"{r['latency_ms']:.2f} | {r['fps']:.1f} |"
        )

    if len(categories) > 1:
        lines.extend([
            "",
            f"## 2. Bảng Trung bình Trên {len(categories)} Danh mục ({', '.join(categories)})",
            "",
            "| Variant | Group | Params (M) | Δ Params | GFLOPs | Mean I-AUROC | Mean P-AUROC | Mean P-AP | Mean AUPRO@0.30 | Mean AUPRO@0.05 | Mean FPS |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for eid in exp_ids:
            sub = [r for r in records if r["exp_id"] == eid]
            if not sub:
                continue
            r0 = sub[0]
            mean_iauroc = float(np.mean([x["auroc_img"] for x in sub]))
            mean_pauroc = float(np.mean([x["auroc_pix"] for x in sub]))
            mean_pap = float(np.mean([x["ap_pix"] for x in sub]))
            mean_pro30 = float(np.mean([x["aupro_30"] for x in sub]))
            mean_pro05 = float(np.mean([x["aupro_05"] for x in sub]))
            mean_fps = float(np.mean([x["fps"] for x in sub]))
            lines.append(
                f"| **{eid}** | {r0['group']} | {r0['student_params_m']:.3f}M | "
                f"-{r0['param_reduction_pct']:.1f}% | {r0['student_gflops']:.3f} | "
                f"{mean_iauroc:.4f} | {mean_pauroc:.4f} | **{mean_pap:.4f}** | "
                f"{mean_pro30:.4f} | **{mean_pro05:.4f}** | {mean_fps:.1f} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return csv_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CS331 Kaggle GPU Benchmark & Ablation Suite.")
    parser.add_argument("--data-root", type=str, default="/kaggle/input/mvtec-ad")
    parser.add_argument(
        "--categories",
        type=str,
        default="carpet,bottle,screw",
        help="Comma-separated MVTec AD categories (e.g. 'carpet,bottle,screw')",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["baseline", "method1", "method2", "unified", "all"],
        help="Which experiment suite to execute",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable FP16 Automatic Mixed Precision on CUDA")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore cached checkpoints and retrain")
    parser.add_argument("--teacher-pretrained", dest="teacher_pretrained", action="store_true", default=True)
    parser.add_argument("--no-teacher-pretrained", dest="teacher_pretrained", action="store_false")
    parser.add_argument("--output-dir", type=str, default="results/kaggle_ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if sys.platform == "win32" else 2

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    exp_defs = get_experiment_definitions(args.suite)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Measure baseline student params & GFLOPs once for relative reduction calculation
    ref_model = STFPM(teacher_pretrained=False).to(device)
    base_params = sum(p.numel() for p in ref_model.student.parameters())
    base_gflops = count_conv_bn_flops(ref_model.student, img_size=args.img_size)
    del ref_model

    all_records: List[Dict[str, Any]] = []

    for cat in categories:
        print("\n" + "=" * 72)
        print(f"[*] CATEGORY: {cat.upper()} | Device: {device} | Suite: {args.suite}")
        print("=" * 72)

        train_ds = MVTecTrainDataset(
            root_dir=args.data_root,
            obj_name=cat,
            resize_shape=args.img_size,
            split="train",
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        val_ds = MVTecTrainDataset(
            root_dir=args.data_root,
            obj_name=cat,
            resize_shape=args.img_size,
            split="val",
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        test_ds = MVTecTestDataset(
            root_dir=args.data_root,
            obj_name=cat,
            resize_shape=args.img_size,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        calib_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if len(val_ds) > 0
            else None
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Cache of fitted SCGNormalStats per ckpt_key so we only calibrate once per trained model
        calib_cache: Dict[str, SCGNormalStats] = {}

        for exp in exp_defs:
            set_seed(args.seed)
            exp_id = exp["exp_id"]
            ckpt_key = exp["ckpt_key"]
            ckpt_dir = out_root / "checkpoints" / cat / ckpt_key

            # Build model & loss
            if exp["arch"] == "stfpm":
                model: nn.Module = STFPM(teacher_pretrained=args.teacher_pretrained).to(device)
            else:
                model = SlimSTFPM(
                    width_mult=exp["width_mult"],
                    proj_type=exp["proj_type"],
                    teacher_pretrained=args.teacher_pretrained,
                ).to(device)

            if exp["focal_gamma"] > 0.0:
                loss_fn: nn.Module = FocalSTFPMLoss(gamma=exp["focal_gamma"], beta=2.0).to(device)
            else:
                loss_fn = STFPMLoss().to(device)

            student_params = sum(p.numel() for p in model.student.parameters() if p.requires_grad)
            student_gflops = count_conv_bn_flops(model.student, img_size=args.img_size)
            param_red_pct = (1.0 - student_params / max(base_params, 1)) * 100.0
            flop_red_pct = (1.0 - student_gflops / max(base_gflops, 1e-6)) * 100.0

            # Train or reuse cached checkpoint
            train_info = train_student_checkpoint(
                model=model,
                loss_fn=loss_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                ckpt_dir=ckpt_dir,
                category=cat,
                variant_key=ckpt_key,
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                use_amp=args.amp,
                device=device,
                seed=args.seed,
                force_retrain=args.force_retrain,
            )

            # Configure Anomaly Mapper
            mapper = SCGSTFPMMapper(
                mode=exp["mapper_mode"],
                target_size=(args.img_size, args.img_size),
                baseline_sigma=exp["baseline_sigma"],
            )

            if exp["mapper_mode"] != "A0_baseline":
                if ckpt_key not in calib_cache:
                    print(f"  [Calibrate] Fitting zero-leakage SCGNormalStats on train/good for '{ckpt_key}'...")
                    stats = mapper.fit_calibration(model, calib_loader, device=device)
                    calib_cache[ckpt_key] = stats
                    with open(ckpt_dir / "scg_normal_stats.json", "w", encoding="utf-8") as sf:
                        json.dump(stats.to_dict(), sf, indent=2)
                else:
                    mapper.stats = calib_cache[ckpt_key]

            # Evaluate on test set
            eval_metrics = evaluate_model_with_mapper(
                model=model,
                mapper=mapper,
                test_loader=test_loader,
                img_size=args.img_size,
                device=device,
            )

            record = {
                "category": cat,
                "exp_id": exp_id,
                "group": exp["group"],
                "student_params": student_params,
                "student_params_m": round(student_params / 1e6, 3),
                "param_reduction_pct": round(param_red_pct, 1),
                "student_gflops": round(student_gflops, 3),
                "flop_reduction_pct": round(flop_red_pct, 1),
                "auroc_img": eval_metrics["auroc_img"],
                "ap_img": eval_metrics["ap_img"],
                "auroc_pix": eval_metrics["auroc_pix"],
                "ap_pix": eval_metrics["ap_pix"],
                "aupro_30": eval_metrics["aupro_30"],
                "aupro_05": eval_metrics["aupro_05"],
                "latency_ms": eval_metrics["latency_ms"],
                "fps": eval_metrics["fps"],
                "peak_vram_mb": train_info["peak_vram_mb"],
                "train_duration_sec": train_info["train_duration_sec"],
                "checkpoint_path": train_info["checkpoint_path"],
                "description": exp["description"],
            }
            all_records.append(record)

            # Save individual experiment JSON
            exp_out_dir = out_root / cat / exp_id
            exp_out_dir.mkdir(parents=True, exist_ok=True)
            with open(exp_out_dir / "metrics.json", "w", encoding="utf-8") as mf:
                json.dump(record, mf, indent=2)

            print(
                f"  [Result] {exp_id:<22} | Params: {record['student_params_m']:.3f}M (-{record['param_reduction_pct']:.1f}%) | "
                f"I-AUROC: {record['auroc_img']:.4f} | P-AP: {record['ap_pix']:.4f} | "
                f"AUPRO@0.05: {record['aupro_05']:.4f} | {record['fps']:.1f} FPS"
            )

    csv_path, md_path = write_summary_reports(all_records, out_root)
    print("\n" + "=" * 72)
    print(f"[*] All experiments completed! Summary reports saved to:")
    print(f"    - CSV : {csv_path}")
    print(f"    - MD  : {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
