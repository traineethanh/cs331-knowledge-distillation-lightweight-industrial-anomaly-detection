#!/usr/bin/env python3
"""Automated Sequential Multi-Class Training & Evaluation Pipeline for DSKD Baseline.

Sequentially trains and evaluates Baseline B0-Clean across MVTec AD categories on GPU:
1. Iterates over categories one by one.
2. Trains on GPU for N epochs (default 200).
3. Evaluates on test set (Image AUROC, Image AP, Pixel AUROC, Pixel AP, AUPRO@0.30 & 0.05).
4. Cleans GPU memory completely after each category.
5. Incrementally updates master summary CSV, JSON, and Markdown tables with Paper comparison.
6. Supports '--skip_existing' so already completed categories (like 'carpet') are not retrained.

Usage:
    # Run all 15 categories sequentially
    python scripts/run_all_baseline.py --data_root /kaggle/input/datasets/ipythonx/mvtec-ad --categories all

    # Run Core-5 subset
    python scripts/run_all_baseline.py --data_root /kaggle/input/datasets/ipythonx/mvtec-ad --categories core5

    # Run specific categories
    python scripts/run_all_baseline.py --data_root /kaggle/input/datasets/ipythonx/mvtec-ad --categories bottle tile screw
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Paper reference results for MVTec AD (arXiv:2402.00448v1 - Tables 1 & 2, p. 8)
PAPER_BENCHMARKS: Dict[str, Dict[str, float]] = {
    # Textures
    "carpet": {"paper_img_auroc": 100.0, "paper_pix_auroc": 99.2, "paper_pro": 97.7},
    "grid": {"paper_img_auroc": 100.0, "paper_pix_auroc": 99.4, "paper_pro": 97.7},
    "leather": {"paper_img_auroc": 100.0, "paper_pix_auroc": 99.4, "paper_pro": 99.1},
    "tile": {"paper_img_auroc": 98.4, "paper_pix_auroc": 94.5, "paper_pro": 88.2},
    "wood": {"paper_img_auroc": 99.6, "paper_pix_auroc": 94.3, "paper_pro": 91.6},
    # Objects
    "bottle": {"paper_img_auroc": 100.0, "paper_pix_auroc": 98.7, "paper_pro": 96.4},
    "cable": {"paper_img_auroc": 99.2, "paper_pix_auroc": 97.5, "paper_pro": 92.5},
    "capsule": {"paper_img_auroc": 93.1, "paper_pix_auroc": 98.7, "paper_pro": 94.4},
    "hazelnut": {"paper_img_auroc": 100.0, "paper_pix_auroc": 98.8, "paper_pro": 94.6},
    "metal_nut": {"paper_img_auroc": 99.7, "paper_pix_auroc": 96.7, "paper_pro": 89.5},
    "pill": {"paper_img_auroc": 96.2, "paper_pix_auroc": 97.1, "paper_pro": 94.7},
    "screw": {"paper_img_auroc": 97.1, "paper_pix_auroc": 99.4, "paper_pro": 94.2},
    "toothbrush": {"paper_img_auroc": 98.9, "paper_pix_auroc": 99.1, "paper_pro": 94.5},
    "transistor": {"paper_img_auroc": 97.5, "paper_pix_auroc": 91.5, "paper_pro": 80.2},
    "zipper": {"paper_img_auroc": 97.4, "paper_pix_auroc": 97.5, "paper_pro": 93.5},
}

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]
ALL_CATEGORIES = TEXTURE_CATEGORIES + OBJECT_CATEGORIES
CORE5_CATEGORIES = ["carpet", "bottle", "tile", "screw", "toothbrush"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Multi-Class Runner for DSKD Baseline")
    parser.add_argument("--config", type=str, default="configs/baseline_b0.yaml", help="Path to config YAML")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec AD dataset directory")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Categories to run: 'all', 'core5', 'textures', 'objects', or a list of category names",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Epochs per category (default: 200)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline_b0", help="Root checkpoint dir")
    parser.add_argument("--output_dir", type=str, default="results/baseline_b0", help="Root results dir")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip category if evaluation result already exists (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Force retrain even if results exist",
    )
    return parser.parse_args()


def resolve_categories(raw_categories: List[str]) -> List[str]:
    """Resolves category argument into an ordered list of valid category names."""
    resolved: List[str] = []
    for item in raw_categories:
        lower = item.lower()
        if lower == "all":
            for cat in ALL_CATEGORIES:
                if cat not in resolved:
                    resolved.append(cat)
        elif lower == "core5":
            for cat in CORE5_CATEGORIES:
                if cat not in resolved:
                    resolved.append(cat)
        elif lower == "textures":
            for cat in TEXTURE_CATEGORIES:
                if cat not in resolved:
                    resolved.append(cat)
        elif lower == "objects":
            for cat in OBJECT_CATEGORIES:
                if cat not in resolved:
                    resolved.append(cat)
        elif lower in ALL_CATEGORIES:
            if lower not in resolved:
                resolved.append(lower)
        else:
            print(f"[WARNING] Unrecognized category '{item}'. Skipping.")
    return resolved


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Formats DataFrame as a GitHub-flavored Markdown table without requiring tabulate."""
    if df.empty:
        return ""
    headers = [str(col) for col in df.columns]
    rows = [[str(val) for val in row] for row in df.values]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def update_summary_tables(output_dir: Path) -> pd.DataFrame:
    """Scans output_dir for all category result JSON files and updates summary files."""
    records: List[Dict[str, Any]] = []

    for cat in ALL_CATEGORIES:
        json_file = output_dir / cat / f"{cat}_eval_metrics.json"
        if not json_file.exists():
            # Check legacy/flat location
            json_file = output_dir / f"{cat}_eval_metrics.json"

        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                paper_ref = PAPER_BENCHMARKS.get(cat, {})
                img_auroc = float(data.get("image_auroc", 0.0)) * 100.0
                pix_auroc = float(data.get("pixel_auroc", 0.0)) * 100.0
                pix_pro = float(data.get("pixel_aupro_30", 0.0)) * 100.0
                pix_pro_05 = float(data.get("pixel_aupro_05", 0.0)) * 100.0

                records.append({
                    "category": cat,
                    "group": "Texture" if cat in TEXTURE_CATEGORIES else "Object",
                    "img_auroc": round(img_auroc, 2),
                    "paper_img_auroc": paper_ref.get("paper_img_auroc", 0.0),
                    "delta_img": round(img_auroc - paper_ref.get("paper_img_auroc", 0.0), 2),
                    "pix_auroc": round(pix_auroc, 2),
                    "paper_pix_auroc": paper_ref.get("paper_pix_auroc", 0.0),
                    "delta_pix": round(pix_auroc - paper_ref.get("paper_pix_auroc", 0.0), 2),
                    "pix_pro_30": round(pix_pro, 2),
                    "paper_pro_30": paper_ref.get("paper_pro", 0.0),
                    "delta_pro": round(pix_pro - paper_ref.get("paper_pro", 0.0), 2),
                    "pix_pro_05": round(pix_pro_05, 2),
                    "num_test": data.get("num_test_samples", 0),
                    "num_anom": data.get("num_anomalous_samples", 0),
                })
            except Exception as e:
                print(f"[WARNING] Error reading {json_file}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Save CSV
    summary_csv = output_dir / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)

    # Compute group averages
    texture_df = df[df["group"] == "Texture"]
    object_df = df[df["group"] == "Object"]

    summary_text = [
        "# DSKD Baseline (B0-Clean) Benchmark Summary on MVTec AD",
        f"\n**Total Categories Completed:** {len(df)} / 15",
        "\n### Per-Category Results vs Paper (Table 1 & Table 2)\n",
        df_to_markdown_simple(df),
        "\n### Group Means\n",
    ]

    means_records = []
    if len(texture_df) > 0:
        means_records.append({
            "Group": f"Texture Mean ({len(texture_df)}/5)",
            "Img AUROC": f"{texture_df['img_auroc'].mean():.2f}%",
            "Paper Img": f"{texture_df['paper_img_auroc'].mean():.2f}%",
            "Pix AUROC": f"{texture_df['pix_auroc'].mean():.2f}%",
            "Paper Pix": f"{texture_df['paper_pix_auroc'].mean():.2f}%",
            "PRO@0.30": f"{texture_df['pix_pro_30'].mean():.2f}%",
            "Paper PRO": f"{texture_df['paper_pro_30'].mean():.2f}%",
            "PRO@0.05": f"{texture_df['pix_pro_05'].mean():.2f}%",
        })
    if len(object_df) > 0:
        means_records.append({
            "Group": f"Object Mean ({len(object_df)}/10)",
            "Img AUROC": f"{object_df['img_auroc'].mean():.2f}%",
            "Paper Img": f"{object_df['paper_img_auroc'].mean():.2f}%",
            "Pix AUROC": f"{object_df['pix_auroc'].mean():.2f}%",
            "Paper Pix": f"{object_df['paper_pix_auroc'].mean():.2f}%",
            "PRO@0.30": f"{object_df['pix_pro_30'].mean():.2f}%",
            "Paper PRO": f"{object_df['paper_pro_30'].mean():.2f}%",
            "PRO@0.05": f"{object_df['pix_pro_05'].mean():.2f}%",
        })
    if len(df) > 0:
        means_records.append({
            "Group": f"Overall Mean ({len(df)}/15)",
            "Img AUROC": f"{df['img_auroc'].mean():.2f}%",
            "Paper Img": f"{df['paper_img_auroc'].mean():.2f}%",
            "Pix AUROC": f"{df['pix_auroc'].mean():.2f}%",
            "Paper Pix": f"{df['paper_pix_auroc'].mean():.2f}%",
            "PRO@0.30": f"{df['pix_pro_30'].mean():.2f}%",
            "Paper PRO": f"{df['paper_pro_30'].mean():.2f}%",
            "PRO@0.05": f"{df['pix_pro_05'].mean():.2f}%",
        })

    means_df = pd.DataFrame(means_records)
    summary_text.append(df_to_markdown_simple(means_df))

    summary_md = output_dir / "summary_table.md"
    summary_md.write_text("\n".join(summary_text), encoding="utf-8")

    return df


def main() -> None:
    args = parse_args()
    categories = resolve_categories(args.categories)

    if not categories:
        print("[ERROR] No valid categories selected. Exiting.")
        sys.exit(1)

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}")
        sys.exit(1)

    checkpoint_root = Path(args.checkpoint_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("DSKD Sequential Multi-Class Pipeline")
    print(f"  Target Categories ({len(categories)}): {categories}")
    print(f"  Data Root:    {data_root}")
    print(f"  Device:       {args.device}")
    print(f"  Epochs/Cat:   {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  Skip Existing:{args.skip_existing}")
    print(f"  Checkpoints:  {checkpoint_root}")
    print(f"  Results:      {output_root}")
    print("=" * 75)

    pipeline_start = time.time()
    completed_cats = 0
    skipped_cats = 0
    failed_cats: List[Tuple[str, str]] = []

    for idx, category in enumerate(categories, 1):
        cat_start = time.time()
        cat_ckpt_dir = checkpoint_root / category
        cat_out_dir = output_root / category
        cat_ckpt_dir.mkdir(parents=True, exist_ok=True)
        cat_out_dir.mkdir(parents=True, exist_ok=True)

        best_ckpt = cat_ckpt_dir / "best_checkpoint.pth"
        eval_json = cat_out_dir / f"{category}_eval_metrics.json"

        print(f"\n>>> [{idx}/{len(categories)}] Processing category: '{category.upper()}'")

        # Check if already completed
        if args.skip_existing and eval_json.exists():
            print(f"    [SKIP] Found existing evaluation results: {eval_json}")
            skipped_cats += 1
            update_summary_tables(output_root)
            continue

        # Step 1: Train (if best_ckpt doesn't exist)
        if not best_ckpt.exists():
            print(f"    [TRAIN] Training {category} for {args.epochs} epochs on GPU...")
            train_cmd = [
                sys.executable,
                "scripts/train.py",
                "--config", args.config,
                "--category", category,
                "--data_root", str(data_root),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--device", args.device,
                "--output_dir", str(checkpoint_root),
            ]
            ret = subprocess.run(train_cmd)
            if ret.returncode != 0:
                print(f"    [ERROR] Training failed for category '{category}' with exit code {ret.returncode}")
                failed_cats.append((category, "Train failed"))
                continue
        else:
            print(f"    [TRAIN SKIP] Found existing checkpoint: {best_ckpt}")

        # Step 2: Evaluate
        print(f"    [EVAL] Evaluating {category} on test set...")
        eval_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint", str(best_ckpt),
            "--config", args.config,
            "--category", category,
            "--data_root", str(data_root),
            "--batch_size", str(args.batch_size),
            "--device", args.device,
            "--output_dir", str(cat_out_dir),
        ]
        ret = subprocess.run(eval_cmd)
        if ret.returncode != 0:
            print(f"    [ERROR] Evaluation failed for category '{category}' with exit code {ret.returncode}")
            failed_cats.append((category, "Evaluation failed"))
            continue

        cat_duration = time.time() - cat_start
        print(f"    [DONE] Category '{category}' completed in {cat_duration / 60:.2f} minutes.")
        completed_cats += 1

        # Step 3: Update Master Summary
        update_summary_tables(output_root)

    total_duration = time.time() - pipeline_start
    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION FINISHED")
    print(f"  Total Duration: {total_duration / 3600:.2f} hours ({total_duration / 60:.2f} mins)")
    print(f"  Completed:      {completed_cats}")
    print(f"  Skipped:        {skipped_cats}")
    print(f"  Failed:         {len(failed_cats)}")
    if failed_cats:
        print(f"  Failed details: {failed_cats}")
    print("=" * 75)

    # Print final summary table
    df_summary = update_summary_tables(output_root)
    if not df_summary.empty:
        summary_md_file = output_root / "summary_table.md"
        if summary_md_file.exists():
            print("\n" + summary_md_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
