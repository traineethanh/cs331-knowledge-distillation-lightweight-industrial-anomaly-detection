"""Stage 2 Dataset Smoke Test and Validation Script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection.data import MVTEC_CATEGORIES, validate_mvtec_category


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MVTec AD dataset structure and splits.")
    parser.add_argument("--data-root", type=str, default="./datasets/mvtec")
    parser.add_argument("--categories", nargs="+", default=["carpet", "bottle", "screw"])
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()

    categories = MVTEC_CATEGORIES if args.categories == ["all"] else args.categories
    results = []
    for cat in categories:
        summary = validate_mvtec_category(
            root_dir=args.data_root,
            obj_name=cat,
            resize_shape=args.img_size,
        )
        results.append(summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
