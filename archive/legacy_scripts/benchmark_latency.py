"""Hardware Latency, FLOPs/MACs, Parameter Count, and Memory Benchmark Script (`B0` vs `B1`-`B3`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anomaly_detection.models.baseline import TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.utils import profile_model_cost, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CDA and LiteCDA variants on identical hardware.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["B0", "B1_aux_csd", "B2_sep_adapter", "B3_cda_lite"],
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--pretrained-teacher", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TeacherEncoder(pretrained=args.pretrained_teacher)

    results = []
    for var in args.variants:
        student = build_student_model(variant=var)
        cost = profile_model_cost(
            encoder=encoder,
            student=student,
            img_size=args.img_size,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        row = {"variant": var, **cost}
        results.append(row)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
