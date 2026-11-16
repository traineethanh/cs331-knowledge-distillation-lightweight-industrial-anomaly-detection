"""Evaluation Script for CDA Baseline (`B0`), Direction A (`A1`-`A3`), and Direction B (`B1`-`B3`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTestDataset, MVTecTrainDataset
from anomaly_detection.inference import CDAAnomalyMapper
from anomaly_detection.losses import CDALoss
from anomaly_detection.models.baseline import TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.training import CDATrainer, load_checkpoint
from anomaly_detection.utils import load_yaml_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CDA / LiteCDA checkpoint.")
    parser.add_argument("--config", type=str, default="configs/baseline/cda_mvtec.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--fusion-mode", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    data_root = args.data_root or cfg["dataset"]["data_root"]
    img_size = int(cfg["dataset"].get("img_size", 256))
    variant = args.variant or cfg["model"].get("variant", "B0")
    infer_cfg = cfg.get("inference", {})
    fusion_mode = args.fusion_mode or infer_cfg.get("fusion_mode", "A0_baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TeacherEncoder(pretrained=bool(cfg["model"].get("teacher_pretrained", True))).to(device).eval()
    student = build_student_model(variant=variant).to(device)
    load_checkpoint(args.checkpoint, student=student, map_location=device)
    student.eval()

    mapper = CDAAnomalyMapper(
        img_size=img_size,
        fusion_mode=fusion_mode,
        gaussian_sigma=float(infer_cfg.get("gaussian_sigma", 4.0)),
        reliability_eta=float(infer_cfg.get("reliability_eta", 0.5)),
        scale_sigmas=infer_cfg.get("scale_sigmas", [2.0, 3.0, 4.0]),
        top_q_percent=float(infer_cfg.get("top_q_percent", 0.005)),
    )

    if fusion_mode != "A0_baseline":
        train_ds = MVTecTrainDataset(root_dir=data_root, obj_name=args.category, resize_shape=img_size)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
        mapper.mapper.fit_normal_stats(encoder, student, train_loader, device=device)

    test_ds = MVTecTestDataset(root_dir=data_root, obj_name=args.category, resize_shape=img_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    dummy_opt = torch.optim.Adam(student.parameters(), lr=0.005)
    trainer = CDATrainer(
        encoder=encoder,
        student=student,
        loss_fn=CDALoss(),
        optimizer=dummy_opt,
        device=device,
        output_dir=Path("./outputs/eval_tmp"),
        img_size=img_size,
    )
    metrics_dict, _, _, _ = trainer.evaluate(test_loader, mapper=mapper, include_aupro=True)
    output = {
        "category": args.category,
        "variant": variant,
        "fusion_mode": fusion_mode,
        "checkpoint": args.checkpoint,
        "metrics": metrics_dict,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
