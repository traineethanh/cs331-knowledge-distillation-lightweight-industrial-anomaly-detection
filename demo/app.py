"""Interactive Web Application for Industrial Anomaly Detection using Gradio.

Designed for CPU-based execution on Hugging Face Spaces.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# 1. Independent Inference Engine (Decoupled from Web UI)
# ---------------------------------------------------------------------------


class LightweightSTFPM(nn.Module):
    """Self-contained ResNet-18 Student-Teacher Feature Extractor for CPU Demo."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.init_block = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x: torch.Tensor):
        x = self.init_block(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


class AnomalyDetectionEngine:
    """Manages model loading, preprocessing, inference, and visualization."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.device = torch.device("cpu")
        self.img_size = 256

        # Build transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("[*] Loading Teacher and Student models onto CPU...")
        self.teacher = LightweightSTFPM(pretrained=True).to(self.device).eval()
        self.student = LightweightSTFPM(pretrained=False).to(self.device).eval()

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.student.parameters():
            p.requires_grad = False

        self.is_trained = False
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[*] Loading student checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("student_state_dict", ckpt)
            self.student.load_state_dict(state_dict)
            self.is_trained = True
            print("[*] Checkpoint loaded successfully.")
        else:
            print("[!] No custom checkpoint found. Using ImageNet-initialized weights for demonstration.")

    def predict(
        self,
        image: Image.Image,
        category: str,
        threshold: float,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, str, str, str]:
        """Run anomaly detection inference on an uploaded image."""
        if image is None:
            raise ValueError("No image provided.")

        start_time = time.perf_counter()

        # Convert to RGB PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        orig_rgb = image.convert("RGB").resize((self.img_size, self.img_size))

        # Tensor preprocessing
        tensor_in = self.transform(orig_rgb).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            t_feats = self.teacher(tensor_in)
            s_feats = self.student(tensor_in)

            # Compute multi-scale difference map
            score_map = torch.ones((1, 1, 64, 64), device=self.device)
            for tf, sf in zip(t_feats, s_feats):
                tn = F.normalize(tf, p=2, dim=1)
                sn = F.normalize(sf, p=2, dim=1)
                diff = 0.5 * torch.sum((tn - sn) ** 2, dim=1, keepdim=True)
                diff_interp = F.interpolate(diff, size=(64, 64), mode="bilinear", align_corners=False)
                score_map = score_map * diff_interp

            final_map = F.interpolate(
                score_map, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Anomaly score: max of anomaly map
        raw_score = float(np.max(final_map))
        # Normalization for display
        norm_map = (final_map - final_map.min()) / (final_map.max() - final_map.min() + 1e-8)

        # Colormap generation (jet) compatible with both modern and older matplotlib
        try:
            import matplotlib
            color_mapper = matplotlib.colormaps["jet"]
        except (AttributeError, KeyError):
            color_mapper = cm.get_cmap("jet")
        heatmap_rgba = color_mapper(norm_map)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_rgb)

        # Alpha overlay blending (50% original, 50% heatmap)
        orig_np = np.array(orig_rgb).astype(np.float32)
        overlay_np = (0.5 * orig_np + 0.5 * heatmap_rgb).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_np)

        # Inspection decision
        is_anomaly = raw_score >= threshold
        decision_str = "DEFECT (Anomaly Detected)" if is_anomaly else "PASS (Normal Product)"
        decision_color = "red" if is_anomaly else "green"

        score_display = f"Anomaly Score: {raw_score:.4f} (Threshold: {threshold:.2f})"
        latency_display = f"CPU Latency: {latency_ms:.1f} ms ({1000.0 / max(latency_ms, 1e-3):.1f} FPS)"

        return orig_rgb, heatmap_pil, overlay_pil, decision_str, score_display, latency_display


# Initialize engine once at module load time
CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT", "demo/checkpoints/stfpm_carpet.pth")
ENGINE = AnomalyDetectionEngine(checkpoint_path=CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# 2. Gradio Web Interface
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

    with gr.Blocks(theme=theme, title="Industrial Anomaly Detection") as demo:
        gr.Markdown(
            """
            # 🔍 Industrial Anomaly Detection via Knowledge Distillation
            **CS331 — Advanced Computer Vision Project** | Team: Thành & Luân  
            *Upload an industrial product image to inspect for anomalies and localize defective regions.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Product Image")
                category_dropdown = gr.Dropdown(
                    choices=["carpet", "bottle", "screw"],
                    value="carpet",
                    label="Product Category",
                )
                threshold_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.00,
                    value=0.35,
                    step=0.01,
                    label="Decision Threshold (τ)",
                    info="Adjust sensitivity: lower thresholds flag subtle deviations; higher thresholds require prominent defects.",
                )
                inspect_btn = gr.Button("⚡ Inspect Surface", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_decision = gr.Textbox(label="Inspection Decision", interactive=False)
                    out_score = gr.Textbox(label="Anomaly Score", interactive=False)
                    out_latency = gr.Textbox(label="Inference Latency", interactive=False)

                with gr.Row():
                    out_orig = gr.Image(label="Processed Input (256x256)", interactive=False)
                    out_heatmap = gr.Image(label="Anomaly Heatmap", interactive=False)
                    out_overlay = gr.Image(label="Localization Overlay", interactive=False)

        gr.Markdown(
            """
            ---
            ### ℹ️ Scientific Methodology & Limitations
            - **Detection Principle:** A frozen ResNet-18 teacher transfers multi-scale pyramid representations to a compact student network trained strictly on normal images. Defective patterns induce sharp Teacher–Student feature divergence.
            - **Limitations:** Specular reflections, lens dust, or extreme illumination shifts can produce localized false positives. Highly textured surfaces require normal-distribution statistical calibration.
            """
        )

        def run_inspection(img, cat, thresh):
            if img is None:
                return None, None, None, "No image uploaded", "N/A", "N/A"
            return ENGINE.predict(img, cat, thresh)

        inspect_btn.click(
            fn=run_inspection,
            inputs=[input_image, category_dropdown, threshold_slider],
            outputs=[out_orig, out_heatmap, out_overlay, out_decision, out_score, out_latency],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
