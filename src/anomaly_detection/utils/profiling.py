"""Parameter, FLOPs/MACs, Latency, and Peak GPU Memory profiling utilities."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Counts total or trainable parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def measure_student_flops(
    student: nn.Module,
    img_size: int = 256,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Measures exact FLOPs and MACs of the student network using PyTorch `FlopCounterMode`."""
    student = student.to(device).eval()
    h1, h2, h3 = img_size // 4, img_size // 8, img_size // 16
    e1 = torch.randn(1, 256, h1, h1, device=device)
    e2 = torch.randn(1, 512, h2, h2, device=device)
    e3 = torch.randn(1, 1024, h3, h3, device=device)

    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            _ = student(e1, e2, e3)
        total_flops = int(flop_counter.get_total_flops())

    return {
        "student_flops_g": round(total_flops / 1e9, 4),
        "student_macs_g": round(total_flops / 2e9, 4),
    }


def profile_model_cost(
    encoder: nn.Module,
    student: nn.Module,
    img_size: int = 256,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 30,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Profiles parameters, FLOPs/MACs, forward latency (ms), and peak memory."""
    dev = torch.device(device)
    encoder = encoder.to(dev).eval()
    student = student.to(dev).eval()

    teacher_params = count_parameters(encoder, trainable_only=False)
    student_total_params = count_parameters(student, trainable_only=False)
    if hasattr(student, "active_inference_parameters"):
        student_active_params = int(student.active_inference_parameters())
    else:
        student_active_params = student_total_params

    flop_info = measure_student_flops(student, img_size=img_size, device=dev)

    dummy_img = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    with torch.no_grad():
        for _ in range(warmup):
            e1, e2, e3 = encoder(dummy_img)
            _ = student(e1, e2, e3)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        timings_ms = []
        for _ in range(iters):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            e1, e2, e3 = encoder(dummy_img)
            _ = student(e1, e2, e3)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0 / batch_size)

    arr = np.asarray(timings_ms, dtype=np.float64)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(dev) / (1024**2), 2)
        if dev.type == "cuda"
        else 0.0
    )

    return {
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
        "img_size": img_size,
        "batch_size": batch_size,
        "teacher_params": teacher_params,
        "student_trainable_params": student_total_params,
        "student_active_infer_params": student_active_params,
        **flop_info,
        "latency_forward_ms_mean": round(float(arr.mean()), 3),
        "latency_forward_ms_std": round(float(arr.std()), 3),
        "latency_forward_ms_p95": round(float(np.percentile(arr, 95)), 3),
        "throughput_fps": round(float(1000.0 / max(arr.mean(), 1e-6)), 2),
        "peak_vram_mb": peak_vram_mb,
    }
