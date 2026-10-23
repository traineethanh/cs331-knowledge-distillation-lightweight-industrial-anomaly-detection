"""Stage 1 Environment Check Script.

Inspects Python, PyTorch, CUDA/GPU availability, package versions, and disk space.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch


def check_environment() -> dict:
    disk = shutil.disk_usage(ROOT)
    packages = {}
    for pkg in ("torch", "torchvision", "numpy", "scipy", "sklearn", "PIL", "yaml", "tqdm", "matplotlib"):
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = "MISSING"

    cuda_avail = torch.cuda.is_available()
    gpu_info = []
    if cuda_avail:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )

    report = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "cuda_available": cuda_avail,
        "cuda_version": torch.version.cuda,
        "gpus": gpu_info,
        "packages": packages,
        "workspace_free_disk_gb": round(disk.free / (1024**3), 2),
    }
    return report


if __name__ == "__main__":
    info = check_environment()
    print(json.dumps(info, indent=2))
