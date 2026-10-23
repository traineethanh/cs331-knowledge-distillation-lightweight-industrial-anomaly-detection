"""YAML configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Loads and validates a YAML configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level of {cfg_path}")
    return data


def save_yaml_config(config: Dict[str, Any], path: str | Path) -> Path:
    """Writes a configuration dictionary to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return target
