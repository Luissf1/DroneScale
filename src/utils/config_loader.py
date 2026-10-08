"""
Configuration Loader
====================

Loads YAML configuration files and provides dot-accessible defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default_config.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Optional path to a YAML config. If None, uses the default.

    Returns:
        Dictionary with configuration values.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path