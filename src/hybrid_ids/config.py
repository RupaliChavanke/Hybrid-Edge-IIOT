"""
Configuration management module for Hybrid IDS.
Loads and validates modular YAML configurations from configs/ directory.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SystemConfig:
    """Consolidated configuration container for Hybrid IDS."""

    def __init__(self, configs_dir: Optional[str | Path] = None):
        if configs_dir is None:
            self.configs_dir = get_project_root() / "configs"
        else:
            self.configs_dir = Path(configs_dir)

        self.data = self._load("data.yaml")
        self.preprocessing = self._load("preprocessing.yaml")
        self.models = self._load("models.yaml")
        self.ensemble = self._load("ensemble.yaml")
        self.evaluation = self._load("evaluation.yaml")
        self.streaming = self._load("streaming.yaml")

    def _load(self, filename: str) -> Dict[str, Any]:
        p = self.configs_dir / filename
        if p.exists():
            return load_yaml(p)
        return {}


# Global default configuration instance
config = SystemConfig()
