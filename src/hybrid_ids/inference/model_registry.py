"""
Model Registry and Artifact Serialization Module.
Saves and loads trained models, ensembles, scalers, and encoders with SHA-256 integrity verification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
from src.hybrid_ids.utils.hashing import compute_sha256
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("model_registry")


class ModelRegistry:
    """Manages versioned, cryptographic model artifact persistence and reloading."""

    def __init__(self, registry_dir: Optional[Path] = None):
        self.registry_dir = registry_dir or (get_project_root() / "models" / "registry")
        self.manifest_dir = get_project_root() / "models" / "manifests"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, obj: Any, artifact_name: str, version: str = "v1") -> Tuple[Path, str]:
        """Serializes a Python object and calculates its SHA-256 digest."""
        filename = f"{artifact_name}_{version}.pkl"
        path = self.registry_dir / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

        digest = compute_sha256(path)
        logger.info(f"Saved artifact: {path} (SHA-256: {digest})")
        return path, digest

    def load_artifact(self, artifact_name: str, version: str = "v1", expected_sha256: Optional[str] = None) -> Any:
        """Loads and verifies an artifact from disk."""
        filename = f"{artifact_name}_{version}.pkl"
        path = self.registry_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        if expected_sha256:
            actual_digest = compute_sha256(path)
            if actual_digest != expected_sha256:
                raise ValueError(f"Integrity check failed for {path}! Expected {expected_sha256}, got {actual_digest}")

        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save_manifest(self, manifest_data: Dict[str, Any], manifest_name: str = "model_manifest") -> Path:
        """Saves a JSON manifest of all registered model artifacts and metadata."""
        path = self.manifest_dir / f"{manifest_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Saved registry manifest: {path}")
        return path
