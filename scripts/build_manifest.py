"""
Dataset & Schema Manifest Generator.
Generates immutable dataset provenance hashes, label ontology tables, and feature harmonization manifests.
"""

from pathlib import Path
import json
import pandas as pd
from src.hybrid_ids.utils.hashing import compute_sha256
from src.hybrid_ids.data.label_mapping import LabelHarmonizer
from src.hybrid_ids.data.harmonization import DatasetHarmonizer
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("build_manifest")


def main():
    root = get_project_root()
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Cryptographic Manifest
    logger.info("Computing cryptographic hashes for dataset files...")
    files_to_hash = {
        "edge_iiot_archive": root / "Edge-IIoTset dataset.zip",
        "cic_iot_archive": root / "archive.zip",
        "edge_iiot_raw_csv": root / "data" / "raw" / "edge_iiot" / "ML-EdgeIIoT-dataset.csv",
        "cic_iot_test_csv": root / "data" / "raw" / "cic_iot_2023" / "test.csv",
        "cic_iot_val_csv": root / "data" / "raw" / "cic_iot_2023" / "validation.csv"
    }

    manifest = {}
    for name, path in files_to_hash.items():
        if path.exists():
            manifest[name] = {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": compute_sha256(path)
            }
            logger.info(f"Verified {name}: {manifest[name]['sha256']}")

    with open(manifest_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Label Mapping Manifest
    logger.info("Exporting canonical label mapping manifest...")
    label_mapper = LabelHarmonizer()
    df_labels = label_mapper.get_manifest_dataframe()
    df_labels.to_csv(manifest_dir / "label_mapping_manifest.csv", index=False)

    # 3. Feature Harmonization Manifest
    logger.info("Exporting feature harmonization manifest...")
    df_features = DatasetHarmonizer.get_manifest_dataframe()
    df_features.to_csv(manifest_dir / "feature_harmonization_manifest.csv", index=False)

    logger.info("Manifest generation complete!")


if __name__ == "__main__":
    main()
