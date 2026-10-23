"""
Standalone Dataset Inspection and Quality Audit Script.
Scans Edge-IIoTset and CICIoT2023 raw data, computes missingness, constant columns,
class distributions, and produces data_quality_report.md.
"""

from pathlib import Path
import json
import pandas as pd
from src.hybrid_ids.data.loaders import DataLoader
from src.hybrid_ids.data.validation import DataValidator
from src.hybrid_ids.config import get_project_root
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("inspect_datasets")


def main():
    root = get_project_root()
    loader = DataLoader(root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Auditing Edge-IIoTset...")
    df_edge = loader.load_edge_iiot(mode="full")
    edge_audit = DataValidator.audit_quality(df_edge, target_col="Attack_type")

    logger.info("Auditing CICIoT2023 (validation partition)...")
    df_cic = loader.load_cic_iot_2023(split="validation", mode="dev", sample_size=50000)
    cic_audit = DataValidator.audit_quality(df_cic, target_col="label")

    # Generate reports/data_quality_report.md
    report_path = reports_dir / "data_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Forensic Data Quality and Schema Validation Report\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("## 1. Edge-IIoTset Machine Learning Partition\n\n")
        f.write(f"- **Total Rows:** {edge_audit['total_rows']:,}\n")
        f.write(f"- **Total Columns:** {edge_audit['total_cols']}\n")
        f.write(f"- **Memory Footprint:** {edge_audit['memory_mb']} MB\n")
        f.write(f"- **Duplicate Rows:** {edge_audit['duplicate_rows']:,}\n")
        f.write(f"- **Constant Columns ({len(edge_audit['constant_columns'])}):** {', '.join(edge_audit['constant_columns'][:10])}...\n\n")

        f.write("### Target Class Distribution (Attack_type):\n\n")
        f.write("| Attack Class | Count | Percentage |\n| :--- | :--- | :--- |\n")
        for cls_name, info in edge_audit["target_distribution"].items():
            f.write(f"| {cls_name} | {info['count']:,} | {info['percentage']}% |\n")

        f.write("\n## 2. CICIoT2023 Telemetry Dataset\n\n")
        f.write(f"- **Sampled Rows:** {cic_audit['total_rows']:,}\n")
        f.write(f"- **Total Columns:** {cic_audit['total_cols']}\n")
        f.write(f"- **Memory Footprint:** {cic_audit['memory_mb']} MB\n")
        f.write(f"- **Duplicate Rows in Sample:** {cic_audit['duplicate_rows']:,}\n")
        f.write(f"- **Constant Columns in Sample:** {', '.join(cic_audit['constant_columns'])}\n\n")

        f.write("### Top 15 Attack Classes in CICIoT2023 Sample:\n\n")
        f.write("| Attack Class | Count | Percentage |\n| :--- | :--- | :--- |\n")
        for i, (cls_name, info) in enumerate(cic_audit["target_distribution"].items()):
            if i >= 15:
                break
            f.write(f"| {cls_name} | {info['count']:,} | {info['percentage']}% |\n")

    logger.info(f"Audit completed. Quality report written to: {report_path}")


if __name__ == "__main__":
    main()
