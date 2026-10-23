# Repository Cleanup & Duplicate Removal Summary (Phase 15)

## 1. Audit & Inventory Metrics
Prior to restructuring, the repository underwent an exhaustive cryptographic audit:
- **Total Tracked Files Scanned:** 255 files
- **Total Space:** 42 GB used, 405 GB available
- **Exact Duplicate Groups Identified:** 24 groups
- **Near-Duplicate AST Modules:** 11 groups
- **Protected Files Identified:** 19 critical assets (Raw datasets, manifests, trained models, licensing)

---

## 2. Protected Files (Non-Disposable)
The following resources were classified as immutable and protected from modification or deletion:
- `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv` (157,800 records, SHA-256 verified)
- `data/raw/cic_iot_2023/test.csv` (1,180,000 records)
- `data/raw/cic_iot_2023/validation.csv` (1,180,000 records)
- `Edge-IIoTset dataset.zip` & `archive.zip` (Original source archives)
- `models/registry/*.pkl` (Trained model artifacts)
- `data/manifests/*.csv`, `*.json` (Data provenance manifests)
- `pyproject.toml`, `requirements.txt`, `.gitignore`

---

## 3. Quarantined Legacy Modules
In accordance with Phase 1 and Phase 14 safety procedures, legacy directories were moved to a timestamped, reversible quarantine directory:
`archive/quarantine_20261022/`

| Quarantined Directory / File | Superseded By | Reason for Retirement |
|---|---|---|
| `preprocessing/` | `src/hybrid_ids/preprocessing/` | Legacy unvectorized scripts replaced with modular pipeline cleaner |
| `training/` | `src/hybrid_ids/training/` | Superseded by GBDT training loop with validation early stopping |
| `evaluation/` | `src/hybrid_ids/evaluation/` | Superseded by multi-class metrics, McNemar testing, and robustness engine |
| `visualization/` | `app/components/` & `artifacts/figures/` | Obsolete matplotlib plotting scripts superseded by interactive Plotly UI |
| `dashboard/` | `app/dashboard.py` | 40+ legacy Streamlit pages consolidated into clean 10-tab SOC dashboard |
| `models/*.py` (BiGRU, CNN, GhostModule) | `src/hybrid_ids/models/` | Legacy deep learning models causing thread segmentation faults on macOS Python 3.14 OpenMP |

---

## 4. Verification Check
Following file moves to `archive/quarantine_20261022/`:
- **Pytest Suite:** 23 / 23 tests passed in 1.50s.
- **Entry Points:** `train.py`, `app.py`, and `scripts/*.py` execute without import errors.
- **Dangling Imports:** 0 broken or unresolved imports detected.
