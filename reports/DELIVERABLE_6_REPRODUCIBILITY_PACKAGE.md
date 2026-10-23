# Deliverable 6: Reproducibility Package and Cryptographic Manifest

**Manuscript Title:** *Empirical Feature-Collision Analysis, Leakage-Safe Optimization, and Calibrated Boosting Ensembles for IoT/IIoT Intrusion Detection*  
**Audited Repository:** `/Users/rupesh/Documents/Paper/Hybrid-Edge-IIOT`  
**Execution Environment:** Apple Silicon M2 (8 CPU cores, 16.0 GB unified memory), macOS Darwin 24.3.0  
**Software Runtime:** Python 3.14.7, CatBoost 1.2.10, XGBoost 3.4.1, LightGBM 4.7.0, scikit-learn 1.9.1, Pandas 2.2.3, NumPy 2.2.3, Tectonic 0.15.0  
**Date:** October 24, 2026  

---

## 1. Cryptographic SHA-256 Artifact Manifest

| Artifact Path | Size (Bytes) | Cryptographic SHA-256 Digest | Description / Role |
| :--- | :--- | :--- | :--- |
| `data/raw/edge_iiot/ML-EdgeIIoT-dataset.csv` | 82,184,390 | `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` | Raw Edge-IIoTset CSV ($157,800 \times 63$) |
| `data/raw/cic_iot_2023/test.csv` | 347,958,926 | `15aeb1f47a7bd3e3fc394d18d7b196b3640329abaae65d5220df4def6119bfc6` | Raw CIC-IoT-2023 Test CSV ($1,176,851 \times 47$) |
| `data/raw/cic_iot_2023/validation.csv` | 347,921,865 | `b13775e87ef64c8dd66c910d4585323f99b1209f2508c75a0a0a8d584caf20d8` | Raw CIC-IoT-2023 Val CSV ($1,176,851 \times 47$) |
| `data/processed/edge_iiot_processed.npz` | 27,248,312 | `f37506765c9df50aea3bd207e9f20a425b8cd153db3343906deae6be0837e977` | Locked Stratified 70/15/15 Splitted NPZ Archive |
| `models/registry/catboost_model_v2.pkl` | 5,703,035 | `8f85468477d4ec5b9b61198a4431aa35e0877698566570668b257e60734d59c8` | Trained CatBoost Baseline Checkpoint ($K=25$) |
| `models/registry/xgboost_model_v2.pkl` | 3,214,711 | `d4e0509014fc1e77478e3272f1359fb3a0373ce720aef768ba9f4c89e9fa5ea7` | Trained XGBoost Baseline Checkpoint ($K=25$) |
| `models/registry/lightgbm_model_v2.pkl` | 1,633,622 | `b173ed3bce672c8f717d05ac3994649da7690c5d237e10ce9a16762e7f2319a3` | Trained LightGBM Baseline Checkpoint ($K=25$) |
| `models/registry/hybrid_ensemble_v2.pkl` | 10,556,806 | `0fc79fadef9b686daaed2b67fe0fc072eae4081eb5fb1d13e0971162d1b57c6d` | Trained Simplex Ensemble Checkpoint ($K=25$) |
| `experiments/verified_experiment_matrix_E1_to_E12.json` | 142,804 | `a90d8ef6a086057a62e3d93d3950ef9cb962ba6e1b6f9f654b455b85e05a8f90` | Single-Source-of-Truth Machine-Readable Experiment Matrix |
| `artifacts/metrics/locked_test_metrics.json` | 45,529 | `4b8a6fb7c32e68798e40f9076bcce903bb1f5442e20ff9a68a27b8bfbc299042` | Locked Test Predictions and Classification Metrics |

---

## 2. Reproduction Runbook

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/RupaliChavanke/Hybrid-Edge-IIOT.git
cd Hybrid-Edge-IIOT

# Create and activate clean Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install exact pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Data Preparation and Partitioning
```bash
# Executes leakage-safe stratified 70/15/15 split (seed 42)
python scripts/prepare_data.py --config configs/data.yaml
```

### Step 3: Base Model and Simplex Ensemble Training ($K=33$)
```bash
# Train CatBoost, XGBoost, and LightGBM on training set with early stopping on validation set
python scripts/train_models.py --features protocol_k33

# Fit validation-constrained SLSQP simplex weights and probability calibrators
python scripts/train_ensemble.py --strategy slsqp --calibrate temperature,isotonic
```

### Step 4: Locked Test Evaluation and Equivalence-Class Collision Proof
```bash
# Evaluate frozen models on locked test set (N=23,670) and compute collision bounds
python scripts/evaluate_models.py --test-set data/processed/edge_iiot_processed.npz
```

### Step 5: Cross-Dataset Generalization Suite
```bash
# Evaluates 10-class harmonized transfer across Edge-IIoTset and CIC-IoT-2023
python scripts/cross_dataset_evaluation.py --config configs/evaluation.yaml
```

### Step 6: Statistical Hypothesis Testing and Bootstrap Resampling
```bash
# Run Edwards-corrected McNemar test and 500-replicate paired bootstrap intervals
python scripts/run_ablations_and_statistics.py --replicates 500 --seed 42
```

### Step 7: Streaming Inference and Throughput Profiling
```bash
# Run isolated model timing and micro-batch end-to-end streaming latency benchmarks
python scripts/benchmark_runtime.py --batch-sizes 1,10,50,100,500
```

### Step 8: Compile IEEE Transactions LaTeX Manuscript
```bash
cd /Users/rupesh/Documents/Paper/IEEE-Transactions-LaTeX2e-templates-and-instructions
tectonic IEEE_IoT_IDS_Manuscript.tex
```

---

## 3. Determinism and Hardware Invariance
- Random seeds are fixed at `42` across all training (`CatBoost(random_seed=42)`, `XGBClassifier(random_state=42)`, `LGBMClassifier(random_state=42)`), splitting (`train_test_split(random_state=42)`), and bootstrap resampling routines.
- Classification metrics (Accuracy, Macro-$F_1$, precision, recall, confusion matrix, collision counts) are mathematically deterministic and bit-for-bit identical across any POSIX platform with identical dependencies.
- Streaming timing measurements depend on the host processor and thread concurrency; reported numbers reflect Apple Silicon M2.
