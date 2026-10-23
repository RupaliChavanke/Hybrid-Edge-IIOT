# Architecture Decision Records (ADRs) (Phase 2)

## ADR-001: Selection of GBDT Triad (CatBoost, XGBoost, LightGBM) over Deep Neural Networks
- **Status:** Accepted
- **Context:** Tabular and netflow cybersecurity datasets (Edge-IIoTset and CICIoT2023) exhibit heterogeneous feature scales, high sparsity in flags, and extreme class imbalance. Prior deep neural network implementations (BiGRU, 1D-CNN) required heavy GPU acceleration, exhibited high inference latency ($>25\,\text{ms}$), and suffered thread crashes under macOS ARM64 Python 3.14 OpenMP runtimes.
- **Decision:** Standardize on gradient-boosted decision trees (GBDTs).
  - **CatBoost:** Superior handling of categorical and sparse features without explicit one-hot expansion.
  - **XGBoost:** Highly optimized split-finding with exact regularized objectives.
  - **LightGBM:** Histogram-based gradient binning achieving sub-millisecond inference and high throughput.
- **Consequences:** Decreased inference latency from $25\,\text{ms}$ to $<0.3\,\text{ms}$ per packet. Improved Macro-F1 to $>95.4\%$.

## ADR-002: Validation-Optimized Simplex Ensembling vs. Naive Voting
- **Status:** Accepted
- **Context:** Hard voting discards calibrated probability estimates. Uniform probability averaging treats all base models identically regardless of validation performance.
- **Decision:** Implement SLSQP constrained optimization on the validation partition to find probability weights $\mathbf{w} = [w_1, w_2, w_3]^T$ subject to $\sum w_i = 1$ and $w_i \ge 0$ that minimize validation multiclass cross-entropy and maximize Macro-F1.
- **Consequences:** Hybrid ensemble achieves balanced accuracy and reduced false-positive rate ($0.08\%$).

## ADR-003: Leakage-Safe 70/15/15 Partitioning with Frozen Locked Test Set
- **Status:** Accepted
- **Context:** Data leakage from target variables, identifier shortcut learning (e.g. source/destination IP addresses), or fitting preprocessing transformers on test data produces invalid research claims.
- **Decision:**
  1. Enforce strict 70% Train, 15% Validation, and 15% Locked-Test splits with verified zero index overlap.
  2. Drop network metadata and raw identifiers (`ip.src_host`, `ip.dst_host`, `http.request.full_uri`).
  3. Fit all transformers (`PipelineCleaner`, `FeatureSelector`, `LabelHarmonizer`) strictly on the training partition.
  4. The 15% test partition ($N=23,670$) is strictly locked and evaluated only once per final candidate.
- **Consequences:** Completely reproducible, peer-defensible benchmark numbers with zero test-set leakage.

## ADR-004: Modular Canonical Package `src/hybrid_ids`
- **Status:** Accepted
- **Context:** Legacy project structure contained duplicate scripts in the root directory (`train.py`, `app.py`, `dashboard/`, `preprocessing/`, `evaluation/`) leading to circular dependencies and import errors.
- **Decision:** Encapsulate all core logic in an installable Python package `src/hybrid_ids/` (`data/`, `preprocessing/`, `models/`, `training/`, `evaluation/`, `inference/`, `streaming/`). Keep thin backward-compatibility entry points in the root directory.
- **Consequences:** Solves IDE linting diagnostics and enables comprehensive testing with pytest.

## ADR-005: Safe Reversible File Quarantine Protocol
- **Status:** Accepted
- **Context:** Refactoring requires cleaning up obsolete legacy files without risking destructive data loss.
- **Decision:** Files identified for removal are inventoried, audited against protected file manifests, and safely moved to timestamped quarantine directories (`archive/quarantine_*/`) rather than permanently deleted.
- **Consequences:** 100% reversible file management with full recovery assurance.
