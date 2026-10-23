"""
Cross-Dataset Generalization Benchmark Suite.
Executes Experiments A, B, C, D, E, and F with stratified splits and per-domain evaluation
to test domain transfer and multi-source adaptation across Edge-IIoTset and CICIoT2023.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from src.hybrid_ids.data.harmonization import DatasetHarmonizer
from src.hybrid_ids.data.label_mapping import LabelHarmonizer
from src.hybrid_ids.evaluation.metrics import MetricsCalculator
from src.hybrid_ids.logging_config import setup_logger

logger = setup_logger("cross_dataset")


class CrossDatasetBenchmark:
    """Manages cross-dataset experiments across Edge-IIoTset and CICIoT2023."""

    def __init__(self):
        self.harmonizer = DatasetHarmonizer()
        self.label_mapper = LabelHarmonizer()

    def run_experiments(
        self,
        edge_train: pd.DataFrame,
        edge_test: pd.DataFrame,
        cic_train: pd.DataFrame,
        cic_test: pd.DataFrame,
        train_model_fn: Any
    ) -> Dict[str, Any]:
        """Runs the standardized cross-dataset generalization experiments with per-domain disaggregated evaluation."""
        logger.info("Starting Cross-Dataset Generalization Benchmark...")
        results = {}

        X_edge_train = self.harmonizer.extract_harmonized_features_edge(edge_train)
        X_edge_test = self.harmonizer.extract_harmonized_features_edge(edge_test)
        X_cic_train = self.harmonizer.extract_harmonized_features_cic(cic_train)
        X_cic_test = self.harmonizer.extract_harmonized_features_cic(cic_test)

        y_edge_train = self.label_mapper.transform_series(edge_train["Attack_type"], "edge_iiot").values
        y_edge_test = self.label_mapper.transform_series(edge_test["Attack_type"], "edge_iiot").values
        y_cic_train = self.label_mapper.transform_series(cic_train["label"], "cic_iot_2023").values
        y_cic_test = self.label_mapper.transform_series(cic_test["label"], "cic_iot_2023").values

        common_classes = sorted(list(set(y_edge_train) & set(y_cic_train)))
        logger.info(f"Common cross-dataset evaluated classes ({len(common_classes)}): {common_classes}")

        class_to_idx = {c: i for i, c in enumerate(common_classes)}

        mask_et = np.isin(y_edge_train, common_classes)
        mask_ete = np.isin(y_edge_test, common_classes)
        mask_ct = np.isin(y_cic_train, common_classes)
        mask_cte = np.isin(y_cic_test, common_classes)

        X_et, y_et = X_edge_train[mask_et].reset_index(drop=True), np.array([class_to_idx[c] for c in y_edge_train[mask_et]])
        X_ete, y_ete = X_edge_test[mask_ete].reset_index(drop=True), np.array([class_to_idx[c] for c in y_edge_test[mask_ete]])
        X_ct, y_ct = X_cic_train[mask_ct].reset_index(drop=True), np.array([class_to_idx[c] for c in y_cic_train[mask_ct]])
        X_cte, y_cte = X_cic_test[mask_cte].reset_index(drop=True), np.array([class_to_idx[c] for c in y_cic_test[mask_cte]])

        # Experiment A: Train Edge -> Test Edge
        logger.info("Executing Experiment A (Train Edge-IIoTset -> Test Edge-IIoTset)...")
        model_edge = train_model_fn(X_et, y_et, X_ete, y_ete)
        preds_a = model_edge.predict(X_ete)
        probas_a = model_edge.predict_proba(X_ete)
        results["experiment_a_edge_to_edge"] = MetricsCalculator.compute_all_metrics(
            y_ete, preds_a, probas_a, class_names=common_classes
        )

        # Experiment B: Train CIC -> Test CIC
        logger.info("Executing Experiment B (Train CICIoT2023 -> Test CICIoT2023)...")
        model_cic = train_model_fn(X_ct, y_ct, X_cte, y_cte)
        preds_b = model_cic.predict(X_cte)
        probas_b = model_cic.predict_proba(X_cte)
        results["experiment_b_cic_to_cic"] = MetricsCalculator.compute_all_metrics(
            y_cte, preds_b, probas_b, class_names=common_classes
        )

        # Experiment C: Pooled Train -> Evaluated on Pooled Test AND Disaggregated by Target Domain
        logger.info("Executing Experiment C (Pooled Train -> Pooled Test, Edge Test, and CIC Test)...")
        X_pool_tr = pd.concat([X_et, X_ct], ignore_index=True)
        y_pool_tr = np.concatenate([y_et, y_ct])
        X_pool_te = pd.concat([X_ete, X_cte], ignore_index=True)
        y_pool_te = np.concatenate([y_ete, y_cte])

        model_pool = train_model_fn(X_pool_tr, y_pool_tr, X_pool_te, y_pool_te)
        preds_c = model_pool.predict(X_pool_te)
        probas_c = model_pool.predict_proba(X_pool_te)
        results["experiment_c_pooled_to_pooled"] = MetricsCalculator.compute_all_metrics(
            y_pool_te, preds_c, probas_c, class_names=common_classes
        )
        results["experiment_c1_pooled_on_edge"] = MetricsCalculator.compute_all_metrics(
            y_ete, model_pool.predict(X_ete), model_pool.predict_proba(X_ete), class_names=common_classes
        )
        results["experiment_c2_pooled_on_cic"] = MetricsCalculator.compute_all_metrics(
            y_cte, model_pool.predict(X_cte), model_pool.predict_proba(X_cte), class_names=common_classes
        )

        # Experiment D: Train Edge -> Test CIC (Zero-Shot Domain Transfer)
        logger.info("Executing Experiment D (Train Edge-IIoTset -> Test CICIoT2023)...")
        preds_d = model_edge.predict(X_cte)
        probas_d = model_edge.predict_proba(X_cte)
        results["experiment_d_edge_to_cic"] = MetricsCalculator.compute_all_metrics(
            y_cte, preds_d, probas_d, class_names=common_classes
        )

        # Experiment E: Train CIC -> Test Edge (Zero-Shot Domain Transfer)
        logger.info("Executing Experiment E (Train CICIoT2023 -> Test Edge-IIoTset)...")
        preds_e = model_cic.predict(X_ete)
        probas_e = model_cic.predict_proba(X_ete)
        results["experiment_e_cic_to_edge"] = MetricsCalculator.compute_all_metrics(
            y_ete, preds_e, probas_e, class_names=common_classes
        )

        # Experiment F: Domain Shift Comparison Summary
        results["experiment_f_comparison"] = {
            "edge_source_f1": results["experiment_a_edge_to_edge"]["macro_f1"],
            "cic_source_f1": results["experiment_b_cic_to_cic"]["macro_f1"],
            "pooled_f1": results["experiment_c_pooled_to_pooled"]["macro_f1"],
            "pooled_on_edge_f1": results["experiment_c1_pooled_on_edge"]["macro_f1"],
            "pooled_on_cic_f1": results["experiment_c2_pooled_on_cic"]["macro_f1"],
            "edge_to_cic_transfer_f1": results["experiment_d_edge_to_cic"]["macro_f1"],
            "cic_to_edge_transfer_f1": results["experiment_e_cic_to_edge"]["macro_f1"],
            "domain_shift_delta_edge_to_cic": round(results["experiment_a_edge_to_edge"]["macro_f1"] - results["experiment_d_edge_to_cic"]["macro_f1"], 4),
            "domain_shift_delta_cic_to_edge": round(results["experiment_b_cic_to_cic"]["macro_f1"] - results["experiment_e_cic_to_edge"]["macro_f1"], 4)
        }

        return results
