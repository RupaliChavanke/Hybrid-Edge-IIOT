"""
Data Cleaning and Sanitization Module.
Extracts netflow length signatures, drops leakage columns (including timestamps and IPs),
and fits median imputers strictly on training data.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


METADATA_DROP_COLUMNS = [
    "frame.time",
    "icmp.transmit_timestamp",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.referer",
    "http.request.uri.query",
    "tcp.payload",
    "tcp.options",
    "mqtt.msg",
    "dns.qry.name",
    "dns.qry.name.len"
]


class PipelineCleaner:
    """Sanitizes raw network flow records and extracts structural length metrics."""

    def __init__(self, drop_metadata: bool = True):
        self.drop_metadata = drop_metadata
        self.fitted_numeric_columns: List[str] = []
        self.impute_values: Dict[str, float] = {}
        self.constant_columns: List[str] = []
        self.is_fitted: bool = False

    def _extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives safe structural flow length statistics prior to dropping raw string metadata."""
        out = df.copy()

        for src_col, dst_col in [
            ("http.request.full_uri", "len_uri"),
            ("http.request.uri.query", "len_query"),
            ("tcp.payload", "len_payload"),
            ("http.referer", "len_referer"),
            ("http.file_data", "len_file_data"),
        ]:
            if src_col in out.columns:
                out[dst_col] = out[src_col].fillna("").astype(str).str.len().astype(np.float32)
            else:
                out[dst_col] = np.float32(0.0)

        return out

    def fit(self, df: pd.DataFrame, target_cols: Optional[List[str]] = None) -> "PipelineCleaner":
        """Fits imputer medians and identifies constant columns exclusively on training data."""
        targets = target_cols or ["Attack_type", "Attack_label", "label"]
        processed = self._extract_flow_features(df)

        cols_to_drop = [c for c in METADATA_DROP_COLUMNS if c in processed.columns] if self.drop_metadata else []
        cols_to_drop += [c for c in targets if c in processed.columns]
        feature_df = processed.drop(columns=cols_to_drop, errors="ignore")

        converted_cols = {}
        for col in feature_df.columns:
            converted = pd.to_numeric(feature_df[col], errors="coerce")
            if converted.notnull().sum() > 0.5 * len(converted):
                converted_cols[col] = converted

        numeric_df = pd.DataFrame(converted_cols, index=feature_df.index)
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

        self.constant_columns = [c for c in numeric_df.columns if numeric_df[c].nunique(dropna=False) <= 1]
        active_cols = [c for c in numeric_df.columns if c not in self.constant_columns]

        self.impute_values = numeric_df[active_cols].median().fillna(0.0).to_dict()
        self.fitted_numeric_columns = active_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame using training-fitted medians and column selections."""
        if not self.is_fitted:
            raise RuntimeError("PipelineCleaner must be fitted before calling transform!")

        processed = self._extract_flow_features(df)
        out = pd.DataFrame(index=processed.index)

        for col in self.fitted_numeric_columns:
            if col in processed.columns:
                series = pd.to_numeric(processed[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
                out[col] = series.fillna(self.impute_values.get(col, 0.0)).astype(np.float32)
            else:
                out[col] = np.float32(self.impute_values.get(col, 0.0))

        return out

    def fit_transform(self, df: pd.DataFrame, target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fits cleaner and transforms training DataFrame."""
        return self.fit(df, target_cols).transform(df)
