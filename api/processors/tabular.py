"""Tabular data processor with preprocessing operations."""

from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
)

from api.core.logging import get_logger
from api.processors.base import BaseProcessor

logger = get_logger(__name__)


class TabularProcessor(BaseProcessor):
    """Process tabular (CSV) data with various preprocessing operations."""

    def __init__(self) -> None:
        self.original_df: pd.DataFrame | None = None
        self.processed_df: pd.DataFrame | None = None
        self.preprocessing_steps: list = []

    def load_data(self, data: bytes) -> None:
        """Load CSV data from bytes into pandas DataFrame."""
        try:
            self.original_df = pd.read_csv(BytesIO(data))
            self.processed_df = self.original_df.copy()
            logger.info(
                "tabular_data_loaded",
                shape=self.original_df.shape,
                columns=list(self.original_df.columns),
            )
        except Exception as e:
            logger.error("tabular_load_error", error=str(e))
            raise ValueError(f"Error loading data: {e}")

    def handle_missing_values(self, strategy: str = "drop") -> None:
        """Handle missing values using the specified strategy.

        Args:
            strategy: One of 'drop', 'mean', or 'median'.
        """
        assert self.processed_df is not None
        try:
            if strategy == "drop":
                self.processed_df = self.processed_df.dropna()
                self.preprocessing_steps.append(("handle_missing", "drop"))

            elif strategy == "mean":
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy="mean")
                    self.processed_df[numeric_cols] = imputer.fit_transform(
                        self.processed_df[numeric_cols]
                    )
                self.preprocessing_steps.append(("handle_missing", "mean"))

            elif strategy == "median":
                # Bug Fix #1: Proper median imputation (was falling through to mean)
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy="median")
                    self.processed_df[numeric_cols] = imputer.fit_transform(
                        self.processed_df[numeric_cols]
                    )
                self.preprocessing_steps.append(("handle_missing", "median"))

            else:
                raise ValueError(f"Unknown strategy: {strategy}. Use 'drop', 'mean', or 'median'.")

            logger.info("missing_values_handled", strategy=strategy)
        except ValueError:
            raise
        except Exception as e:
            logger.error("missing_values_error", error=str(e))
            raise ValueError(f"Error handling missing values: {e}")

    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset."""
        assert self.processed_df is not None
        original_count = len(self.processed_df)
        self.processed_df = self.processed_df.drop_duplicates()
        removed = original_count - len(self.processed_df)
        self.preprocessing_steps.append(("remove_duplicates", {"removed": removed}))
        logger.info("duplicates_removed", count=removed)

    def scale_features(
        self, method: str = "standard", columns: list[str] | None = None
    ) -> None:
        """Scale numeric features using the specified method.

        Args:
            method: One of 'standard', 'minmax', or 'robust'.
            columns: Specific columns to scale. Defaults to all numeric.
        """
        assert self.processed_df is not None
        if columns is None:
            columns = list(self.processed_df.select_dtypes(include=[np.number]).columns)

        if not columns:
            logger.warning("no_numeric_columns_for_scaling")
            return

        missing_cols = [c for c in columns if c not in self.processed_df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        scalers = {
            "standard": lambda: _StandardScaler(),
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }

        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        if method == "standard":
            # Manual standard scaling for numerical stability
            for col in columns:
                data = self.processed_df[col].values.astype(float)
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                if std > 0:
                    self.processed_df[col] = (data - mean) / std
                else:
                    self.processed_df[col] = 0.0
        else:
            scaler = scalers[method]()
            self.processed_df[columns] = scaler.fit_transform(self.processed_df[columns])

        self.preprocessing_steps.append(
            ("scale_features", {"method": method, "columns": columns})
        )
        logger.info("features_scaled", method=method, columns=columns)

    def drop_columns(self, columns: list[str]) -> None:
        """Drop specified columns from the dataset."""
        assert self.processed_df is not None
        self.processed_df = self.processed_df.drop(columns=columns, errors="ignore")
        self.preprocessing_steps.append(("drop_columns", columns))
        logger.info("columns_dropped", columns=columns)

    def detect_outliers(self, method: str = "iqr") -> dict[str, int]:
        """Detect outliers in numeric columns.

        Args:
            method: Detection method — 'iqr' or 'zscore'.

        Returns:
            Dict mapping column name to outlier count.
        """
        assert self.processed_df is not None
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        outliers: dict[str, int] = {}

        for col in numeric_cols:
            data = self.processed_df[col].dropna()
            if len(data) == 0:
                outliers[col] = 0
                continue

            if method == "iqr":
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((data < lower) | (data > upper)).sum())
            elif method == "zscore":
                mean = data.mean()
                std = data.std()
                if std > 0:
                    z_scores = np.abs((data - mean) / std)
                    outlier_count = int((z_scores > 3).sum())
                else:
                    outlier_count = 0
            else:
                raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

            outliers[col] = outlier_count

        logger.info("outliers_detected", method=method, outliers=outliers)
        return outliers

    def encode_categoricals(self, strategy: str = "label") -> None:
        """Encode categorical (string) columns.

        Args:
            strategy: 'label' for label encoding or 'onehot' for one-hot encoding.
        """
        assert self.processed_df is not None
        cat_cols = list(self.processed_df.select_dtypes(include=["object"]).columns)

        if not cat_cols:
            logger.warning("no_categorical_columns")
            return

        if strategy == "label":
            for col in cat_cols:
                le = LabelEncoder()
                # Handle NaN by filling temporarily
                mask = self.processed_df[col].isna()
                non_null = self.processed_df[col].fillna("__NAN__")
                self.processed_df[col] = le.fit_transform(non_null)
                if mask.any():
                    self.processed_df.loc[mask, col] = np.nan
        elif strategy == "onehot":
            self.processed_df = pd.get_dummies(
                self.processed_df, columns=cat_cols, drop_first=True, dtype=int
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'label' or 'onehot'.")

        self.preprocessing_steps.append(
            ("encode_categoricals", {"strategy": strategy, "columns": cat_cols})
        )
        logger.info("categoricals_encoded", strategy=strategy, columns=cat_cols)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        assert self.original_df is not None and self.processed_df is not None

        numeric_cols = list(self.processed_df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(self.processed_df.select_dtypes(include=["object"]).columns)

        # Descriptive stats per numeric column
        descriptive_stats = {}
        if numeric_cols:
            desc = self.processed_df[numeric_cols].describe()
            for col in numeric_cols:
                descriptive_stats[col] = {
                    "mean": round(float(desc.loc["mean", col]), 4) if "mean" in desc.index else None,
                    "std": round(float(desc.loc["std", col]), 4) if "std" in desc.index else None,
                    "min": round(float(desc.loc["min", col]), 4) if "min" in desc.index else None,
                    "max": round(float(desc.loc["max", col]), 4) if "max" in desc.index else None,
                    "25%": round(float(desc.loc["25%", col]), 4) if "25%" in desc.index else None,
                    "50%": round(float(desc.loc["50%", col]), 4) if "50%" in desc.index else None,
                    "75%": round(float(desc.loc["75%", col]), 4) if "75%" in desc.index else None,
                }

        # Column types
        column_types = {col: str(dtype) for col, dtype in self.processed_df.dtypes.items()}

        # Sample preview (up to 5 rows)
        sample_preview = self.processed_df.head(5).to_dict(orient="records")

        # Outlier counts
        outliers = self.detect_outliers(method="iqr")

        stats = {
            "original_shape": list(self.original_df.shape),
            "processed_shape": list(self.processed_df.shape),
            "rows_removed": self.original_df.shape[0] - self.processed_df.shape[0],
            "missing_values": {
                k: int(v) for k, v in self.processed_df.isnull().sum().to_dict().items()
            },
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "preprocessing_steps": [
                {"step": step, "params": params} for step, params in self.preprocessing_steps
            ],
            "descriptive_stats": descriptive_stats,
            "column_types": column_types,
            "sample_preview": sample_preview,
            "outliers": outliers,
        }
        return stats

    def get_processed_data(self) -> bytes:
        """Get processed data as CSV bytes."""
        assert self.processed_df is not None
        return self.processed_df.to_csv(index=False).encode()


class _StandardScaler:
    """Placeholder — standard scaling is done manually for numerical stability."""
    pass
