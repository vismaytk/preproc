"""Unit tests for TabularProcessor."""

import numpy as np
import pandas as pd
import pytest
from io import BytesIO

from api.processors.tabular import TabularProcessor


@pytest.fixture
def processor():
    return TabularProcessor()


@pytest.fixture
def loaded_processor(processor, sample_csv_data):
    processor.load_data(sample_csv_data)
    return processor


class TestTabularProcessor:
    """Tests for the TabularProcessor class."""

    def test_load_data(self, loaded_processor):
        assert loaded_processor.original_df is not None
        assert loaded_processor.processed_df is not None
        assert loaded_processor.original_df.shape == (5, 3)

    def test_handle_missing_drop(self, loaded_processor):
        loaded_processor.handle_missing_values("drop")
        assert loaded_processor.processed_df.isnull().sum().sum() == 0
        assert loaded_processor.processed_df.shape[0] == 3

    def test_handle_missing_mean(self, loaded_processor):
        loaded_processor.handle_missing_values("mean")
        assert loaded_processor.processed_df["A"].isnull().sum() == 0
        # Mean of [1, 2, 4, 1] = 2.0; after imputation mean should be ~2.0
        expected_mean = np.mean([1.0, 2.0, 4.0, 1.0])
        filled_val = loaded_processor.processed_df.loc[2, "A"]
        assert filled_val == pytest.approx(expected_mean, abs=0.1)

    def test_handle_missing_median(self, loaded_processor):
        """Bug Fix #1: Verify median imputation actually uses median, not mean."""
        loaded_processor.handle_missing_values("median")
        assert loaded_processor.processed_df["A"].isnull().sum() == 0
        # Median of [1, 2, 4, 1] = 1.5
        expected_median = np.median([1.0, 2.0, 4.0, 1.0])
        filled_val = loaded_processor.processed_df.loc[2, "A"]
        assert filled_val == pytest.approx(expected_median, abs=0.1)

    def test_handle_missing_invalid(self, loaded_processor):
        with pytest.raises(ValueError, match="Unknown strategy"):
            loaded_processor.handle_missing_values("invalid")

    def test_remove_duplicates(self, loaded_processor):
        loaded_processor.remove_duplicates()
        assert loaded_processor.processed_df.shape[0] == 4

    def test_scale_features_standard(self, loaded_processor):
        loaded_processor.handle_missing_values("mean")
        loaded_processor.scale_features("standard")
        assert loaded_processor.processed_df["A"].mean() == pytest.approx(0, abs=1e-10)
        assert loaded_processor.processed_df["A"].std(ddof=1) == pytest.approx(1, abs=1e-10)

    def test_scale_features_minmax(self, loaded_processor):
        loaded_processor.handle_missing_values("mean")
        loaded_processor.scale_features("minmax")
        assert loaded_processor.processed_df["B"].min() == pytest.approx(0, abs=1e-10)
        assert loaded_processor.processed_df["B"].max() == pytest.approx(1, abs=1e-10)

    def test_drop_columns(self, loaded_processor):
        loaded_processor.drop_columns(["A"])
        assert "A" not in loaded_processor.processed_df.columns
        assert loaded_processor.processed_df.shape[1] == 2

    def test_detect_outliers_iqr(self, loaded_processor):
        outliers = loaded_processor.detect_outliers("iqr")
        assert isinstance(outliers, dict)
        assert "A" in outliers
        assert "B" in outliers

    def test_detect_outliers_zscore(self, loaded_processor):
        outliers = loaded_processor.detect_outliers("zscore")
        assert isinstance(outliers, dict)

    def test_encode_categoricals_label(self, loaded_processor):
        loaded_processor.encode_categoricals("label")
        # After label encoding, column C should be numeric
        assert loaded_processor.processed_df["C"].dtype in [np.int32, np.int64, np.float64]

    def test_encode_categoricals_onehot(self, loaded_processor):
        original_cols = len(loaded_processor.processed_df.columns)
        loaded_processor.encode_categoricals("onehot")
        # One-hot encoding should increase column count
        assert len(loaded_processor.processed_df.columns) >= original_cols

    def test_get_statistics(self, loaded_processor):
        stats = loaded_processor.get_statistics()
        assert stats["original_shape"] == [5, 3]
        assert stats["processed_shape"] == [5, 3]
        assert "missing_values" in stats
        assert "numeric_columns" in stats
        assert "categorical_columns" in stats
        assert "descriptive_stats" in stats
        assert "column_types" in stats
        assert "sample_preview" in stats
        assert "outliers" in stats
        assert len(stats["sample_preview"]) <= 5

    def test_get_processed_data(self, loaded_processor):
        data = loaded_processor.get_processed_data()
        assert isinstance(data, bytes)
        df = pd.read_csv(BytesIO(data))
        assert df.shape == (5, 3)

    def test_full_pipeline(self, loaded_processor):
        loaded_processor.handle_missing_values("median")
        loaded_processor.remove_duplicates()
        loaded_processor.scale_features("standard")
        stats = loaded_processor.get_statistics()
        assert stats["rows_removed"] >= 0
        assert len(stats["preprocessing_steps"]) == 3
