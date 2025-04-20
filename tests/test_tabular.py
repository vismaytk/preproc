import pytest
import pandas as pd
import numpy as np
from io import BytesIO
from backend.processors.tabular import TabularProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 1],
        'B': [10, 20, 30, 40, 10],
        'C': ['x', 'y', 'z', None, 'x']
    })
    return df.to_csv(index=False).encode()

@pytest.fixture
def processor():
    """Create a TabularProcessor instance."""
    return TabularProcessor()

def test_load_data(processor, sample_data):
    """Test loading data into the processor."""
    processor.load_data(sample_data)
    assert processor.original_df is not None
    assert processor.processed_df is not None
    assert processor.original_df.shape == (5, 3)

def test_handle_missing_values_drop(processor, sample_data):
    """Test handling missing values with drop strategy."""
    processor.load_data(sample_data)
    processor.handle_missing_values(strategy="drop")
    assert processor.processed_df.isnull().sum().sum() == 0
    assert processor.processed_df.shape[0] == 3

def test_handle_missing_values_mean(processor, sample_data):
    """Test handling missing values with mean strategy."""
    processor.load_data(sample_data)
    processor.handle_missing_values(strategy="mean")
    assert processor.processed_df['A'].isnull().sum() == 0
    assert processor.processed_df['A'].mean() == pytest.approx(2.0)

def test_remove_duplicates(processor, sample_data):
    """Test removing duplicate rows."""
    processor.load_data(sample_data)
    processor.remove_duplicates()
    assert processor.processed_df.shape[0] == 4

def test_scale_features(processor, sample_data):
    """Test scaling numeric features."""
    processor.load_data(sample_data)
    processor.handle_missing_values(strategy="mean")
    processor.scale_features(method="standard")
    assert processor.processed_df['A'].mean() == pytest.approx(0, abs=1e-10)
    assert processor.processed_df['A'].std() == pytest.approx(1, abs=1e-10)

def test_drop_columns(processor, sample_data):
    """Test dropping columns."""
    processor.load_data(sample_data)
    processor.drop_columns(['A'])
    assert 'A' not in processor.processed_df.columns
    assert processor.processed_df.shape[1] == 2

def test_get_statistics(processor, sample_data):
    """Test getting dataset statistics."""
    processor.load_data(sample_data)
    stats = processor.get_statistics()
    assert stats['original_shape'] == (5, 3)
    assert stats['processed_shape'] == (5, 3)
    assert 'missing_values' in stats
    assert 'numeric_columns' in stats
    assert 'categorical_columns' in stats
    assert 'preprocessing_steps' in stats

def test_get_processed_data(processor, sample_data):
    """Test getting processed data as CSV bytes."""
    processor.load_data(sample_data)
    processed_data = processor.get_processed_data()
    assert isinstance(processed_data, bytes)
    # Verify we can read it back as a DataFrame
    df = pd.read_csv(BytesIO(processed_data))
    assert df.shape == (5, 3)