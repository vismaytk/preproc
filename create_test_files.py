import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

# Write test_tabular.py
tabular_content = '''import pytest
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
    assert processor.processed_df['A'].mean() == pytest.approx(2.5)

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
    assert df.shape == (5, 3)'''

# Write test_text.py
text_content = '''import pytest
from backend.processors.text import TextProcessor

@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    text = """Hello World! This is a test text with numbers 123 and punctuation...
    It has multiple   spaces and line
    breaks. It also has StopWords like 'the' and 'is'."""
    return text.encode()

@pytest.fixture
def processor():
    """Create a TextProcessor instance."""
    return TextProcessor()

def test_load_data(processor, sample_text):
    """Test loading text data."""
    processor.load_data(sample_text)
    assert processor.original_text is not None
    assert processor.processed_text is not None
    assert len(processor.original_text) > 0

def test_to_lowercase(processor, sample_text):
    """Test converting text to lowercase."""
    processor.load_data(sample_text)
    processor.to_lowercase()
    assert "Hello" not in processor.processed_text
    assert "hello" in processor.processed_text

def test_remove_punctuation(processor, sample_text):
    """Test removing punctuation."""
    processor.load_data(sample_text)
    processor.remove_punctuation()
    assert "!" not in processor.processed_text
    assert "..." not in processor.processed_text

def test_remove_numbers(processor, sample_text):
    """Test removing numbers."""
    processor.load_data(sample_text)
    processor.remove_numbers()
    assert "123" not in processor.processed_text

def test_remove_whitespace(processor, sample_text):
    """Test removing extra whitespace."""
    processor.load_data(sample_text)
    processor.remove_whitespace()
    assert "  " not in processor.processed_text
    assert "\\n" not in processor.processed_text

def test_remove_stopwords(processor, sample_text):
    """Test removing stop words."""
    processor.load_data(sample_text)
    processor.remove_stopwords()
    assert " the " not in processor.processed_text.lower()
    assert " is " not in processor.processed_text.lower()

def test_lemmatize_text(processor, sample_text):
    """Test lemmatization."""
    processor.load_data(sample_text)
    original_words = len(processor.processed_text.split())
    processor.lemmatize_text()
    lemmatized_words = len(processor.processed_text.split())
    assert lemmatized_words <= original_words

def test_get_statistics(processor, sample_text):
    """Test getting text statistics."""
    processor.load_data(sample_text)
    stats = processor.get_statistics()
    assert "original_length" in stats
    assert "processed_length" in stats
    assert "original_word_count" in stats
    assert "processed_word_count" in stats
    assert "original_sentence_count" in stats
    assert "processed_sentence_count" in stats
    assert "preprocessing_steps" in stats

def test_get_processed_data(processor, sample_text):
    """Test getting processed data as bytes."""
    processor.load_data(sample_text)
    processed_data = processor.get_processed_data()
    assert isinstance(processed_data, bytes)
    assert len(processed_data) > 0

def test_get_word_frequencies(processor, sample_text):
    """Test getting word frequencies."""
    processor.load_data(sample_text)
    freq = processor.get_word_frequencies()
    assert isinstance(freq, dict)
    assert len(freq) > 0
    
    # Test with top_n parameter
    top_3 = processor.get_word_frequencies(top_n=3)
    assert len(top_3) <= 3

def test_get_pos_tags(processor, sample_text):
    """Test getting part-of-speech tags."""
    processor.load_data(sample_text)
    tags = processor.get_pos_tags()
    assert isinstance(tags, list)
    assert len(tags) > 0
    assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in tags)

def test_full_pipeline(processor, sample_text):
    """Test running full preprocessing pipeline."""
    processor.load_data(sample_text)
    processor.to_lowercase()
    processor.remove_punctuation()
    processor.remove_numbers()
    processor.remove_whitespace()
    processor.remove_stopwords()
    processor.lemmatize_text()
    
    processed = processor.processed_text
    assert processed.islower()
    assert "..." not in processed
    assert "123" not in processed
    assert "  " not in processed
    assert " the " not in processed
    assert len(processor.preprocessing_steps) == 6'''

# Write the files
write_file('tests/test_tabular.py', tabular_content)
write_file('tests/test_text.py', text_content)
write_file('tests/__init__.py', '"""Test package initialization file."""') 