"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from io import BytesIO


@pytest.fixture
def sample_csv_data() -> bytes:
    """Create sample CSV data with missing values and duplicates."""
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0, 1.0],
        "B": [10.0, 20.0, 30.0, 40.0, 10.0],
        "C": ["x", "y", "z", None, "x"],
    })
    return df.to_csv(index=False).encode()


@pytest.fixture
def sample_text_data() -> bytes:
    """Create sample text data with URLs and emails."""
    text = (
        "Hello World! This is a test text with numbers 123 and punctuation... "
        "It has multiple   spaces and line "
        "breaks. Visit https://example.com or email test@example.com. "
        "StopWords like 'the' and 'is' should be removed."
    )
    return text.encode()


@pytest.fixture
def sample_image_data() -> bytes:
    """Create a small sample PNG image."""
    img = Image.fromarray(
        np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
