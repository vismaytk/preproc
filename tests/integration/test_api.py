"""Integration tests for the FastAPI application."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from io import BytesIO

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def csv_file():
    """Create a CSV file for testing."""
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0],
        "B": [10.0, 20.0, 30.0, 40.0],
        "C": ["x", "y", "z", "w"],
    })
    return ("test.csv", df.to_csv(index=False).encode(), "text/csv")


@pytest.fixture
def txt_file():
    """Create a text file for testing."""
    text = "Hello World! This is a test. Visit https://example.com for more info."
    return ("test.txt", text.encode(), "text/plain")


@pytest.fixture
def img_file():
    """Create an image file for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return ("test.png", buf.getvalue(), "image/png")


class TestHealthEndpoints:
    """Test health and root endpoints."""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "DataPrep Pro" in data["message"]

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestTabularEndpoint:
    """Test tabular processing endpoint."""

    def test_process_csv_basic(self, client, csv_file):
        response = client.post(
            "/tabular/process",
            files={"file": csv_file},
            data={"handle_missing": "drop", "remove_duplicates": "true"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_shape" in data
        assert "processed_shape" in data
        assert "processed_data" in data
        assert "rows_removed" in data

    def test_process_csv_mean(self, client, csv_file):
        response = client.post(
            "/tabular/process",
            files={"file": csv_file},
            data={"handle_missing": "mean", "remove_duplicates": "false"},
        )
        assert response.status_code == 200

    def test_process_csv_median(self, client, csv_file):
        """Bug Fix #1: Verify median endpoint works."""
        response = client.post(
            "/tabular/process",
            files={"file": csv_file},
            data={"handle_missing": "median", "remove_duplicates": "false"},
        )
        assert response.status_code == 200

    def test_process_csv_with_scaling(self, client, csv_file):
        response = client.post(
            "/tabular/process",
            files={"file": csv_file},
            data={
                "handle_missing": "mean",
                "remove_duplicates": "true",
                "scaling_method": "standard",
            },
        )
        assert response.status_code == 200

    def test_reject_non_csv(self, client):
        response = client.post(
            "/tabular/process",
            files={"file": ("test.txt", b"hello", "text/plain")},
            data={"handle_missing": "drop"},
        )
        assert response.status_code == 400

    def test_reject_empty_file(self, client):
        response = client.post(
            "/tabular/process",
            files={"file": ("test.csv", b"", "text/csv")},
            data={"handle_missing": "drop"},
        )
        assert response.status_code == 400


class TestTextEndpoint:
    """Test text processing endpoint."""

    def test_process_text(self, client, txt_file):
        response = client.post(
            "/text/process",
            files={"file": txt_file},
            data={
                "remove_stopwords": "true",
                "lemmatize": "true",
                "lowercase": "true",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "processed_text" in data
        assert "detected_language" in data

    def test_word_frequencies(self, client, txt_file):
        """Bug Fix #4: Test new word frequencies endpoint."""
        # First process text to get a session
        response = client.post(
            "/text/process",
            files={"file": txt_file},
            data={"remove_stopwords": "true", "lemmatize": "true", "lowercase": "true"},
        )
        session_id = response.json()["session_id"]

        # Get frequencies
        response = client.get(f"/text/frequencies/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "frequencies" in data
        assert "total_words" in data

    def test_pos_tags(self, client, txt_file):
        """Bug Fix #4: Test new POS tags endpoint."""
        response = client.post(
            "/text/process",
            files={"file": txt_file},
            data={"remove_stopwords": "false", "lemmatize": "false", "lowercase": "false"},
        )
        session_id = response.json()["session_id"]

        response = client.get(f"/text/pos-tags/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert "pos_tags" in data

    def test_invalid_session(self, client):
        response = client.get("/text/frequencies/invalid-id")
        assert response.status_code == 404

    def test_reject_non_txt(self, client):
        response = client.post(
            "/text/process",
            files={"file": ("test.csv", b"a,b\n1,2", "text/csv")},
        )
        assert response.status_code == 400


class TestImageEndpoint:
    """Test image processing endpoint."""

    def test_process_image(self, client, img_file):
        response = client.post(
            "/image/process",
            files={"file": img_file},
            data={"normalize": "true", "grayscale": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "processed_image" in data
        assert "original_shape" in data
        assert "color_mode" in data
        assert "channel_means" in data

    def test_process_image_grayscale(self, client, img_file):
        response = client.post(
            "/image/process",
            files={"file": img_file},
            data={"normalize": "true", "grayscale": "true"},
        )
        assert response.status_code == 200
        assert response.json()["color_mode"] == "Grayscale"

    def test_process_image_resize(self, client, img_file):
        response = client.post(
            "/image/process",
            files={"file": img_file},
            data={
                "normalize": "false",
                "grayscale": "false",
                "resize_dimensions": "100x100",
            },
        )
        assert response.status_code == 200

    def test_reject_non_image(self, client):
        response = client.post(
            "/image/process",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
