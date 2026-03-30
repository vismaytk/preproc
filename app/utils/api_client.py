"""Centralized API client for the Streamlit frontend.

Bug Fix #6: All API URLs come from the API_URL environment variable.
No hardcoded URLs (fixes the "bakend" typo).
"""

import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx
import pandas as pd

# Bug Fix #6: URL from env var only, never hardcoded
try:
    import streamlit as st
    API_BASE_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
except Exception:
    API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60.0  # seconds


@dataclass
class TabularResult:
    """Result from tabular processing."""
    original_shape: list[int]
    processed_shape: list[int]
    rows_removed: int
    missing_values: dict[str, int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    preprocessing_steps: list[dict[str, Any]]
    descriptive_stats: dict[str, dict[str, float | None]]
    column_types: dict[str, str]
    sample_preview: list[dict[str, Any]]
    outliers: dict[str, int]
    processed_df: pd.DataFrame | None = None


@dataclass
class TextResult:
    """Result from text processing."""
    session_id: str
    original_length: int
    processed_length: int
    original_word_count: int
    processed_word_count: int
    preprocessing_steps: list[dict[str, Any]]
    processed_text: str
    detected_language: str | None = None
    is_english: bool | None = None
    top_10_words: dict[str, int] = field(default_factory=dict)
    avg_word_length: float = 0.0
    vocabulary_richness: float = 0.0


@dataclass
class ImageResult:
    """Result from image processing."""
    original_shape: list[int]
    processed_shape: list[int]
    format: str
    color_mode: str
    original_size_bytes: int
    processed_size_bytes: int
    channel_means: dict[str, float]
    preprocessing_steps: list[dict[str, Any]]
    processed_image_b64: str


class APIError(Exception):
    """Raised when an API call fails."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


def _handle_response(response: httpx.Response) -> dict:
    """Handle API response, raising APIError on failure."""
    if response.status_code == 200:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    raise APIError(response.status_code, detail)


def process_tabular(
    file_content: bytes,
    filename: str,
    handle_missing: str = "drop",
    remove_duplicates: bool = True,
    columns_to_drop: str | None = None,
    scaling_method: str | None = None,
) -> TabularResult:
    """Process tabular data via the API."""
    with httpx.Client(timeout=TIMEOUT) as client:
        files = {"file": (filename, file_content, "text/csv")}
        data = {
            "handle_missing": handle_missing,
            "remove_duplicates": str(remove_duplicates),
        }
        if scaling_method:
            data["scaling_method"] = scaling_method
        if columns_to_drop:
            data["columns_to_drop"] = columns_to_drop

        response = client.post(f"{API_BASE_URL}/tabular/process", files=files, data=data)
        result = _handle_response(response)

        # Parse processed CSV into DataFrame
        processed_df = None
        if "processed_data" in result:
            processed_df = pd.read_csv(BytesIO(result["processed_data"].encode()))

        return TabularResult(
            original_shape=result["original_shape"],
            processed_shape=result["processed_shape"],
            rows_removed=result["rows_removed"],
            missing_values=result["missing_values"],
            numeric_columns=result["numeric_columns"],
            categorical_columns=result["categorical_columns"],
            preprocessing_steps=result["preprocessing_steps"],
            descriptive_stats=result.get("descriptive_stats", {}),
            column_types=result.get("column_types", {}),
            sample_preview=result.get("sample_preview", []),
            outliers=result.get("outliers", {}),
            processed_df=processed_df,
        )


def process_text(
    file_content: bytes,
    filename: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    lowercase: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
) -> TextResult:
    """Process text data via the API."""
    with httpx.Client(timeout=TIMEOUT) as client:
        files = {"file": (filename, file_content, "text/plain")}
        data = {
            "remove_stopwords": str(remove_stopwords),
            "lemmatize": str(lemmatize),
            "lowercase": str(lowercase),
            "remove_urls": str(remove_urls),
            "remove_emails": str(remove_emails),
        }

        response = client.post(f"{API_BASE_URL}/text/process", files=files, data=data)
        result = _handle_response(response)

        return TextResult(
            session_id=result["session_id"],
            original_length=result["original_length"],
            processed_length=result["processed_length"],
            original_word_count=result["original_word_count"],
            processed_word_count=result["processed_word_count"],
            preprocessing_steps=result["preprocessing_steps"],
            processed_text=result["processed_text"],
            detected_language=result.get("detected_language"),
            is_english=result.get("is_english"),
            top_10_words=result.get("top_10_words", {}),
            avg_word_length=result.get("avg_word_length", 0.0),
            vocabulary_richness=result.get("vocabulary_richness", 0.0),
        )


def get_word_frequencies(session_id: str, top_n: int | None = None) -> dict[str, int]:
    """Get word frequencies for a text session."""
    with httpx.Client(timeout=TIMEOUT) as client:
        params = {}
        if top_n:
            params["top_n"] = top_n
        response = client.get(f"{API_BASE_URL}/text/frequencies/{session_id}", params=params)
        result = _handle_response(response)
        return result.get("frequencies", {})


def get_pos_tags(session_id: str) -> list[list[str]]:
    """Get POS tags for a text session."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{API_BASE_URL}/text/pos-tags/{session_id}")
        result = _handle_response(response)
        return result.get("pos_tags", [])


def process_image(
    file_content: bytes,
    filename: str,
    resize_dimensions: str | None = None,
    normalize: bool = True,
    grayscale: bool = False,
    output_format: str | None = None,
    brightness: float | None = None,
    contrast: float | None = None,
) -> ImageResult:
    """Process image data via the API."""
    with httpx.Client(timeout=TIMEOUT) as client:
        files = {"file": (filename, file_content, "image/png")}
        data = {
            "normalize": str(normalize),
            "grayscale": str(grayscale),
        }
        if resize_dimensions:
            data["resize_dimensions"] = resize_dimensions
        if output_format:
            data["output_format"] = output_format
        if brightness is not None:
            data["brightness"] = str(brightness)
        if contrast is not None:
            data["contrast"] = str(contrast)

        response = client.post(f"{API_BASE_URL}/image/process", files=files, data=data)
        result = _handle_response(response)

        return ImageResult(
            original_shape=result["original_shape"],
            processed_shape=result["processed_shape"],
            format=result["format"],
            color_mode=result["color_mode"],
            original_size_bytes=result["original_size_bytes"],
            processed_size_bytes=result["processed_size_bytes"],
            channel_means=result["channel_means"],
            preprocessing_steps=result.get("preprocessing_steps", []),
            processed_image_b64=result["processed_image"],
        )
