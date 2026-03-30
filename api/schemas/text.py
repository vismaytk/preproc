"""Pydantic v2 schemas for text processing endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class TextProcessResponse(BaseModel):
    """Response model for text data processing."""

    session_id: str = Field(..., description="Session ID for subsequent requests")
    original_length: int = Field(..., description="Original text length in characters")
    processed_length: int = Field(..., description="Processed text length in characters")
    original_word_count: int = Field(..., description="Original word count")
    processed_word_count: int = Field(..., description="Processed word count")
    preprocessing_steps: list[dict[str, Any]] = Field(
        ..., description="List of preprocessing steps applied"
    )
    processed_text: str = Field(..., description="The processed text content")
    detected_language: str | None = Field(None, description="Detected language code (ISO 639-1)")
    is_english: bool | None = Field(None, description="Whether the text is detected as English")
    top_10_words: dict[str, int] = Field(
        default_factory=dict, description="Top 10 most frequent words"
    )
    avg_word_length: float = Field(0.0, description="Average word length in characters")
    vocabulary_richness: float = Field(
        0.0, description="Vocabulary richness (unique words / total words)"
    )


class WordFrequencyResponse(BaseModel):
    """Response model for word frequency endpoint."""

    session_id: str = Field(..., description="Session ID")
    frequencies: dict[str, int] = Field(..., description="Word frequency mapping")
    total_words: int = Field(..., description="Total word count")


class PosTagResponse(BaseModel):
    """Response model for POS tagging endpoint."""

    session_id: str = Field(..., description="Session ID")
    pos_tags: list[list[str]] = Field(
        ..., description="List of [word, tag] pairs"
    )
    total_tokens: int = Field(..., description="Total token count")
