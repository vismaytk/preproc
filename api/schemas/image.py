"""Pydantic v2 schemas for image processing endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class ImageProcessResponse(BaseModel):
    """Response model for image data processing."""

    original_shape: list[int] = Field(..., description="Original image dimensions [H, W, C]")
    processed_shape: list[int] = Field(..., description="Processed image dimensions")
    format: str = Field(..., description="Output image format (PNG, JPEG, WEBP)")
    color_mode: str = Field(..., description="Color mode (RGB or Grayscale)")
    original_size_bytes: int = Field(..., description="Original file size in bytes")
    processed_size_bytes: int = Field(..., description="Processed file size in bytes")
    channel_means: dict[str, float] = Field(
        ..., description="Per-channel mean pixel values"
    )
    preprocessing_steps: list[dict[str, Any]] = Field(
        default_factory=list, description="List of preprocessing steps applied"
    )
    processed_image: str = Field(..., description="Base64-encoded processed image")
