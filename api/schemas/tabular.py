"""Pydantic v2 schemas for tabular processing endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class TabularProcessResponse(BaseModel):
    """Response model for tabular data processing."""

    original_shape: list[int] = Field(..., description="Original [rows, cols] shape")
    processed_shape: list[int] = Field(..., description="Processed [rows, cols] shape")
    rows_removed: int = Field(..., description="Number of rows removed during processing")
    missing_values: dict[str, int] = Field(
        ..., description="Missing value count per column after processing"
    )
    numeric_columns: list[str] = Field(..., description="Names of numeric columns")
    categorical_columns: list[str] = Field(..., description="Names of categorical columns")
    preprocessing_steps: list[dict[str, Any]] = Field(
        ..., description="List of preprocessing steps applied"
    )
    descriptive_stats: dict[str, dict[str, float | None]] = Field(
        default_factory=dict, description="Per-column descriptive statistics"
    )
    column_types: dict[str, str] = Field(
        default_factory=dict, description="Column name to dtype mapping"
    )
    sample_preview: list[dict[str, Any]] = Field(
        default_factory=list, description="First 5 rows as list of dicts"
    )
    outliers: dict[str, int] = Field(
        default_factory=dict, description="Outlier count per numeric column (IQR method)"
    )
    processed_data: str = Field(..., description="Processed CSV data as string")
