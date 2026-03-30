"""Tabular data processing routes."""

import json
import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.core.config import get_settings
from api.core.exceptions import EmptyFileError, FileTooLargeError, InvalidFileTypeError
from api.core.logging import get_logger
from api.processors.tabular import TabularProcessor
from api.schemas.tabular import TabularProcessResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/tabular", tags=["Tabular"])


@router.post("/process", response_model=TabularProcessResponse)
async def process_tabular(
    request: Request,
    file: UploadFile = File(...),
    handle_missing: str = Form("drop"),
    remove_duplicates: bool = Form(True),
    columns_to_drop: str | None = Form(None),
    scaling_method: str | None = Form(None),
) -> TabularProcessResponse:
    """Preprocess tabular (CSV) data.

    Parameters:
    - **file**: CSV file to process
    - **handle_missing**: Strategy for missing values — 'drop', 'mean', or 'median'
    - **remove_duplicates**: Whether to remove duplicate rows
    - **columns_to_drop**: JSON string of column names to drop
    - **scaling_method**: Feature scaling — 'standard', 'minmax', or 'robust'
    """
    start_time = time.time()
    settings = get_settings()

    # --- Validation ---
    # File type validation
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise InvalidFileTypeError([".csv"])

    # File size validation — check Content-Length before reading
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_csv_size:
        raise FileTooLargeError(settings.max_csv_size / (1024 * 1024))

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise EmptyFileError()

    # Double-check size after reading
    if len(content) > settings.max_csv_size:
        raise FileTooLargeError(settings.max_csv_size / (1024 * 1024))

    # Validate strategy
    if handle_missing not in ("drop", "mean", "median"):
        raise HTTPException(status_code=400, detail="Invalid missing value strategy. Use 'drop', 'mean', or 'median'.")

    if scaling_method and scaling_method not in ("standard", "minmax", "robust"):
        raise HTTPException(status_code=400, detail="Invalid scaling method. Use 'standard', 'minmax', or 'robust'.")

    # --- Processing ---
    try:
        processor = TabularProcessor()
        processor.load_data(content)

        # Validate columns_to_drop
        if columns_to_drop:
            try:
                columns = json.loads(columns_to_drop)
                available = list(processor.processed_df.columns)  # type: ignore
                invalid = [c for c in columns if c not in available]
                if invalid:
                    raise HTTPException(status_code=400, detail=f"Invalid columns: {invalid}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid columns_to_drop JSON format")

        # Apply preprocessing
        processor.handle_missing_values(handle_missing)

        if remove_duplicates:
            processor.remove_duplicates()

        if columns_to_drop:
            columns = json.loads(columns_to_drop)
            processor.drop_columns(columns)

        if scaling_method:
            processor.scale_features(scaling_method)

        # Gather results
        stats = processor.get_statistics()
        processed_csv = processor.get_processed_data().decode()

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "tabular_processing_complete",
            filename=file.filename,
            file_size=len(content),
            handle_missing=handle_missing,
            scaling_method=scaling_method,
            processing_time_ms=duration_ms,
            final_shape=stats["processed_shape"],
        )

        return TabularProcessResponse(
            processed_data=processed_csv,
            **stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("tabular_processing_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
