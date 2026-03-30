"""Image data processing routes."""

import time

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.core.config import get_settings
from api.core.exceptions import EmptyFileError, FileTooLargeError, InvalidFileTypeError
from api.core.logging import get_logger
from api.processors.image import ImageProcessor
from api.schemas.image import ImageProcessResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/image", tags=["Image"])

ALLOWED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@router.post("/process", response_model=ImageProcessResponse)
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    resize_dimensions: str | None = Form(None),
    normalize: bool = Form(True),
    grayscale: bool = Form(False),
    output_format: str | None = Form(None),
    brightness: float | None = Form(None),
    contrast: float | None = Form(None),
) -> ImageProcessResponse:
    """Preprocess image data with various transformations.

    Parameters:
    - **file**: Image file (PNG, JPG, JPEG, WEBP)
    - **resize_dimensions**: Target dimensions as 'WIDTHxHEIGHT' (e.g., '224x224')
    - **normalize**: Normalize pixel values to [0,1] range
    - **grayscale**: Convert to grayscale
    - **output_format**: Output format — 'PNG', 'JPEG', or 'WEBP'
    - **brightness**: Brightness adjustment factor (1.0 = no change)
    - **contrast**: Contrast adjustment factor (1.0 = no change)
    """
    start_time = time.time()
    settings = get_settings()

    # --- Validation ---
    if not file.filename or not file.filename.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
        raise InvalidFileTypeError(list(ALLOWED_IMAGE_EXTENSIONS))

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_image_size:
        raise FileTooLargeError(settings.max_image_size / (1024 * 1024))

    content = await file.read()
    if len(content) == 0:
        raise EmptyFileError()

    if len(content) > settings.max_image_size:
        raise FileTooLargeError(settings.max_image_size / (1024 * 1024))

    # --- Processing ---
    try:
        processor = ImageProcessor()
        processor.load_data(content)

        # Apply grayscale first (changes channel count)
        if grayscale:
            processor.to_grayscale()

        # Resize
        if resize_dimensions:
            try:
                width, height = map(int, resize_dimensions.split("x"))
                processor.resize(width, height)
            except ValueError as ve:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid resize dimensions. Use 'WIDTHxHEIGHT' format: {ve}",
                )

        # Brightness and contrast
        if brightness is not None:
            processor.adjust_brightness(brightness)

        if contrast is not None:
            processor.adjust_contrast(contrast)

        # Normalize
        if normalize:
            processor.normalize()

        # Output format
        if output_format:
            processor.convert_format(output_format)

        # Gather results
        stats = processor.get_statistics()
        encoded_image = processor.get_base64_image()

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "image_processing_complete",
            filename=file.filename,
            file_size=len(content),
            processing_time_ms=duration_ms,
            final_shape=stats["processed_dims"],
        )

        return ImageProcessResponse(
            original_shape=stats["original_dims"],
            processed_shape=stats["processed_dims"],
            format=stats["format"],
            color_mode=stats["color_mode"],
            original_size_bytes=stats["original_size_bytes"],
            processed_size_bytes=stats["processed_size_bytes"],
            channel_means=stats["channel_means"],
            preprocessing_steps=stats["preprocessing_steps"],
            processed_image=encoded_image,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("image_processing_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
