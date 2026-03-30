"""Custom exception classes for the API."""

from fastapi import HTTPException, status


class FileTooLargeError(HTTPException):
    """Raised when an uploaded file exceeds the size limit."""

    def __init__(self, max_size_mb: float):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the maximum allowed size of {max_size_mb:.1f}MB.",
        )


class InvalidFileTypeError(HTTPException):
    """Raised when an uploaded file has an unsupported extension."""

    def __init__(self, allowed_extensions: list[str]):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed extensions: {', '.join(allowed_extensions)}",
        )


class ProcessingError(HTTPException):
    """Raised when data processing fails."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {detail}",
        )


class ValidationError(HTTPException):
    """Raised when request validation fails."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {detail}",
        )


class EmptyFileError(HTTPException):
    """Raised when an uploaded file is empty."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )


class SessionNotFoundError(HTTPException):
    """Raised when a text processing session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. Please process text first.",
        )
