"""Text data processing routes."""

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.core.config import get_settings
from api.core.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidFileTypeError,
    SessionNotFoundError,
)
from api.core.logging import get_logger
from api.processors.text import TextProcessor
from api.schemas.text import PosTagResponse, TextProcessResponse, WordFrequencyResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/text", tags=["Text"])

# In-memory session store for text processor state (Bug Fix #4)
_text_sessions: dict[str, TextProcessor] = {}


@router.post("/process", response_model=TextProcessResponse)
async def process_text(
    request: Request,
    file: UploadFile = File(...),
    remove_stopwords: bool = Form(True),
    lemmatize: bool = Form(True),
    lowercase: bool = Form(True),
    remove_urls: bool = Form(False),
    remove_emails: bool = Form(False),
) -> TextProcessResponse:
    """Preprocess text data using NLP pipeline.

    Bug Fix #3: Uses TextProcessor class instead of inline NLTK logic.
    Bug Fix #4: Returns a session_id for subsequent frequency/POS requests.

    Parameters:
    - **file**: TXT file to process
    - **remove_stopwords**: Remove English stop words
    - **lemmatize**: Apply lemmatization
    - **lowercase**: Convert to lowercase
    - **remove_urls**: Remove URLs from text
    - **remove_emails**: Remove email addresses from text
    """
    start_time = time.time()
    settings = get_settings()

    # --- Validation ---
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise InvalidFileTypeError([".txt"])

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_txt_size:
        raise FileTooLargeError(settings.max_txt_size / (1024 * 1024))

    content = await file.read()
    if len(content) == 0:
        raise EmptyFileError()

    if len(content) > settings.max_txt_size:
        raise FileTooLargeError(settings.max_txt_size / (1024 * 1024))

    # --- Processing (Bug Fix #3: use TextProcessor, not inline NLTK) ---
    try:
        processor = TextProcessor()
        processor.load_data(content)

        if lowercase:
            processor.to_lowercase()

        if remove_urls:
            processor.remove_urls()

        if remove_emails:
            processor.remove_emails()

        processor.remove_whitespace()

        if remove_stopwords:
            processor.remove_stopwords()

        if lemmatize:
            processor.lemmatize_text()

        # Get statistics and processed text
        stats = processor.get_statistics()

        # Store session for subsequent requests (Bug Fix #4)
        session_id = str(uuid.uuid4())
        _text_sessions[session_id] = processor

        # Limit stored sessions (simple LRU-like cleanup)
        if len(_text_sessions) > 100:
            oldest_key = next(iter(_text_sessions))
            del _text_sessions[oldest_key]

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "text_processing_complete",
            filename=file.filename,
            file_size=len(content),
            options={
                "lowercase": lowercase,
                "remove_stopwords": remove_stopwords,
                "lemmatize": lemmatize,
            },
            processing_time_ms=duration_ms,
            processed_length=stats["processed_length"],
        )

        return TextProcessResponse(
            session_id=session_id,
            processed_text=processor.processed_text or "",
            **stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("text_processing_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frequencies/{session_id}", response_model=WordFrequencyResponse)
async def get_word_frequencies(
    session_id: str,
    top_n: int | None = None,
) -> WordFrequencyResponse:
    """Get word frequencies from a previously processed text session.

    Bug Fix #4: New endpoint — was documented in README but missing.

    Parameters:
    - **session_id**: Session ID returned from POST /text/process
    - **top_n**: Optional limit on number of words returned
    """
    if session_id not in _text_sessions:
        raise SessionNotFoundError(session_id)

    processor = _text_sessions[session_id]
    frequencies = processor.get_word_frequencies(top_n=top_n)
    total = sum(frequencies.values())

    return WordFrequencyResponse(
        session_id=session_id,
        frequencies=frequencies,
        total_words=total,
    )


@router.get("/pos-tags/{session_id}", response_model=PosTagResponse)
async def get_pos_tags(session_id: str) -> PosTagResponse:
    """Get part-of-speech tags from a previously processed text session.

    Bug Fix #4: New endpoint — was documented in README but missing.

    Parameters:
    - **session_id**: Session ID returned from POST /text/process
    """
    if session_id not in _text_sessions:
        raise SessionNotFoundError(session_id)

    processor = _text_sessions[session_id]
    tags = processor.get_pos_tags()

    return PosTagResponse(
        session_id=session_id,
        pos_tags=[[word, tag] for word, tag in tags],
        total_tokens=len(tags),
    )
