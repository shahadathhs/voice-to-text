"""Pydantic schemas for request/response validation."""

from app.schemas.base import (
    DataResponse,
    ErrorResponse,
    HealthResponse,
    MetaData,
    PaginatedResponse,
)
from app.schemas.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionMetadata,
)
from app.schemas.validation import (
    AudioFileValidation,
    TranscriptionValidateQuery,
)

__all__ = [
    # Base schemas
    "DataResponse",
    "ErrorResponse",
    "HealthResponse",
    "MetaData",
    "PaginatedResponse",
    # Transcription schemas
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranscriptionMetadata",
    # Validation schemas
    "AudioFileValidation",
    "TranscriptionValidateQuery",
]
