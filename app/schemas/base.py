"""Base Pydantic schemas for request/response validation."""

from typing import Any, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

T = TypeVar("T")


class MetaData(BaseModel):
    """Metadata for paginated responses."""

    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

    model_config = {"json_schema_extra": {"examples": [{"total": 100, "page": 1, "page_size": 20, "has_next": True, "has_prev": False}]}}


class DataResponse(BaseModel, Generic[T]):
    """Standard success response format."""

    status_code: int = Field(..., description="HTTP status code", ge=100, le=599)
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: T = Field(..., description="Response data")
    metadata: Optional[MetaData] = Field(None, description="Optional metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status_code": 200,
                    "success": True,
                    "message": "Success",
                    "data": {"key": "value"},
                    "metadata": None,
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response format."""

    status_code: int = Field(..., description="HTTP status code", ge=100, le=599)
    success: bool = Field(False, description="Always False for errors")
    message: str = Field(..., description="Error message")
    errors: Optional[list[Any]] = Field(None, description="Optional list of validation errors")
    details: Optional[dict[str, Any]] = Field(None, description="Optional error details")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status_code": 400,
                    "success": False,
                    "message": "Bad request",
                    "errors": None,
                    "details": {"field": "Invalid value"},
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    device: str = Field(..., description="Device being used")
    whisper_backend: str = Field(..., description="Whisper backend")
    model_size: str = Field(..., description="Whisper model size")
    features: dict[str, Any] = Field(..., description="Enabled features")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "device": "cpu",
                    "whisper_backend": "openai",
                    "model_size": "base",
                    "features": {"translation": False, "diarization": False},
                }
            ]
        }
    }


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response format."""

    status_code: int = Field(..., description="HTTP status code", ge=100, le=599)
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: list[T] = Field(..., description="List of items")
    metadata: MetaData = Field(..., description="Pagination metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status_code": 200,
                    "success": True,
                    "message": "Data retrieved successfully",
                    "data": [{"id": 1}, {"id": 2}],
                    "metadata": {
                        "total": 100,
                        "page": 1,
                        "page_size": 20,
                        "has_next": True,
                        "has_prev": False,
                    },
                }
            ]
        }
    }

