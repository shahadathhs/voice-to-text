"""Custom exceptions for Voice-to-Text application."""

from typing import Any


class AppError(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        errors: list[Any] | None = None,
    ):
        """Initialize exception.

        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
            errors: List of validation errors
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.errors = errors or []
        super().__init__(self.message)


class ValidationError(AppError):
    """Validation error (422)."""

    def __init__(
        self,
        message: str = "Validation error",
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            details: Additional error details
        """
        if field:
            details = details or {}
            details["field"] = field
        super().__init__(message, status_code=422, details=details)


class BadRequestError(AppError):
    """Bad request error (400)."""

    def __init__(
        self,
        message: str = "Bad request",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=400, details=details)


class UnauthorizedError(AppError):
    """Unauthorized error (401)."""

    def __init__(
        self,
        message: str = "Unauthorized",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=401, details=details)


class ForbiddenError(AppError):
    """Forbidden error (403)."""

    def __init__(
        self,
        message: str = "Forbidden",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(AppError):
    """Not found error (404)."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize not found error.

        Args:
            message: Error message
            resource: Type of resource that was not found
            details: Additional error details
        """
        if resource:
            details = details or {}
            details["resource_type"] = resource
        super().__init__(message, status_code=404, details=details)


class AudioFileError(AppError):
    """Audio file processing error (400)."""

    def __init__(
        self,
        message: str,
        filename: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize audio file error.

        Args:
            message: Error message
            filename: Name of the audio file
            details: Additional error details
        """
        if filename:
            details = details or {}
            details["filename"] = filename
        super().__init__(message, status_code=400, details=details)


class TranscriptionError(AppError):
    """Transcription processing error (500)."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, status_code=500, details=details)


class ModelLoadError(AppError):
    """Model loading error (500)."""

    def __init__(
        self,
        message: str,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize model load error.

        Args:
            message: Error message
            model: Name of the model that failed to load
            details: Additional error details
        """
        if model:
            details = details or {}
            details["model"] = model
        super().__init__(message, status_code=500, details=details)


class ServiceUnavailableError(AppError):
    """Service unavailable error (503)."""

    def __init__(
        self,
        message: str = "Service unavailable",
        service: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize service unavailable error.

        Args:
            message: Error message
            service: Name of the unavailable service
            details: Additional error details
        """
        if service:
            details = details or {}
            details["service"] = service
        super().__init__(message, status_code=503, details=details)


class ConfigurationError(AppError):
    """Configuration error (500)."""

    def __init__(
        self,
        message: str,
        setting: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            setting: Name of the configuration setting
            details: Additional error details
        """
        if setting:
            details = details or {}
            details["setting"] = setting
        super().__init__(message, status_code=500, details=details)
