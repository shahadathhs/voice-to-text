"""Custom exceptions for Voice-to-Text application."""


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict | None = None,
    ):
        """Initialize exception.

        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppException):
    """Validation error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=422, details=details)


class AudioFileError(AppException):
    """Audio file processing error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=400, details=details)


class TranscriptionError(AppException):
    """Transcription processing error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, details=details)


class ModelLoadError(AppException):
    """Model loading error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, details=details)


class ConfigurationError(AppException):
    """Configuration error."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, status_code=500, details=details)
