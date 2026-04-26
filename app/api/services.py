"""API services for business logic layer."""

from typing import Any

from app.core.config import settings
from app.core.errors import AudioFileError, TranscriptionError
from app.core.logger import logger
from app.services.transcriber import TranscriptionService, lifespan_manager, transcription_service


class TranscriptionAPIService:
    """Service for handling transcription API business logic."""

    def __init__(self) -> None:
        """Initialize transcription API service."""
        self.transcription_service = transcription_service

    async def validate_transcription_request(
        self,
        translate: bool,
        diarize: bool,
        diarize_threshold: float,
        max_speakers: int | None,
        use_silhouette: bool,
    ) -> None:
        """
        Validate transcription request parameters.

        Args:
            translate: Whether translation is enabled
            diarize: Whether diarization is enabled
            diarize_threshold: Clustering threshold
            max_speakers: Maximum number of speakers
            use_silhouette: Whether to use silhouette analysis

        Raises:
            ValidationError: If parameters are invalid
        """
        from app.core.errors import ValidationError

        # Validate diarize_threshold
        if diarize_threshold < 0.0 or diarize_threshold > 1.0:
            raise ValidationError(
                message="diarize_threshold must be between 0.0 and 1.0",
                field="diarize_threshold",
                details={
                    "provided": diarize_threshold,
                    "valid_range": "0.0-1.0",
                },
            )

        # Validate max_speakers
        if max_speakers is not None and max_speakers < 1:
            raise ValidationError(
                message="max_speakers must be at least 1",
                field="max_speakers",
                details={
                    "provided": max_speakers,
                    "minimum": 1,
                },
            )

        # Validate diarize parameters
        if not diarize and (max_speakers is not None or use_silhouette):
            raise ValidationError(
                message="Diarization must be enabled to use max_speakers or use_silhouette",
                field="diarize",
                details={
                    "diarize": diarize,
                    "max_speakers": max_speakers,
                    "use_silhouette": use_silhouette,
                },
            )

        # Validate translation
        if translate and not settings.enable_translation:
            logger.warning("Translation requested but not enabled in settings")

        # Validate diarization
        if diarize and not settings.enable_diarization:
            logger.warning("Diarization requested but not enabled in settings")

    async def process_transcription_request(
        self,
        file: Any,
        translate: bool,
        diarize: bool,
        diarize_threshold: float,
        max_speakers: int | None,
        use_silhouette: bool,
    ) -> dict[str, Any]:
        """
        Process transcription request with validation and error handling.

        Args:
            file: Uploaded audio file
            translate: Whether translation is enabled
            diarize: Whether diarization is enabled
            diarize_threshold: Clustering threshold
            max_speakers: Maximum number of speakers
            use_silhouette: Whether to use silhouette analysis

        Returns:
            Transcription result dict

        Raises:
            ValidationError: If parameters are invalid
            AudioFileError: If audio file is invalid
            TranscriptionError: If transcription fails
        """
        # Validate parameters
        await self.validate_transcription_request(
            translate=translate,
            diarize=diarize,
            diarize_threshold=diarize_threshold,
            max_speakers=max_speakers,
            use_silhouette=use_silhouette,
        )

        # Process transcription
        result = await self.transcription_service.transcribe_file(
            audio_file=file,
            translate=translate or settings.enable_translation,
            diarize=diarize or settings.enable_diarization,
            diarize_threshold=diarize_threshold or settings.diarize_threshold,
            max_speakers=max_speakers or settings.max_speakers,
            use_silhouette=use_silhouette or settings.use_silhouette,
        )

        return result


class HealthAPIService:
    """Service for handling health check business logic."""

    def __init__(self) -> None:
        """Initialize health API service."""
        self.transcription_service = transcription_service

    def get_health_status(self) -> dict[str, Any]:
        """
        Get comprehensive health status.

        Returns:
            Health status dict with service information
        """
        health = self.transcription_service.health_check()

        health["environment"] = settings.environment
        health["app_version"] = settings.app_version
        health["api_prefix"] = settings.api_prefix
        health["media_dir"] = str(settings.media_dir)
        health["audio_dir"] = str(settings.audio_dir)
        health["transcript_dir"] = str(settings.transcript_dir)

        return health

    def is_healthy(self) -> bool:
        """
        Check if the service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            health = self.transcription_service.health_check()
            return health.get("status") == "ok"
        except Exception:
            return False


class RootAPIService:
    """Service for handling root endpoint business logic."""

    @staticmethod
    def get_root_info() -> dict[str, Any]:
        """
        Get API root information.

        Returns:
            Root information dict
        """
        return {
            "app_name": settings.app_name,
            "version": settings.app_version,
            "description": "AI-powered voice transcription service using OpenAI Whisper",
            "environment": settings.environment,
            "debug": settings.debug,
            "endpoints": {
                "health": "/health",
                "transcribe": "/transcribe",
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
            },
            "features": {
                "translation": settings.enable_translation,
                "diarization": settings.enable_diarization,
                "whisper_backends": ["openai", "transformers"],
                "supported_formats": settings.allowed_formats,
            },
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
            },
        }


# Global service instances
transcription_api_service = TranscriptionAPIService()
health_api_service = HealthAPIService()
root_api_service = RootAPIService()
