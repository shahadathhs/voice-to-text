"""Transcription service for handling audio transcription."""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AudioFileError, ModelLoadError, TranscriptionError
from app.core.logger import logger

# Import from app package
try:
    from app.whisper import load_openai_whisper, load_transformers_whisper
    from app.utils import get_unique_filename, save_transcript
    from app.services.pipeline import transcribe as legacy_transcribe
    LEGACY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Legacy app package not available: {e}")
    LEGACY_AVAILABLE = False


class TranscriptionService:
    """Service for managing audio transcription."""

    def __init__(self) -> None:
        """Initialize transcription service."""
        self.models: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize transcription models."""
        if self._initialized:
            logger.debug("Transcription service already initialized")
            return

        try:
            logger.info("Initializing transcription service...")

            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Load Whisper model
            if settings.whisper_backend == "transformers":
                if not LEGACY_AVAILABLE:
                    raise ModelLoadError("Transformers backend not available")
                logger.info(f"Loading Transformers Whisper ({settings.whisper_model})...")
                self.models["whisper"] = load_transformers_whisper(
                    settings.whisper_model, device
                )
            else:
                if not LEGACY_AVAILABLE:
                    raise ModelLoadError("OpenAI Whisper backend not available")
                logger.info(f"Loading OpenAI Whisper ({settings.whisper_model})...")
                self.models["whisper"] = load_openai_whisper(
                    settings.whisper_model, device
                )

            self.models["device"] = device
            self.models["whisper_backend"] = settings.whisper_backend

            # Load SpeechBrain classifier for diarization
            if settings.enable_diarization:
                logger.info("Loading SpeechBrain classifier...")
                from speechbrain.inference.speaker import EncoderClassifier

                self.models["classifier"] = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": device},
                    savedir=str(settings.model_cache_dir / "speechbrain"),
                )

            self._initialized = True
            logger.info("Transcription service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise ModelLoadError(f"Failed to initialize models: {e}")

    def health_check(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        return {
            "status": "ok" if self._initialized else "not_initialized",
            "device": self.models.get("device", "unknown"),
            "whisper_backend": self.models.get("whisper_backend", settings.whisper_backend),
            "model_size": settings.whisper_model,
            "features": {
                "translation": settings.enable_translation,
                "diarization": settings.enable_diarization,
            },
        }

    async def transcribe_file(
        self,
        audio_file: Path | UploadFile,
        translate: bool = False,
        diarize: bool = False,
        diarize_threshold: float = 0.35,
        max_speakers: int | None = None,
        use_silhouette: bool = False,
    ) -> dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_file: Path to audio file or UploadFile object
            translate: Translate to English
            diarize: Enable speaker diarization
            diarize_threshold: Clustering threshold for diarization
            max_speakers: Maximum number of speakers
            use_silhouette: Use silhouette analysis

        Returns:
            Transcription result with text and metadata

        Raises:
            AudioFileError: If audio file is invalid
            TranscriptionError: If transcription fails
        """
        if not self._initialized:
            raise TranscriptionError("Service not initialized")

        temp_path: Path | None = None

        try:
            # Handle UploadFile
            if isinstance(audio_file, UploadFile):
                # Validate file size
                if audio_file.size and audio_file.size > settings.max_file_size:
                    raise AudioFileError(
                        f"File too large: {audio_file.size} > {settings.max_file_size}"
                    )

                # Validate file format
                if audio_file.filename:
                    file_ext = audio_file.filename.split(".")[-1].lower()
                    if file_ext not in settings.allowed_formats:
                        raise AudioFileError(
                            f"Invalid file format: {file_ext}. Allowed: {settings.allowed_formats}"
                        )

                # Save to temp file
                temp_filename = f"temp_{audio_file.filename}"
                temp_path = settings.audio_dir / temp_filename

                with temp_path.open("wb") as buffer:
                    shutil.copyfileobj(audio_file.file, buffer)

                audio_path = temp_path

                logger.info(f"Processing uploaded file: {audio_file.filename}")

            # Handle Path
            else:
                if not audio_file.exists():
                    raise AudioFileError(f"Audio file not found: {audio_file}")

                audio_path = audio_file
                logger.info(f"Processing file: {audio_path}")

            # Validate file exists
            if not audio_path.exists():
                raise AudioFileError(f"Audio file not found: {audio_path}")

            # Transcribe
            if LEGACY_AVAILABLE:
                transcript_text = legacy_transcribe(
                    str(audio_path),
                    model=self.models["whisper"],
                    translate=translate or settings.enable_translation,
                    diarize=diarize or settings.enable_diarization,
                    device=self.models["device"],
                    classifier=self.models.get("classifier"),
                    diarize_threshold=diarize_threshold,
                    max_speakers=max_speakers or settings.max_speakers,
                    whisper_backend=self.models.get("whisper_backend", settings.whisper_backend),
                    use_silhouette=use_silhouette or settings.use_silhouette,
                )
            else:
                raise TranscriptionError("Legacy transcription not available")

            # Save transcript
            output_filename = get_unique_filename(audio_path.name)
            saved_path = save_transcript(transcript_text, output_filename)

            logger.info(f"Transcription saved to: {saved_path}")

            return {
                "transcript": transcript_text,
                "saved_to": str(saved_path),
                "metadata": {
                    "model": settings.whisper_model,
                    "backend": self.models.get("whisper_backend", settings.whisper_backend),
                    "device": self.models.get("device", "unknown"),
                    "translated": translate or settings.enable_translation,
                    "diarized": diarize or settings.enable_diarization,
                    "audio_file": str(audio_path),
                },
            }

        except AudioFileError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}")

        finally:
            # Cleanup temp file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up transcription service...")
        self.models.clear()
        self._initialized = False


# Global service instance
transcription_service = TranscriptionService()


@asynccontextmanager
async def lifespan_manager():
    """Manage service lifespan for FastAPI.

    Yields:
        None
    """
    # Startup
    logger.info("Starting transcription service...")
    transcription_service.initialize()
    yield
    # Shutdown
    logger.info("Stopping transcription service...")
    transcription_service.cleanup()
