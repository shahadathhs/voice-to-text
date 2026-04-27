"""Transcription service for handling audio transcription."""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import torch
from fastapi import UploadFile

from app.core.config import settings
from app.core.errors import AudioFileError, ModelLoadError, TranscriptionError
from app.core.logger import logger

try:
    from app.services.pipeline import transcribe as legacy_transcribe
    from app.utils import save_transcript
    from app.whisper import load_openai_whisper, load_transformers_whisper

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
                logger.info(
                    f"Loading Transformers Whisper ({settings.whisper_model})..."
                )
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
                    savedir=str(Path(settings.model_cache_dir) / "speechbrain"),
                )

            self._initialized = True
            logger.info("Transcription service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            raise ModelLoadError(f"Failed to initialize models: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary
        """
        return {
            "status": "ok" if self._initialized else "not_initialized",
            "device": self.models.get("device", "unknown"),
            "whisper_backend": self.models.get(
                "whisper_backend", settings.whisper_backend
            ),
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
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file.

        Args:
            audio_file: Path to audio file or UploadFile object
            translate: Translate to English
            diarize: Enable speaker diarization
            diarize_threshold: Clustering threshold for diarization
            max_speakers: Maximum number of speakers
            use_silhouette: Use silhouette analysis
            base_url: Base URL for constructing full URLs (e.g., http://localhost:8000)

        Returns:
            Transcription result with text and metadata

        Raises:
            AudioFileError: If audio file is invalid
            TranscriptionError: If transcription fails
        """
        if not self._initialized:
            raise TranscriptionError("Service not initialized")

        uploaded_file_path: Path | None = None
        is_uploaded_file = False

        try:
            # Handle UploadFile (check for file attribute which is unique to UploadFile)
            if hasattr(audio_file, "file") and hasattr(audio_file, "filename"):
                upload = cast(UploadFile, audio_file)

                # Validate file size
                if upload.size and upload.size > settings.max_file_size:
                    raise AudioFileError(
                        f"File too large: {upload.size} > {settings.max_file_size}"
                    )

                # Validate file format
                if not upload.filename:
                    raise AudioFileError("Uploaded file has no filename")

                file_ext = upload.filename.split(".")[-1].lower()
                if file_ext not in settings.allowed_formats:
                    raise AudioFileError(
                        f"Invalid file format: {file_ext}. Allowed: {settings.allowed_formats}"
                    )

                # Save to uploads directory with unique filename (preserving original extension)
                from datetime import datetime

                original_name = Path(upload.filename).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{original_name}_{timestamp}.{file_ext}"
                uploaded_file_path = Path(settings.uploads_dir) / filename

                with uploaded_file_path.open("wb") as buffer:
                    shutil.copyfileobj(upload.file, buffer)

                audio_path = uploaded_file_path
                is_uploaded_file = True

                logger.info(
                    f"Processing uploaded file: {upload.filename} -> {filename}"
                )

            # Handle Path
            elif isinstance(audio_file, Path):
                if not audio_file.exists():
                    raise AudioFileError(f"Audio file not found: {audio_file}")

                audio_path = audio_file
                logger.info(f"Processing file: {audio_path}")

            else:
                raise AudioFileError(f"Invalid audio file type: {type(audio_file)}")

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
                    whisper_backend=self.models.get(
                        "whisper_backend", settings.whisper_backend
                    ),
                    use_silhouette=use_silhouette or settings.use_silhouette,
                )
            else:
                raise TranscriptionError("Legacy transcription not available")

            from app.utils import get_unique_filename

            output_filename = get_unique_filename(audio_path.name)
            saved_path = save_transcript(transcript_text, output_filename)

            logger.info(f"Transcription saved to: {saved_path}")

            # Build response metadata
            response_metadata = {
                "model": settings.whisper_model,
                "backend": self.models.get("whisper_backend", settings.whisper_backend),
                "device": self.models.get("device", "unknown"),
                "translated": translate or settings.enable_translation,
                "diarized": diarize or settings.enable_diarization,
            }

            # Determine base URL (use provided or fall back to settings)
            if base_url is None:
                base_url = settings.api_host or ""

            # Normalize base URL (remove trailing slash)
            base_url = base_url.rstrip("/")

            # Add audio file info
            if is_uploaded_file and uploaded_file_path:
                # For uploaded files, return full URL instead of path
                audio_filename = uploaded_file_path.name
                audio_url_path = f"/uploads/{audio_filename}"
                response_metadata["audio_file"] = audio_url_path
                response_metadata["audio_url"] = (
                    f"{base_url}{audio_url_path}" if base_url else audio_url_path
                )
            else:
                # For local files, return the path
                response_metadata["audio_file"] = str(audio_path)

            # Add transcript URL
            transcript_path = f"/transcripts/{saved_path.name}"
            response_metadata["transcript_file"] = str(saved_path.name)
            response_metadata["transcript_url"] = (
                f"{base_url}{transcript_path}" if base_url else transcript_path
            )

            return {
                "transcript": transcript_text,
                "saved_to": str(saved_path),
                "metadata": response_metadata,
            }

        except AudioFileError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise TranscriptionError(f"Transcription failed: {e}") from e

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
