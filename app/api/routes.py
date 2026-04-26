"""FastAPI routes for voice-to-text API."""

from typing import Any

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.errors import AudioFileError, TranscriptionError
from app.core.logger import logger
from app.services.transcriber import transcription_service

router = APIRouter()


@router.get("/")
def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered voice transcription service using OpenAI Whisper",
        "environment": settings.environment,
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@router.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status with service information
    """
    return transcription_service.health_check()


@router.post(
    "/transcribe",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid audio file"},
        422: {"description": "Validation error"},
        500: {"description": "Transcription error"},
    },
)
async def transcribe_endpoint(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    translate: bool = False,
    diarize: bool = False,
    diarize_threshold: float = 0.35,
    max_speakers: int | None = None,
    use_silhouette: bool = False,
) -> JSONResponse:
    """
    Transcribe an audio file.

    Args:
        file: Audio file to transcribe (supported formats: wav, mp3, ogg, m4a, flac, aac)
        translate: Translate non-English audio to English
        diarize: Enable speaker diarization (identify different speakers)
        diarize_threshold: Clustering distance for speaker diarization (lower = more speakers)
        max_speakers: Fixed number of speakers (overrides diarize_threshold)
        use_silhouette: Estimate number of speakers from embeddings

    Returns:
        JSON response with transcription result

    Raises:
        AudioFileError: If audio file is invalid
        TranscriptionError: If transcription fails
    """
    try:
        logger.info(
            f"Transcription request: file={file.filename}, "
            f"translate={translate}, diarize={diarize}"
        )

        # Validate parameters
        if diarize_threshold < 0 or diarize_threshold > 1:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": "diarize_threshold must be between 0 and 1",
                },
            )

        if max_speakers is not None and max_speakers < 1:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "message": "max_speakers must be at least 1",
                },
            )

        # Transcribe
        result = await transcription_service.transcribe_file(
            audio_file=file,
            translate=translate,
            diarize=diarize,
            diarize_threshold=diarize_threshold,
            max_speakers=max_speakers,
            use_silhouette=use_silhouette,
        )

        logger.info(f"Transcription completed successfully for {file.filename}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "data": result,
            },
        )

    except AudioFileError as e:
        logger.warning(f"Audio file error: {e.message}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "error",
                "message": e.message,
                "details": e.details,
            },
        )

    except TranscriptionError as e:
        logger.error(f"Transcription error: {e.message}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "error",
                "message": e.message,
                "details": e.details,
            },
        )

    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error during transcription",
                "details": {"error": str(e)} if settings.debug else {},
            },
        )
