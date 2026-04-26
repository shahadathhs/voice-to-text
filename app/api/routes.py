"""FastAPI routes for voice-to-text API."""

from typing import Any

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.response import ResponseBuilder
from app.core.errors import AudioFileError, TranscriptionError
from app.core.logger import logger
from app.services.transcriber import transcription_service

router = APIRouter()


@router.get(
    "/",
    summary="API Root Information",
    description="Returns basic information about the API including name, version, and available documentation links.",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "app_name": "Voice-to-Text API",
                        "version": "1.0.0",
                        "description": "AI-powered voice transcription service using OpenAI Whisper",
                        "environment": "development",
                        "endpoints": {
                            "health": "/health",
                            "transcribe": "/transcribe",
                            "docs": "/docs",
                            "redoc": "/redoc",
                        },
                    }
                }
            },
        }
    },
)
def root() -> dict[str, Any]:
    """
    API Root Endpoint

    Provides basic information about the Voice-to-Text API including:
    - Application name and version
    - Available documentation URLs
    - Current system status

    **Example Request:**
    ```bash
    curl http://localhost:8000/
    ```

    **Example Response:**
    ```json
    {
      "app_name": "Voice-to-Text API",
      "version": "1.0.0",
      "description": "AI-powered voice transcription service",
      "environment": "development",
      "endpoints": {
        "health": "/health",
        "transcribe": "/transcribe",
        "docs": "/docs",
        "redoc": "/redoc"
      }
    }
    ```
    """
    return {
        "app_name": settings.app_name,
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


@router.get(
    "/health",
    summary="Health Check",
    description="Check the health status of the service including model initialization and feature availability.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "device": "cpu",
                        "whisper_backend": "openai",
                        "model_size": "base",
                        "features": {
                            "translation": False,
                            "diarization": False,
                        },
                    }
                }
            }
        }
    },
)
def health_check() -> dict[str, Any]:
    """
    Health Check Endpoint

    Returns the current health status of the transcription service.

    **Example Request:**
    ```bash
    curl http://localhost:8000/health
    ```

    **Example Response:**
    ```json
    {
      "status": "ok",
      "device": "cpu",
      "whisper_backend": "openai",
      "model_size": "base",
      "features": {
        "translation": false,
        "diarization": false
      }
    }
    ```

    **Use Cases:**
    - Service monitoring
    - Load balancer health checks
    - Deployment verification
    """
    return transcription_service.health_check()


@router.post(
    "/transcribe",
    summary="Transcribe Audio File",
    description="Convert audio files to text using OpenAI Whisper model. Supports multiple audio formats with optional features like translation and speaker diarization.",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "status_code": 200,
                        "success": True,
                        "message": "Transcription completed successfully",
                        "transcript": "SPEAKER_00: Hello world\nSPEAKER_01: Hi there",
                        "saved_to": "media/transcripts/audio_20260426_123456.txt",
                        "metadata": {
                            "model": "base",
                            "backend": "openai",
                            "device": "cpu",
                            "translated": False,
                            "diarized": True,
                            "audio_file": "media/audio/sample.wav",
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid audio file or parameters"},
        422: {"description": "Validation error"},
        500: {"description": "Transcription error"},
    },
)
async def transcribe_endpoint(
    file: UploadFile = File(
        ...,
        description="Audio file to transcribe (supported: wav, mp3, ogg, m4a, flac, aac)",
    ),
    translate: bool = False,
    diarize: bool = False,
    diarize_threshold: float = 0.35,
    max_speakers: int | None = None,
    use_silhouette: bool = False,
) -> JSONResponse:
    """
    Audio Transcription Endpoint

    Transcribes audio files using state-of-the-art Whisper model with optional features for translation and speaker diarization.

    **Example Request:**
    ```bash
    curl -X POST "http://localhost:8000/transcribe" \\
      -F "file=@audio.mp3" \\
      -F "translate=true" \\
      -F "diarize=true"
    ```

    **Example Response:**
    ```json
    {
      "status_code": 200,
      "success": true,
      "message": "Transcription completed successfully",
      "transcript": "SPEAKER_00: Hello world\\nSPEAKER_01: Hi there",
      "saved_to": "media/transcripts/audio_20260426_123456.txt",
      "metadata": {
        "model": "base",
        "backend": "openai",
        "device": "cpu",
        "translated": false,
        "diarized": true,
        "audio_file": "media/audio/sample.wav"
      }
    }
    ```

    **Use Cases:**
    - Meeting transcription
    - Podcast transcription
    - Voice note conversion
    - Automated captioning
    - Multi-speaker identification

    **Parameters:**
    - `file`: Audio file (required)
    - `translate`: Translate non-English audio to English
    - `diarize`: Identify different speakers
    - `diarize_threshold`: Clustering distance (0.0-1.0, lower = more speakers)
    - `max_speakers`: Fixed number of speakers (overrides diarize_threshold)
    - `use_silhouette`: Estimate speakers from embeddings
    """
    try:
        logger.info(
            f"Transcription request: file={file.filename}, "
            f"translate={translate}, diarize={diarize}"
        )

        # Validate parameters
        if diarize_threshold < 0 or diarize_threshold > 1:
            return JSONResponse(
                content=ResponseBuilder.bad_request(
                    message="diarize_threshold must be between 0 and 1"
                ).model_dump(),
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        if max_speakers is not None and max_speakers < 1:
            return JSONResponse(
                content=ResponseBuilder.bad_request(
                    message="max_speakers must be at least 1"
                ).model_dump(),
                status_code=status.HTTP_400_BAD_REQUEST,
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

        response = {
            "status_code": status.HTTP_200_OK,
            "success": True,
            "message": "Transcription completed successfully",
            "transcript": result["transcript"],
            "saved_to": result["saved_to"],
            "metadata": result["metadata"],
        }

        return JSONResponse(
            content=response,
            status_code=status.HTTP_200_OK,
        )

    except AudioFileError as e:
        logger.warning(f"Audio file error: {e.message}")
        return JSONResponse(
            content=ResponseBuilder.error(
                message=e.message,
                status_code=e.status_code,
                details=e.details,
            ).model_dump(),
            status_code=e.status_code,
        )

    except TranscriptionError as e:
        logger.error(f"Transcription error: {e.message}")
        return JSONResponse(
            content=ResponseBuilder.error(
                message=e.message,
                status_code=e.status_code,
                details=e.details,
            ).model_dump(),
            status_code=e.status_code,
        )

    except Exception as e:
        logger.exception(f"Unexpected error during transcription: {e}")
        return JSONResponse(
            content=ResponseBuilder.internal_server_error(
                message="Internal server error during transcription",
                details={"error": str(e)} if settings.debug else None,
            ).model_dump(),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
