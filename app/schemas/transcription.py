"""Transcription-specific Pydantic schemas."""

from typing import Optional
from pydantic import BaseModel, Field
from fastapi import status


class TranscriptionMetadata(BaseModel):
    """Metadata for transcription results."""

    model: str = Field(..., description="Whisper model size used")
    backend: str = Field(..., description="Whisper backend used (openai or transformers)")
    device: str = Field(..., description="Device used (cpu or cuda)")
    translated: bool = Field(..., description="Whether translation was performed")
    diarized: bool = Field(..., description="Whether speaker diarization was performed")
    audio_file: str = Field(..., description="Path to audio file")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "base",
                    "backend": "openai",
                    "device": "cpu",
                    "translated": False,
                    "diarized": True,
                    "audio_file": "media/audio/sample.wav",
                }
            ]
        }
    }


class TranscriptionResponse(BaseModel):
    """Response schema for transcription endpoint."""

    status_code: int = Field(status.HTTP_200_OK, description="HTTP status code")
    success: bool = Field(True, description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    transcript: str = Field(..., description="Transcription text")
    saved_to: str = Field(..., description="Path where transcript was saved")
    metadata: TranscriptionMetadata = Field(..., description="Transcription metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class TranscriptionRequest(BaseModel):
    """Request schema for transcription (for future use with query params)."""

    translate: bool = Field(default=False, description="Translate to English")
    diarize: bool = Field(default=False, description="Enable speaker diarization")
    diarize_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="Clustering threshold (0.0-1.0)")
    max_speakers: Optional[int] = Field(default=None, ge=1, description="Fixed number of speakers")
    use_silhouette: bool = Field(default=False, description="Estimate speakers from embeddings")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "translate": True,
                    "diarize": True,
                    "diarize_threshold": 0.35,
                    "max_speakers": 2,
                    "use_silhouette": False,
                }
            ]
        }
    }
