"""Request validation schemas for API endpoints."""

from pydantic import BaseModel, Field, field_validator


class TranscriptionValidateQuery(BaseModel):
    """Query parameters for transcription validation."""

    translate: bool = Field(default=False, description="Translate to English")
    diarize: bool = Field(default=False, description="Enable speaker diarization")
    diarize_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Clustering threshold for speaker diarization (0.0-1.0)",
    )
    max_speakers: int | None = Field(
        default=None,
        ge=1,
        description="Fixed number of speakers (overrides diarize_threshold)",
    )
    use_silhouette: bool = Field(
        default=False,
        description="Estimate number of speakers from embeddings",
    )

    @field_validator("max_speakers")
    @classmethod
    def validate_max_speakers(cls, v, info):
        """Validate max_speakers is only used when diarize is True."""
        if v is not None and not info.data.get("diarize", False):
            raise ValueError("max_speakers can only be used when diarize=True")
        return v

    @field_validator("use_silhouette")
    @classmethod
    def validate_use_silhouette(cls, v, info):
        """Validate use_silhouette is only used when diarize is True."""
        if v and not info.data.get("diarize", False):
            raise ValueError("use_silhouette can only be used when diarize=True")
        return v

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


class AudioFileValidation(BaseModel):
    """Schema for audio file validation."""

    filename: str = Field(..., description="Name of the audio file")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME type of the file")

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v):
        """Validate content type is an audio format."""
        if not v.startswith("audio/"):
            raise ValueError("File must be an audio file")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "filename": "sample.mp3",
                    "size": 1024000,
                    "content_type": "audio/mpeg",
                }
            ]
        }
    }
