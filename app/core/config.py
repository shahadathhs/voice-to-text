"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Voice-to-Text API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "production", "testing"] = Field(
        default="development", description="Environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    api_prefix: str = Field(default="", description="API prefix")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    max_file_size: int = Field(
        default=500 * 1024 * 1024, description="Max file size in bytes"
    )
    allowed_formats: list[str] = Field(
        default=["wav", "mp3", "ogg", "m4a", "flac", "aac"],
        description="Allowed audio formats",
    )

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    media_dir: str | Path = Field(
        default="media",
        description="Media files directory (contains audio/ and transcripts/)",
    )
    audio_dir: str | Path = Field(
        default="media/audio", description="Audio files directory"
    )
    transcript_dir: str | Path = Field(
        default="media/transcripts", description="Transcript output directory"
    )
    model_cache_dir: str | Path = Field(
        default="model-cache", description="Model cache directory"
    )

    # Whisper Configuration
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base", description="Whisper model size"
    )
    whisper_backend: Literal["openai", "transformers"] = Field(
        default="openai", description="Whisper backend"
    )
    whisper_device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Whisper device"
    )

    # Whisper Constants
    WHISPER_BACKEND_DEFAULT: str = Field(
        default="openai", description="Default Whisper backend"
    )

    # Features
    enable_translation: bool = Field(default=False, description="Enable translation")
    enable_diarization: bool = Field(
        default=False, description="Enable speaker diarization"
    )

    # Diarization
    diarize_threshold: float = Field(default=0.35, description="Diarization threshold")
    max_speakers: int | None = Field(
        default=None, description="Maximum number of speakers"
    )
    use_silhouette: bool = Field(default=False, description="Use silhouette analysis")

    # Diarization Constants
    DIARIZE_SAMPLE_RATE: int = Field(
        default=16000, description="Sample rate for diarization"
    )
    MIN_CHUNK_MS: int = Field(default=500, description="Minimum chunk duration in ms")
    MIN_SEGMENT_MS: int = Field(
        default=1000, description="Minimum segment duration in ms"
    )
    SMOOTHING_MAX_DURATION_S: float = Field(
        default=2.0, description="Max duration for smoothing in seconds"
    )
    SUBSEGMENT_MIN_DURATION_S: float = Field(
        default=3.0, description="Min duration to use subsegments in seconds"
    )
    SUBSEGMENT_WINDOW_S: float = Field(
        default=1.5, description="Subsegment window size in seconds"
    )
    SUBSEGMENT_STRIDE_S: float = Field(
        default=0.5, description="Subsegment stride in seconds"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_file: str | None = Field(default=None, description="Log file path")

    # CORS
    cors_origins: str | list[str] = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, list):
            return v

        # v is a string at this point
        if v.startswith("["):
            # JSON-like format: ["http://...", "http://..."]
            import json

            parsed: list[str] = json.loads(v)
            return parsed
        else:
            # Comma-separated format
            return [origin.strip() for origin in v.split(",")]

    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() in ("development", "dev")

    @property
    def is_prod(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() in ("production", "prod")

    @property
    def is_test(self) -> bool:
        """Check if running in testing mode."""
        return self.environment.lower() in ("testing", "test")

    @field_validator("audio_dir", "transcript_dir", "model_cache_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v: str | Path) -> Path:
        """Resolve path strings to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("audio_dir", "transcript_dir", "model_cache_dir")
    @classmethod
    def make_absolute(cls, v: Path, info) -> Path:
        """Make paths absolute relative to base directory."""
        if not v.is_absolute():
            base_dir = info.data.get("base_dir", Path.cwd())
            return base_dir / v if isinstance(base_dir, Path) else Path(base_dir) / v
        return v

    class Config:
        """Pydantic config."""

        validate_assignment = True


settings = Settings()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ensure_directories() -> None:
    """Ensure required directories exist."""
    directories = [
        settings.media_dir,
        settings.audio_dir,
        settings.transcript_dir,
        settings.model_cache_dir,
    ]

    for directory in directories:
        dir_path = directory if isinstance(directory, Path) else Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
try:
    ensure_directories()
except Exception as e:
    # If we're early in initialization, logger might not be available
    print(f"Warning: Could not ensure directories exist: {e}")
