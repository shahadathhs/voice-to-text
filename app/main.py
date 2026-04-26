"""FastAPI application for Voice-to-Text service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import docs
from app.api.routes import router
from app.core.config import ensure_directories, settings
from app.core.logger import logger
from app.core.middleware import (
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    RequestContextMiddleware,
)
from app.services.transcriber import lifespan_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Ensure required directories exist
    ensure_directories()
    logger.info("Required directories ensured")

    # Initialize transcription service
    async with lifespan_manager():
        yield

    # Shutdown
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="""
## 🎙️ Voice-to-Text API

AI-powered voice transcription service using OpenAI Whisper model with support for multiple audio formats, translation, and speaker diarization.

### 🎯 Key Features

- **Multiple Audio Formats**: Support for WAV, MP3, OGG, M4A, FLAC, AAC
- **Translation**: Convert non-English audio to English
- **Speaker Diarization**: Identify and label different speakers automatically
- **Multiple Backends**: OpenAI Whisper or Hugging Face Transformers
- **Fully Local**: All processing happens on your machine - no data sent to cloud

### 📚 Documentation Viewers

This API offers multiple documentation experiences:
- **RapiDoc** (⭐ Recommended) - Modern, responsive with dark theme
- **Swagger UI** - Classic interactive docs
- **ReDoc** - Beautiful reference documentation
- **Documentation Hub** - Choose your preferred viewer

Visit `/docs-hub` to compare and choose your documentation viewer!

### 🔒 Privacy

All audio processing is done locally. Your audio files never leave your server.

### 📖 Usage

1. Upload an audio file using the `/transcribe` endpoint
2. Optionally enable translation and/or speaker diarization
3. Receive transcribed text with metadata
4. Transcripts are automatically saved to `media/transcripts/`
""",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Voice-to-Text API",
        "url": "https://github.com/shahadathhs/voice-to-text",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add middleware (order matters!)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

app.include_router(docs.router)


# Global exception handlers (fallback if middleware doesn't catch)
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions with proper error response."""
    from app.core.response import ResponseBuilder

    logger.exception(f"Unhandled exception: {exc}")

    if settings.debug:
        # Detailed error in debug mode
        return ResponseBuilder.internal_server_error(
            message=str(exc),
            details={
                "exception": type(exc).__name__,
                "request_path": str(request.url),
            },
        ).model_dump()
    else:
        # Generic error in production
        return ResponseBuilder.internal_server_error(
            message="Internal server error"
        ).model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
