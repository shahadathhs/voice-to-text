"""Voice-to-Text Application Package."""

__version__ = "1.0.0"

# Main application
from app.main import app

# Core transcription functionality
from app.core.config import WHISPER_BACKEND_DEFAULT
from app.utils import get_unique_filename, save_transcript
from app.services.pipeline import transcribe

# CLI interface
from app.cli import main as cli_main

__all__ = [
    # Version
    "__version__",
    # FastAPI app
    "app",
    # Core
    "WHISPER_BACKEND_DEFAULT",
    "get_unique_filename",
    "save_transcript",
    "transcribe",
    # CLI
    "cli_main",
]
