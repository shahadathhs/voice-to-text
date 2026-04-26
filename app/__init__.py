"""Voice-to-Text Application Package."""

__version__ = "1.0.0"

# Main application
# CLI interface
from app.cli import main as cli_main

# Core transcription functionality
from app.core.config import settings
from app.main import app
from app.services.pipeline import transcribe
from app.utils import get_unique_filename, save_transcript

__all__ = [
    "__version__",
    "app",
    "cli_main",
    "get_unique_filename",
    "save_transcript",
    "settings",
    "transcribe",
]
