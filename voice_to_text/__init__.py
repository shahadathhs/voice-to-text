"""Voice-to-Text: local Whisper ASR + translation + speaker diarization (SpeechBrain).

This package provides both CLI and FastAPI interfaces for voice transcription.
"""

__version__ = "1.0.0"

# Core transcription functionality
from voice_to_text.config import WHISPER_BACKEND_DEFAULT
from voice_to_text.io_utils import get_unique_filename, save_transcript
from voice_to_text.pipeline import transcribe

# CLI interface
from voice_to_text.cli import main as cli_main

# API interface
from voice_to_text.api.routes import app as api_app

__all__ = [
    # Core
    "transcribe",
    "get_unique_filename",
    "save_transcript",
    "WHISPER_BACKEND_DEFAULT",
    # CLI
    "cli_main",
    # API
    "api_app",
]
