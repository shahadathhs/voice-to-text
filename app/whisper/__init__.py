"""Whisper implementations: openai-whisper and Hugging Face Transformers. Both run locally."""

from app.whisper.openai_whisper import (
    load_openai_whisper,
    transcribe_openai,
)
from app.whisper.transformers_whisper import (
    load_transformers_whisper,
    transcribe_transformers,
)

__all__ = [
    "load_openai_whisper",
    "load_transformers_whisper",
    "transcribe_openai",
    "transcribe_transformers",
]
