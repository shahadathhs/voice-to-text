"""Whisper backends: openai-whisper and Hugging Face Transformers. Both run locally."""

from voice_to_text.backends.openai_whisper import (
    load_openai_whisper,
    transcribe_openai,
)
from voice_to_text.backends.transformers_whisper import (
    load_transformers_whisper,
    transcribe_transformers,
)

__all__ = [
    "load_openai_whisper",
    "transcribe_openai",
    "load_transformers_whisper",
    "transcribe_transformers",
]
