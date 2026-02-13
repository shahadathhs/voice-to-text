"""Voice-to-Text: local Whisper ASR + translation + speaker diarization (SpeechBrain)."""

from voice_to_text.config import WHISPER_BACKEND_DEFAULT
from voice_to_text.io_utils import get_unique_filename, save_transcript
from voice_to_text.pipeline import transcribe

__all__ = [
    "transcribe",
    "get_unique_filename",
    "save_transcript",
    "WHISPER_BACKEND_DEFAULT",
]
