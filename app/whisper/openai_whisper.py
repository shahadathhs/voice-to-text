"""OpenAI Whisper backend (openai-whisper package). Runs 100% locally."""

from __future__ import annotations

from typing import Any


def load_openai_whisper(model_size: str, device: str) -> Any:
    """Load OpenAI Whisper model."""
    import whisper
    return whisper.load_model(model_size, device=device)


def transcribe_openai(
    model: Any,
    audio_path: str,
    task: str,
) -> list[dict[str, Any]]:
    """Run transcription. task: 'transcribe' | 'translate'."""
    result = model.transcribe(audio_path, task=task, verbose=False)
    segments = result.get("segments", [])
    return [
        {"start": s["start"], "end": s["end"], "text": (s.get("text") or "").strip()}
        for s in segments
    ]
