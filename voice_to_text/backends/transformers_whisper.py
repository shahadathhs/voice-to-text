"""Hugging Face Transformers Whisper backend. Runs 100% locally after model download."""

from __future__ import annotations

from typing import Any

HF_WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


def load_transformers_whisper(model_size: str, device: str) -> Any:
    """Load HF automatic-speech-recognition pipeline with Whisper."""
    from transformers import pipeline
    model_id = HF_WHISPER_MODELS.get(model_size, f"openai/whisper-{model_size}")
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=0 if device == "cuda" else -1,
        return_timestamps="segment",
    )


def transcribe_transformers(
    pipeline_or_model: Any,
    audio_path: str,
    task: str,
) -> list[dict[str, Any]]:
    """Run transcription. task: 'transcribe' | 'translate'."""
    out = pipeline_or_model(
        audio_path,
        return_timestamps="segment",
        generate_kwargs={"task": task},
    )
    segments = []
    if isinstance(out, dict) and "chunks" in out:
        for ch in out["chunks"]:
            ts = ch.get("timestamp")
            if ts is not None and isinstance(ts, (tuple, list)) and len(ts) >= 2:
                s, e = float(ts[0]), float(ts[1])
            else:
                s, e = 0.0, 0.0
            text = (ch.get("text") or "").strip()
            segments.append({"start": s, "end": e, "text": text})
    elif isinstance(out, dict) and "text" in out:
        segments.append({"start": 0.0, "end": 0.0, "text": (out["text"] or "").strip()})
    return segments
