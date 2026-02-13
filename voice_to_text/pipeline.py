"""Main transcription pipeline: Whisper (openai or HF) + diarization and translation."""

from typing import Any

import torch

from voice_to_text.config import WHISPER_BACKEND_DEFAULT
from voice_to_text.diarization import assign_speaker_by_overlap, perform_diarization
from voice_to_text.backends import (
    load_openai_whisper,
    load_transformers_whisper,
    transcribe_openai,
    transcribe_transformers,
)


def _run_whisper(
    model_or_pipeline: Any,
    audio_path: str,
    task: str,
    whisper_backend: str,
) -> list[dict[str, Any]]:
    """Return segments from the chosen Whisper backend."""
    if whisper_backend == "transformers":
        return transcribe_transformers(model_or_pipeline, audio_path, task)
    return transcribe_openai(model_or_pipeline, audio_path, task)


def transcribe(
    audio_path: str,
    model: str | Any = "base",
    translate: bool = False,
    diarize: bool = False,
    device: str | None = None,
    classifier: Any = None,
    diarize_threshold: float = 0.35,
    max_speakers: int | None = None,
    whisper_backend: str = WHISPER_BACKEND_DEFAULT,
    use_silhouette: bool = False,
) -> str:
    """
    Transcribe audio to text. Supports translation to English and speaker diarization.
    All models run locally.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Using device: {device.upper()}")

    if isinstance(model, str):
        model_size = model
        print(f"[*] Loading Whisper model '{model_size}' (backend: {whisper_backend})...")
        try:
            if whisper_backend == "transformers":
                model = load_transformers_whisper(model_size, device)
            else:
                model = load_openai_whisper(model_size, device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    combined_output = ""
    diarized_orig = None

    print(f"[*] Running transcription (original) on '{audio_path}'...")
    try:
        orig_segments = _run_whisper(model, audio_path, "transcribe", whisper_backend)
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

    if diarize:
        diarized_orig = perform_diarization(
            audio_path,
            orig_segments,
            device,
            classifier=classifier,
            distance_threshold=diarize_threshold,
            max_speakers=max_speakers,
            use_silhouette=use_silhouette,
        )
        if diarized_orig:
            orig_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in diarized_orig])
        else:
            orig_text = "\n".join([s["text"].strip() for s in orig_segments])
    else:
        orig_text = "\n".join([s["text"].strip() for s in orig_segments])

    combined_output += "--- ORIGINAL TRANSCRIPT ---\n" + orig_text + "\n"

    if translate:
        print("[*] Running translation to English...")
        try:
            trans_segments = _run_whisper(model, audio_path, "translate", whisper_backend)
            if diarize and diarized_orig:
                trans_lines = []
                for seg in trans_segments:
                    speaker = assign_speaker_by_overlap(
                        seg["start"], seg["end"], diarized_orig
                    )
                    trans_lines.append(f"{speaker}: {seg['text'].strip()}")
                trans_text = "\n".join(trans_lines)
            else:
                trans_text = "\n".join([s["text"].strip() for s in trans_segments])
            combined_output += "\n--- ENGLISH TRANSLATION ---\n" + trans_text + "\n"
        except Exception as e:
            print(f"Error during translation: {e}")

    return combined_output.strip()
