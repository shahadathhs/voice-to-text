"""FastAPI routes for voice-to-text API."""

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import File, FastAPI, UploadFile
from speechbrain.inference.speaker import EncoderClassifier

from voice_to_text import (
    WHISPER_BACKEND_DEFAULT,
    get_unique_filename,
    save_transcript,
    transcribe,
)

# Global models dictionary
models: dict = {}
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", WHISPER_BACKEND_DEFAULT)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load models on startup."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Server starting. Loading Whisper ({WHISPER_BACKEND}, {WHISPER_MODEL_SIZE}) on {device.upper()}...")

    if WHISPER_BACKEND == "transformers":
        from voice_to_text.backends import load_transformers_whisper
        models["whisper"] = load_transformers_whisper(WHISPER_MODEL_SIZE, device)
    else:
        from voice_to_text.backends import load_openai_whisper
        models["whisper"] = load_openai_whisper(WHISPER_MODEL_SIZE, device)

    models["device"] = device
    models["whisper_backend"] = WHISPER_BACKEND

    print("[*] Loading SpeechBrain classifier...")
    models["classifier"] = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain"),
    )
    print("[*] Models loaded successfully!")

    yield

    # Cleanup
    models.clear()


# Create FastAPI app
app = FastAPI(
    title="Voice-to-Text API",
    description="AI-powered voice transcription service using OpenAI Whisper",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Voice-to-Text API",
        "version": "1.0.0",
        "description": "AI-powered voice transcription service",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe",
            "docs": "/docs",
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": models.get("device", "unknown"),
        "whisper_backend": models.get("whisper_backend", WHISPER_BACKEND),
        "model_size": WHISPER_MODEL_SIZE,
    }


@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    translate: bool = False,
    diarize: bool = False,
    diarize_threshold: float = 0.35,
    max_speakers: int | None = None,
    use_silhouette: bool = False,
):
    """
    Transcribe an audio file.

    Args:
        file: Audio file to transcribe
        translate: Translate to English
        diarize: Enable speaker diarization
        diarize_threshold: Clustering distance for diarization
        max_speakers: Fixed number of speakers
        use_silhouette: Estimate speakers from embeddings

    Returns:
        Transcription result with text and metadata
    """
    temp_filename = f"temp_{file.filename}"
    temp_path = Path(temp_filename)

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[*] Processing file: {temp_filename}")

        # Transcribe
        transcript_text = transcribe(
            str(temp_path),
            model=models["whisper"],
            translate=translate,
            diarize=diarize,
            device=models["device"],
            classifier=models["classifier"],
            diarize_threshold=diarize_threshold,
            max_speakers=max_speakers,
            whisper_backend=models.get("whisper_backend", WHISPER_BACKEND),
            use_silhouette=use_silhouette,
        )

        # Save transcript
        output_filename = get_unique_filename(temp_filename)
        saved_path = save_transcript(transcript_text, output_filename)

        return {
            "status": "success",
            "data": {
                "transcript": transcript_text,
                "saved_to": str(saved_path),
                "metadata": {
                    "model": WHISPER_MODEL_SIZE,
                    "backend": models.get("whisper_backend", WHISPER_BACKEND),
                    "device": models.get("device", "unknown"),
                    "translated": translate,
                    "diarized": diarize,
                }
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None
        }
    finally:
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)
