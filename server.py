"""FastAPI server for voice-to-text: load models once, transcribe via HTTP."""

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

models: dict = {}
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", WHISPER_BACKEND_DEFAULT)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": models.get("device", "unknown"),
        "whisper_backend": models.get("whisper_backend", WHISPER_BACKEND),
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
    temp_filename = f"temp_{file.filename}"
    temp_path = Path(temp_filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[*] Processing file: {temp_filename}")
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
        output_filename = get_unique_filename(temp_filename)
        saved_path = save_transcript(transcript_text, output_filename)
        return {"status": "success", "transcript": transcript_text, "saved_to": str(saved_path)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if temp_path.exists():
            os.remove(temp_path)
