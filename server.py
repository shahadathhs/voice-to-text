from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
import whisper
import torch
import os
import shutil
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from transcribe import transcribe, get_unique_filename, save_transcript

# Global models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Whisper Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Server starting. Loading Whisper model on {device.upper()}...")
    models["whisper"] = whisper.load_model("base", device=device)
    models["device"] = device
    
    # Load SpeechBrain Classifier
    print("[*] Loading SpeechBrain classifier...")
    models["classifier"] = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
    )
    print("[*] Models loaded successfully!")
    yield
    # Cleanup (not strictly necessary for this use case)
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "device": models.get("device", "unknown")}

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    translate: bool = False,
    diarize: bool = False
):
    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    temp_path = Path(temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[*] Processing file: {temp_filename}")
        
        # Run transcription with pre-loaded models
        transcript_text = transcribe(
            str(temp_path),
            model=models["whisper"],
            translate=translate,
            diarize=diarize,
            device=models["device"],
            classifier=models["classifier"]
        )
        
        # Save persistence
        output_filename = get_unique_filename(temp_filename)
        saved_path = save_transcript(transcript_text, output_filename)
        
        return {
            "status": "success",
            "transcript": transcript_text,
            "saved_to": str(saved_path)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # Cleanup temp file
        if temp_path.exists():
            os.remove(temp_path)
