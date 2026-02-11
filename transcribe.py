#!/usr/bin/env python3
import argparse
import sys
import torch
import whisper
import os
from datetime import datetime
from pathlib import Path

def check_file(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    return path

def get_unique_filename(audio_path):
    base_name = Path(audio_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.txt"

def save_transcript(text, filename):
    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def transcribe(audio_path, model_name="base", translate=False, diarize=False, hf_token=None):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device.upper()}")
    
    # Load model
    print(f"[*] Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Transcribe/Translate
    task = "translate" if translate else "transcribe"
    print(f"[*] Running {task} on '{audio_path}'...")
    try:
        result = model.transcribe(audio_path, task=task, verbose=False)
        # Add line breaks between segments for better readability
        segments = result.get('segments', [])
        if segments:
            transcript_text = "\n".join([seg['text'].strip() for seg in segments])
        else:
            transcript_text = result['text'].strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


    # Diarization
    if diarize:
        if not hf_token:
            print("[!] Warning: Speaker diarization requested but no Hugging Face token provided.")
            print("[!] Skipping diarization...")
        else:
            print("[*] Running speaker diarization...")
            try:
                from pyannote.audio import Pipeline
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                if device == "cuda":
                    pipeline.to(torch.device("cuda"))
                
                diarization = pipeline(audio_path)
                
                # Combine transcription with diarization (Simplified version)
                # Note: For professional results, we should align whisper segments with diarization turns.
                # Here we'll just print the diarization map for now.
                diarization_output = "\n\n--- Speaker Timeline ---\n"
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    diarization_output += f"[{turn.start:0.1f}s - {turn.end:0.1f}s] {speaker}\n"
                transcript_text += diarization_output
            except Exception as e:
                print(f"Error during diarization: {e}")
                print("[!] Ensure you have accepted the conditions for 'pyannote/speaker-diarization-3.1' on Hugging Face.")

    return transcript_text

def main():
    parser = argparse.ArgumentParser(description="Advanced Voice-to-Text using Whisper.")
    parser.add_argument("input", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large). Default: base")
    parser.add_argument("--translate", action="store_true", help="Translate non-English audio to English")
    parser.add_argument("--diarize", action="store_true", help="Perform speaker diarization (requires HF token)")
    parser.add_argument("--hf-token", help="Hugging Face token for diarization")
    
    args = parser.parse_args()
    
    audio_file = check_file(args.input)
    transcript = transcribe(audio_file, args.model, args.translate, args.diarize, args.hf_token)
    
    # Print to terminal
    print("\n" + "="*20 + " TRANSCRIPT " + "="*20)
    print(transcript)
    print("="*52 + "\n")
    
    # Save to file
    output_filename = get_unique_filename(audio_file)
    saved_path = save_transcript(transcript, output_filename)
    print(f"[*] Transcript saved to: {saved_path}")

if __name__ == "__main__":
    main()

