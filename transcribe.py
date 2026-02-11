#!/usr/bin/env python3
import argparse
import sys
import torch
import whisper
import os

def check_file(path):
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    return path

def transcribe(audio_path, model_name="base"):
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
    
    # Transcribe
    print(f"[*] Transcribing '{audio_path}'...")
    try:
        # verbose=False to keep terminal output clean, we will print the result manually
        result = model.transcribe(audio_path, verbose=False)
        return result['text'].strip()
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Standalone Voice-to-Text using Whisper.")
    parser.add_argument("input", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large). Default: base")
    
    args = parser.parse_args()
    
    audio_file = check_file(args.input)
    transcript = transcribe(audio_file, args.model)
    
    print("\n" + "="*20 + " TRANSCRIPT " + "="*20)
    print(transcript)
    print("="*52 + "\n")

if __name__ == "__main__":
    main()
