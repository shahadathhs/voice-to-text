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

def align_segments_with_speakers(segments, diarization):
    aligned_transcript = []
    
    for segment in segments:
        s_start = segment['start']
        s_end = segment['end']
        s_text = segment['text'].strip()
        
        # Find the speaker who was talking the most during this segment
        speaker_overlaps = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Calculate overlap
            overlap_start = max(s_start, turn.start)
            overlap_end = min(s_end, turn.end)
            
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                speaker_overlaps[speaker] = speaker_overlaps.get(speaker, 0) + overlap_duration
        
        if speaker_overlaps:
            # Pick the speaker with the most overlap
            best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
            aligned_transcript.append(f"{best_speaker}: {s_text}")
        else:
            aligned_transcript.append(f"Unknown: {s_text}")
            
    return "\n".join(aligned_transcript)

def transcribe(audio_path, model_name="base", translate=False, diarize=False, hf_token=None):
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using device: {device.upper()}")
    
    # Load token from env if not provided
    token = hf_token or os.environ.get("HF_TOKEN")
    
    # Load model
    print(f"[*] Loading Whisper model '{model_name}'...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # 1. Diarization (Optional) - Do this first to align later
    diarization_data = None
    if diarize:
        if not token:
            print("[!] Warning: Speaker diarization requested but no Hugging Face token found in args or environment.")
            print("[!] Skipping diarization...")
        else:
            print("[*] Running speaker diarization...")
            try:
                from pyannote.audio import Pipeline
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token
                )
                if device == "cuda":
                    pipeline.to(torch.device("cuda"))
                
                diarization_data = pipeline(audio_path)
            except Exception as e:
                print(f"Error during diarization: {e}")
                print("[!] Ensure you have accepted the conditions for 'pyannote/speaker-diarization-3.1' and 'pyannote/segmentation-3.0' on Hugging Face.")

    combined_output = ""
    
    # 2. Original Transcription
    print(f"[*] Running transcription (original) on '{audio_path}'...")
    try:
        orig_result = model.transcribe(audio_path, task="transcribe", verbose=False)
        orig_segments = orig_result.get('segments', [])
        
        if diarize and diarization_data:
            orig_text = align_segments_with_speakers(orig_segments, diarization_data)
        elif orig_segments:
            orig_text = "\n".join([seg['text'].strip() for seg in orig_segments])
        else:
            orig_text = orig_result['text'].strip()
        
        combined_output += "--- ORIGINAL TRANSCRIPT ---\n" + orig_text + "\n"
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

    # 3. Optional Translation
    if translate:
        print(f"[*] Running translation to English...")
        try:
            trans_result = model.transcribe(audio_path, task="translate", verbose=False)
            trans_segments = trans_result.get('segments', [])
            
            if diarize and diarization_data:
                trans_text = align_segments_with_speakers(trans_segments, diarization_data)
            elif trans_segments:
                trans_text = "\n".join([seg['text'].strip() for seg in trans_segments])
            else:
                trans_text = trans_result['text'].strip()
            
            combined_output += "\n--- ENGLISH TRANSLATION ---\n" + trans_text + "\n"
        except Exception as e:
            print(f"Error during translation: {e}")

    return combined_output.strip()


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

