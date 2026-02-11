#!/usr/bin/env python3
print("[DEBUG] Starting transcribe.py...")
import argparse
import sys
print("[DEBUG] Importing os...")
import os
print("[DEBUG] Importing numpy...")
import numpy as np
print("[DEBUG] Importing datetime...")
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier

# Defer heavy imports to avoid segfaults during initial load
print("[DEBUG] Importing torch...")
import torch
print("[DEBUG] Importing whisper...")
import whisper

print("[DEBUG] Imports complete.")

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

def perform_diarization(audio_path, segments, device):
    """
    Non-gated diarization using SpeechBrain embeddings and Clustering.
    """
    print("[*] Extracting speaker embeddings for diarization...")
    try:
        from pydub import AudioSegment
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier
        
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
        )
    except Exception as e:
        print(f"[!] Error loading SpeechBrain model: {e}")
        return None

    audio = AudioSegment.from_file(audio_path)
    embeddings = []
    valid_segments = []

    for seg in segments:
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        
        # Extract segment audio
        seg_audio = audio[start_ms:end_ms]
        if len(seg_audio) < 100: # Skip very short segments
            continue
            
        # Convert pydub audio to torch tensor
        samples = np.array(seg_audio.get_array_of_samples()).astype(np.float32)
        # Normalize
        samples = samples / (2**15)
        if seg_audio.channels > 1:
            samples = samples.reshape((-1, seg_audio.channels)).mean(axis=1)
            
        signal = torch.from_numpy(samples).to(device)
        
        with torch.no_grad():
            emb = classifier.encode_batch(signal.unsqueeze(0))
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_segments.append(seg)

    if not embeddings:
        return None

    # Cluster embeddings
    embeddings = np.array(embeddings)
    # Simple heuristic: if we have few segments, don't over-cluster
    num_clusters = min(len(embeddings), 5) # Cap at 5 for small files
    
    # Use Agglomerative Clustering (Standard for cold-start diarization)
    # We use 'cosine' distance as it's best for embeddings
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.8, # Adjust for sensitivity
        metric='cosine', 
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    # Map labels to segments
    diarization_map = []
    for seg, label in zip(valid_segments, labels):
        diarization_map.append({
            'start': seg['start'],
            'end': seg['end'],
            'speaker': f"SPEAKER_{label:02d}",
            'text': seg['text'].strip()
        })
    
    return diarization_map

def transcribe(audio_path, model_name="base", translate=False, diarize=False):
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
    
    combined_output = ""
    
    # 1. Original Transcription (Always needed for base)
    print(f"[*] Running transcription (original) on '{audio_path}'...")
    try:
        orig_result = model.transcribe(audio_path, task="transcribe", verbose=False)
        orig_segments = orig_result.get('segments', [])
        
        if diarize:
            diarized_segments = perform_diarization(audio_path, orig_segments, device)
            if diarized_segments:
                orig_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in diarized_segments])
            else:
                orig_text = "\n".join([seg['text'].strip() for seg in orig_segments])
        else:
            orig_text = "\n".join([seg['text'].strip() for seg in orig_segments])
        
        combined_output += "--- ORIGINAL TRANSCRIPT ---\n" + orig_text + "\n"
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

    # 2. Optional Translation
    if translate:
        print(f"[*] Running translation to English...")
        try:
            trans_result = model.transcribe(audio_path, task="translate", verbose=False)
            trans_segments = trans_result.get('segments', [])
            
            if diarize:
                # Reuse diarization logic for translated segments
                # Note: Whisper produces different segments for translation, so we re-align
                diarized_trans = perform_diarization(audio_path, trans_segments, device)
                if diarized_trans:
                    trans_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in diarized_trans])
                else:
                    trans_text = "\n".join([seg['text'].strip() for seg in trans_segments])
            else:
                trans_text = "\n".join([seg['text'].strip() for seg in trans_segments])
            
            combined_output += "\n--- ENGLISH TRANSLATION ---\n" + trans_text + "\n"
        except Exception as e:
            print(f"Error during translation: {e}")

    return combined_output.strip()

def main():
    parser = argparse.ArgumentParser(description="Standalone Voice-to-Text using Whisper & SpeechBrain.")
    parser.add_argument("input", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large). Default: base")
    parser.add_argument("--translate", action="store_true", help="Translate non-English audio to English")
    parser.add_argument("--diarize", action="store_true", help="Perform speaker diarization (No token required)")
    
    args = parser.parse_args()
    
    audio_file = check_file(args.input)
    transcript = transcribe(audio_file, args.model, args.translate, args.diarize)
    
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

