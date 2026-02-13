#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
from datetime import datetime
from pathlib import Path

# Defer heavy imports to avoid segfaults during initial load
import torch
import whisper

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

# SpeechBrain ECAPA expects 16 kHz mono; minimum duration for reliable embedding (ms)
DIARIZE_SAMPLE_RATE = 16000
MIN_SEGMENT_MS = 500

def perform_diarization(audio_path, segments, device, classifier=None):
    """
    Non-gated diarization using SpeechBrain embeddings and clustering.
    Audio is resampled to 16 kHz for the encoder; clustering uses cosine distance.
    """
    print("[*] Extracting speaker embeddings for diarization...")
    try:
        from pydub import AudioSegment
        from sklearn.cluster import AgglomerativeClustering
        from speechbrain.inference.speaker import EncoderClassifier
        
        if classifier is None:
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain")
            )
    except Exception as e:
        print(f"[!] Error loading SpeechBrain model: {e}")
        return None

    audio = AudioSegment.from_file(audio_path)
    # Resample to 16 kHz mono for SpeechBrain ECAPA (required for good embeddings)
    if audio.frame_rate != DIARIZE_SAMPLE_RATE:
        audio = audio.set_frame_rate(DIARIZE_SAMPLE_RATE)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    embeddings = []
    valid_indices = []  # index into segments for each embedding

    for i, seg in enumerate(segments):
        start_ms = int(seg['start'] * 1000)
        end_ms = int(seg['end'] * 1000)
        seg_audio = audio[start_ms:end_ms]
        if len(seg_audio) < MIN_SEGMENT_MS:
            continue

        samples = np.array(seg_audio.get_array_of_samples()).astype(np.float32) / (2**15)
        signal = torch.from_numpy(samples).to(device)

        with torch.no_grad():
            emb = classifier.encode_batch(signal.unsqueeze(0))
            embeddings.append(emb.squeeze().cpu().numpy())
            valid_indices.append(i)

    if not embeddings:
        return None

    embeddings = np.array(embeddings)
    # Stricter threshold = more speaker separation. 0.25–0.4 typical; 0.8 merged everyone.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.35,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings)

    # Build label per segment index (invalid segments get previous speaker)
    seg_label = {}
    last_label = 0
    for idx, label in zip(valid_indices, cluster_labels):
        seg_label[idx] = label
        last_label = label
    for i in range(len(segments)):
        if i not in seg_label:
            seg_label[i] = seg_label.get(i - 1, 0)

    num_speakers = max(seg_label.values()) + 1
    diarization_map = []
    for i, seg in enumerate(segments):
        label = seg_label.get(i, 0)
        diarization_map.append({
            'start': seg['start'],
            'end': seg['end'],
            'speaker': f"SPEAKER_{label:02d}",
            'text': seg['text'].strip()
        })

    print(f"[*] Diarization: {num_speakers} speaker(s) across {len(segments)} segments")
    return diarization_map

def transcribe(audio_path, model="base", translate=False, diarize=False, device=None, classifier=None):
    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Using device: {device.upper()}")
    
    # Load model if it's a string name, otherwise use the passed object
    if isinstance(model, str):
        print(f"[*] Loading Whisper model '{model}'...")
        try:
            model = whisper.load_model(model, device=device)
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
            diarized_segments = perform_diarization(audio_path, orig_segments, device, classifier=classifier)
            if diarized_segments:
                orig_text = "\n".join([f"{s['speaker']}: {s['text']}" for s in diarized_segments])
            else:
                orig_text = "\n".join([seg['text'].strip() for seg in orig_segments])
        else:
            orig_text = "\n".join([seg['text'].strip() for seg in orig_segments])
        
        combined_output += "--- ORIGINAL TRANSCRIPT ---\n" + orig_text + "\n"
    except Exception as e:
        print(f"Error during transcription: {e}")
        # Only exit/raise if we are in CLI mode (inferred by model being a string loaded locally)
        # In server mode, we might want to raise the exception to be handled by the route
        if isinstance(model, str):
            sys.exit(1)
        raise e

    # 2. Optional Translation
    if translate:
        print(f"[*] Running translation to English...")
        try:
            trans_result = model.transcribe(audio_path, task="translate", verbose=False)
            trans_segments = trans_result.get('segments', [])
            
            if diarize:
                # Reuse diarization logic for translated segments
                # Note: Whisper produces different segments for translation, so we re-align
                diarized_trans = perform_diarization(audio_path, trans_segments, device, classifier=classifier)
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
    transcript = transcribe(audio_file, model=args.model, translate=args.translate, diarize=args.diarize)
    
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
