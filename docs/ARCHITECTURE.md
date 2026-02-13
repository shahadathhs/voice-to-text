# Architecture

## Overview

Voice-to-Text is a **fully local** pipeline: audio → Whisper (ASR) → translation and speaker diarization (SpeechBrain). No API keys; models run on your machine after one-time download.

## Folder structure

```
voice-to-text/
├── voice_to_text/           # Main package
│   ├── __init__.py          # Public API: transcribe, save_transcript, get_unique_filename
│   ├── config.py            # Constants (backends, diarization params)
│   ├── io_utils.py          # File I/O, unique filenames, save transcript
│   ├── diarization.py       # Speaker diarization (SpeechBrain, sub-segments, smoothing)
│   ├── pipeline.py          # Orchestration: load Whisper → transcribe → diarize/translate
│   ├── cli.py               # argparse and CLI entry
│   └── backends/
│       ├── __init__.py
│       ├── openai_whisper.py    # openai-whisper package
│       └── transformers_whisper.py  # Hugging Face pipeline
├── transcribe.py            # CLI entrypoint (calls voice_to_text.cli.main)
├── server.py                # FastAPI app (loads models once, POST /transcribe)
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   └── PROJECT_STRUCTURE.md
├── transcripts/             # Output directory (created on first run)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Data flow

1. **Input:** Audio file path (WAV, MP3, etc.).
2. **Whisper:** Load model (openai-whisper or HF Transformers) → `transcribe` task → segments `[{start, end, text}]`.
3. **Diarization:** For each segment (or sub-segments for long segments), extract SpeechBrain ECAPA embedding → cluster → assign SPEAKER_XX per segment → temporal smoothing.
4. **Translation:** Same Whisper run with `translate` task; speaker labels from original diarization (time overlap).
5. **Output:** Combined text (original + translation when enabled) saved under `transcripts/`.

## Backends

- **openai-whisper:** Default. Uses the `whisper` package; same segment format.
- **transformers:** Hugging Face `automatic-speech-recognition` pipeline with `openai/whisper-*`; segments normalized to the same format.

Both run locally; no data sent to the cloud.

## Diarization

- **Sub-segments:** Segments longer than ~3 s are split into sliding windows (1.5 s, 0.5 s stride); one embedding per window; segment label = majority vote of its windows.
- **Short segments:** Label from temporally nearest embedded chunk.
- **Smoothing:** Segments &lt; 2 s that differ from both neighbors are flipped to the previous label.
- **Clustering:** AgglomerativeClustering (cosine); fixed threshold or `max_speakers` or silhouette-based k.

## Running

- **CLI:** `python transcribe.py audio.mp3 [--diarize] [--translate] [--whisper-backend ...]`
- **Server:** `uvicorn server:app` (or `make server`). Models loaded at startup; `POST /transcribe` with file upload.

All execution is local (CPU/GPU on your machine).
