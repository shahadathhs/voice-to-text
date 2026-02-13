# Simple Voice-to-Text System

A standalone, **fully local** voice-to-text system using OpenAI's Whisper (open-source). All models run on your machine—no API keys, no cloud calls; audio never leaves your device. This version supports **Dual Output (Original + Translation)** and **Local-First Speaker Diarization** (no tokens required).

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **FFmpeg** (only if running locally without Docker).
- **Make** (installed by default on most Linux systems).

---

## 🚀 Quick Start with Makefile (Easiest)

We provide a `Makefile` to simplify the Docker commands.

### 1. Build the Image

```bash
make build
```

### 2. Run Basic Transcription

```bash
make run AUDIO=audio/mix.mp3
```

### 3. Translate to English (Includes Original Transcript)

```bash
make translate AUDIO=audio/mix.mp3
```

### 4. Speaker Diarization

To identify different speakers and label the transcript (`SPEAKER_00: Hello`):

```bash
make diarize AUDIO=audio/mix.mp3
```

### 5. All-in-one (Transcribe + Translate + Diarize)

```bash
make all AUDIO=audio/mix.mp3
```

---

## 🌐 API Server (FastAPI)

For persistent model loading and instant transcription via HTTP:

### 1. Start the Server

```bash
make server
```

The server loads models **once** at startup. Subsequent requests are instant.

### 2. Access Swagger UI

```bash
make docs
```

Or visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Example API Request

```bash
curl -X POST "http://localhost:8000/transcribe?translate=true&diarize=true" \
  -F "file=@audio/multi_person.mp3"
```

Query params: `diarize_threshold`, `max_speakers`, `use_silhouette` (estimate speaker count from embeddings).

---

## 🐳 Quick Start with Docker (Manual)

If you don't have `make` installed:

### 1. Build

```bash
docker compose build
```

### 2. Run

```bash
# Basic
docker compose run --rm whisper audio.wav

# Dual Output (Original + translation)
docker compose run --rm whisper audio.wav --translate

# Diarization (No token required!)
docker compose run --rm whisper audio.wav --diarize
```

---

## 📂 Output

- Transcripts are saved to `transcripts/`.
- If diarization is enabled, output follows the format: `SPEAKER_XX: [Text segment]`
- If translation is enabled, files contain both the **Original language** and the **English translation**.

---

## 🛠️ Local Installation (Manual)

1. **Install FFmpeg**: `sudo apt update && sudo apt install ffmpeg libsndfile1`
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   python3 transcribe.py audio.wav --translate --diarize
   ```

---

## ⚙️ Options

- `--model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Default: `base`.
- `--whisper-backend`: `openai-whisper` (default) or `transformers` (Hugging Face). Both run locally.
- `--translate`: Translate non-English audio to English.
- `--diarize`: Enable speaker diarization (SpeechBrain + sliding-window sub-segments + temporal smoothing).
- `--diarize-threshold`: Clustering distance (lower = more speakers). Default: `0.35`. Ignored if `--max-speakers` is set.
- `--max-speakers`: Fix number of speakers (e.g. `2` for two-person dialogue). Overrides `--diarize-threshold`.
- `--use-silhouette`: Estimate number of speakers from embeddings when `--max-speakers` is not set.

**Server:** Set `WHISPER_BACKEND=transformers` or `WHISPER_MODEL=small` in the environment to change the loaded model.

---

## 📁 Project structure

- **`voice_to_text/`** — Main package: `config`, `io_utils`, `diarization`, `pipeline`, `cli`, `backends/` (openai + transformers Whisper).
- **`transcribe.py`** — CLI entrypoint.
- **`server.py`** — FastAPI server.
- **`docs/`** — Architecture and structure (see `docs/ARCHITECTURE.md`, `docs/PROJECT_STRUCTURE.md`).

See **`docs/ARCHITECTURE.md`** for data flow and **`docs/PROJECT_STRUCTURE.md`** for a file-by-file reference.

---

## 🧹 Code Quality

We use **Ruff** for linting and formatting.

### Check for issues

```bash
make lint
```

### Auto-format code

```bash
make format
```
