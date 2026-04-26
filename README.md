# Voice-to-Text: CLI + FastAPI Server

A standalone, **fully local** voice-to-text system using OpenAI's Whisper (open-source). All models run on your machine—no API keys, no cloud calls; audio never leaves your device. This version supports **Dual Output (Original + Translation)** and **Local-First Speaker Diarization** (no tokens required).

**Features:**
- 🖥️ **CLI Interface**: Command-line tool for local transcription
- 🌐 **FastAPI Server**: REST API for transcription services
- 🔄 **Dual Mode**: Use as CLI or deploy as API server
- 🌍 **Translation**: Translate to English with original transcript
- 👥 **Speaker Diarization**: Identify and label different speakers
- 🐳 **Docker Support**: Easy containerization

## Prerequisites

- **Docker** and **Docker Compose** installed (for containerized usage)
- **FFmpeg** (only if running locally without Docker)
- **Make** (installed by default on most Linux systems)
- **UV** (Python package manager - 10-100x faster than pip)

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

## 🛠️ Local Installation

### Using UV (Recommended - 10-100x faster)

1. **Install FFmpeg**:
   ```bash
   sudo apt update && sudo apt install ffmpeg libsndfile1
   ```

2. **Install UV**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Setup and Install**:
   ```bash
   make setup
   # Or manually:
   uv venv
   uv sync
   ```

4. **Run CLI**:
   ```bash
   uv run python transcribe.py audio.wav --translate --diarize
   ```

5. **Run API Server**:
   ```bash
   make dev
   # Or manually:
   uv run python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

### Using pip (Legacy - Deprecated)

1. **Install FFmpeg**: `sudo apt update && sudo apt install ffmpeg libsndfile1`
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
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

## 📁 Project Structure

```
voice-to-text/
├── voice_to_text/          # Main package
│   ├── __init__.py        # Package initialization
│   ├── cli.py             # CLI interface
│   ├── config.py          # Configuration
│   ├── diarization.py     # Speaker diarization
│   ├── io_utils.py        # File I/O utilities
│   ├── pipeline.py        # Transcription pipeline
│   └── backends/          # Whisper backends (openai, transformers)
├── transcribe.py          # CLI entrypoint
├── server.py              # FastAPI server
├── .claude/               # Development guide
│   └── CLAUDE.md          # AI development documentation
├── pyproject.toml         # Project configuration (UV)
├── Makefile               # Automation commands
├── compose.yaml           # Docker Compose configuration
└── Dockerfile             # Container image
```

- **CLI Mode**: Run `transcribe.py` for local transcription
- **API Mode**: Run `server.py` for REST API service
- **Docker Mode**: Use `make server` for containerized API

See **[`.claude/CLAUDE.md`](.claude/CLAUDE.md)** for comprehensive development guide.

---

## 🧹 Code Quality

We use **Ruff** for linting and formatting, **Black** for code formatting, **MyPy** for type checking, and **Bandit** for security scanning.

### Check for issues

```bash
make lint           # Ruff linting
make type-check     # MyPy type checking
make format         # Black formatting
make check-all      # Run all checks
make fix-all        # Auto-fix all issues
```

### Run full CI locally

```bash
make ci             # Run pre-commit hooks, security scan, and build
```

---

## 🚀 Development

### Setup Development Environment

```bash
# Full setup (recommended)
make setup

# Install pre-commit hooks
make pre-commit-install

# Start development server with hot reload
make dev
```

### Available Commands

```bash
make help           # Show all available commands
```

---

## 📚 Documentation

- **[`.claude/CLAUDE.md`](.claude/CLAUDE.md)** - Comprehensive development guide
- **[`docs/`](docs/)** - Additional documentation
- **API Documentation** - Available at http://localhost:8000/docs when server is running
