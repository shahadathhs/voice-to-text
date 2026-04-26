# Voice-to-Text: CLI + FastAPI Server

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Server](#-api-server)
- [Configuration](#️-configuration)
- [Output Format](#-output-format)
- [Code Quality](#-code-quality)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Roadmap](#-roadmap)

---

## Overview

**Voice-to-Text** is a **fully local** AI-powered voice transcription system using OpenAI's Whisper (open-source). All models run on your machine—no API keys, no cloud calls, audio never leaves your device. Supports **translation**, **speaker diarization**, and multiple **Whisper backends**.

---

## ✨ Features

- 🖥️ **CLI Interface**: Command-line tool for local transcription
- 🌐 **FastAPI Server**: REST API with persistent model loading
- 🔄 **Dual Mode**: Use as standalone CLI or deploy as API server
- 🌍 **Translation**: Translate non-English audio to English (includes original)
- 👥 **Speaker Diarization**: Identify and label different speakers (token-free)
- 🔧 **Multiple Backends**: OpenAI Whisper or Hugging Face Transformers
- 🐳 **Docker Support**: Easy containerization and deployment
- 📊 **Smart Clustering**: Sub-segments, temporal smoothing, silhouette analysis
- 📁 **Unified Media Storage**: Organized media folder structure for audio and transcripts
- 🔒 **Fully Local**: No data sent to cloud, complete privacy

## 🏗️ Architecture

### Overview

```
Audio Input → Whisper (ASR) → Segments → Diarization/Translation → Output
                                    ↓
                            SpeechBrain ECAPA
                            (speaker embeddings)
```

### Data Flow

1. **Input**: Audio file (WAV, MP3, OGG, M4A, FLAC, AAC)
2. **Whisper**: Load model → transcribe task → segments `[{start, end, text}]`
3. **Diarization**: Extract SpeechBrain ECAPA embeddings → cluster → assign `SPEAKER_XX` → temporal smoothing
4. **Translation**: Run Whisper with `translate` task → map speakers via time overlap
5. **Output**: Combined text saved to `media/transcripts/`

### Backends

- **OpenAI Whisper** (default): Uses the `whisper` package
- **Transformers**: Hugging Face `automatic-speech-recognition` pipeline with `openai/whisper-*`

Both run locally with identical segment format. Switch via `--whisper-backend` or `WHISPER_BACKEND` env var.

### Diarization Algorithm

- **Sub-segments**: Long segments (>3s) split into sliding windows (1.5s window, 0.5s stride)
- **Embeddings**: One SpeechBrain ECAPA embedding per window
- **Segment label**: Majority vote of its windows
- **Short segments**: Label from temporally nearest embedded chunk
- **Smoothing**: Segments <2s that differ from both neighbors are flipped to previous label
- **Clustering**: AgglomerativeClustering (cosine distance)

### Technology Stack

```python
# Core Dependencies
fastapi>=0.115.0          # Web framework
uvicorn[standard]>=0.32.0 # ASGI server
pydantic>=2.0            # Data validation
pydantic-settings>=2.0   # Configuration management
openai-whisper           # Transcription model
transformers>=4.40.0     # Hugging Face transformers
speechbrain              # Speaker diarization
loguru>=0.7.0            # Logging

# Development Tools
uv                        # Package manager (10-100x faster)
ruff==0.15.11            # Linting and formatting
black==26.3.1             # Code formatting
mypy==1.20.1             # Type checking
bandit>=1.9.4            # Security scanning
pre-commit>=4.5.1        # Git hooks
```

## 📁 Project Structure

```
voice-to-text/
├── app/                       # Main application package
│   ├── __init__.py           # Package exports
│   ├── main.py               # FastAPI application
│   ├── api/                  # REST API layer
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints with Swagger docs
│   │   ├── services.py      # API business logic
│   │   └── docs.py          # API documentation helpers
│   ├── cli/                  # CLI interface
│   │   ├── __init__.py
│   │   └── main.py          # CLI entry point
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration (Pydantic Settings)
│   │   ├── errors.py        # Custom exceptions
│   │   ├── logger.py        # Logging setup
│   │   └── response.py      # ResponseBuilder for standardized responses
│   ├── schemas/              # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── base.py          # Base response schemas
│   │   └── transcription.py # Transcription-specific schemas
│   ├── services/             # Business logic layer
│   │   ├── __init__.py
│   │   ├── diarization.py   # Speaker diarization
│   │   ├── pipeline.py      # Transcription orchestration
│   │   └── transcriber.py   # Transcription service
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── io_utils.py      # File I/O operations
│   └── whisper/              # Whisper implementations
│       ├── __init__.py
│       ├── openai_whisper.py    # OpenAI backend
│       └── transformers_whisper.py # HuggingFace backend
├── media/                     # Unified media directory
│   ├── audio/                 # Input audio files
│   └── transcripts/           # Output transcripts (auto-created)
├── cli.py                     # CLI entry point
├── server.py                  # Server entry point
├── pyproject.toml            # UV package manager config
├── Makefile                   # Automation commands
├── compose.yaml               # Docker Compose setup
├── Dockerfile                 # Production container image
└── .claude/                   # Development guide
    └── CLAUDE.md             # AI development documentation
```

### Module Responsibilities

| Layer | Purpose |
|-------|---------|
| **Core** | Configuration, errors, logging, response builders |
| **Schemas** | Pydantic models for request/response validation |
| **Services** | Business logic, orchestration, diarization |
| **Utils** | File I/O, helper functions |
| **Whisper** | Model implementations, backends |
| **API** | HTTP endpoints, request handling |
| **CLI** | Command-line interface, argument parsing |

---

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **Make** (installed by default on most systems)
- **FFmpeg** (only for local installation without Docker)
- **UV** (Python package manager - 10-100x faster than pip)

### Option 1: Using Makefile (Easiest)

#### Build the Image
```bash
make build
```

#### Basic Transcription
```bash
make run AUDIO=media/audio/mix.mp3
```

#### Translate to English
```bash
make translate AUDIO=media/audio/mix.mp3
```

#### Speaker Diarization
```bash
make diarize AUDIO=media/audio/multi_person.mp3
```

#### All Features Combined
```bash
make all AUDIO=media/audio/multi_person.mp3
```

### Option 2: Local Installation

#### Using UV (Recommended - 10-100x faster)

```bash
# 1. Install FFmpeg
sudo apt update && sudo apt install ffmpeg libsndfile1

# 2. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Setup and install
make setup
# Or manually:
uv venv
uv sync

# 4. Run CLI
uv run python cli.py audio.wav --translate --diarize

# 5. Run API server
make dev
# Or manually:
uv run python -m uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

---

## 🌐 API Server

### Start the Server

```bash
make server
```

The server loads models **once** at startup. Subsequent requests are instant.

### Access API Documentation

```bash
make docs
```

Or visit: **http://localhost:8000/docs**

### Interactive API Features

- **Swagger UI**: Try out endpoints directly from your browser at `/docs`
- **ReDoc**: Alternative documentation at `/redoc`
- **Detailed Examples**: Each endpoint includes example requests/responses
- **Validation**: Automatic request validation with detailed error messages
- **Response Schemas**: Complete response format documentation

### Example API Request

```bash
# Basic transcription
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@media/audio/multi_person.mp3"

# With translation and diarization
curl -X POST "http://localhost:8000/transcribe?translate=true&diarize=true" \
  -F "file=@media/audio/multi_person.mp3"

# Advanced options
curl -X POST "http://localhost:8000/transcribe?diarize=true&max_speakers=2" \
  -F "file=@media/audio/meeting.mp3"
```

### API Response Format

All API responses follow a standardized format:

```json
{
  "status_code": 200,
  "success": true,
  "message": "Transcription completed successfully",
  "transcript": "SPEAKER_00: Hello world\nSPEAKER_01: Hi there",
  "saved_to": "media/transcripts/audio_20260426_123456.txt",
  "metadata": {
    "model": "base",
    "backend": "openai",
    "device": "cpu",
    "translated": false,
    "diarized": true,
    "audio_file": "media/audio/sample.wav"
  }
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/transcribe` | POST | Transcribe audio file |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `translate` | boolean | `false` | Translate to English |
| `diarize` | boolean | `false` | Enable speaker diarization |
| `diarize_threshold` | float | `0.35` | Clustering distance (0.0-1.0) |
| `max_speakers` | int | `null` | Fixed number of speakers |
| `use_silhouette` | boolean | `false` | Estimate speakers from embeddings |

---

## ⚙️ Configuration

### CLI Options

```bash
python cli.py audio.wav [OPTIONS]

Options:
  --model {tiny,base,small,medium,large}
                          Whisper model size (default: base)
  --backend {openai,transformers}
                          Whisper backend (default: openai)
  --translate             Translate to English
  --diarize               Enable speaker diarization
  --diarize-threshold FLOAT
                          Clustering threshold 0.0-1.0 (default: 0.35)
  --max-speakers INT      Fixed number of speakers
  --use-silhouette        Estimate speakers from embeddings
  --output PATH           Output file path
  --verbose, -v           Enable verbose output
  --debug                 Enable debug mode
```

### Environment Variables

Set these in `.env` or export before running:

```bash
# Application
ENVIRONMENT=production      # development, production, testing
DEBUG=false                # Debug mode

# Server
HOST=0.0.0.0               # Server host
PORT=8000                  # Server port
WORKERS=1                  # Number of workers

# Whisper Configuration
WHISPER_MODEL=base         # tiny, base, small, medium, large
WHISPER_BACKEND=openai     # openai, transformers
WHISPER_DEVICE=cpu         # cpu, cuda

# Features
ENABLE_TRANSLATION=false   # Enable translation
ENABLE_DIARIZATION=false   # Enable diarization

# Diarization
DIARIZE_THRESHOLD=0.35     # Clustering threshold
MAX_SPEAKERS=null          # Max speakers (null = unlimited)
USE_SILHOUETTE=false       # Use silhouette analysis

# Logging
LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## 📂 Output Format

### Without Diarization
```
--- ORIGINAL TRANSCRIPT ---
This is the first segment.
This is the second segment.
```

### With Diarization
```
--- ORIGINAL TRANSCRIPT ---
SPEAKER_00: This is the first speaker.
SPEAKER_01: This is the second speaker.
SPEAKER_00: Back to the first speaker.
```

### With Translation
```
--- ORIGINAL TRANSCRIPT ---
SPEAKER_00: Bonjour, comment allez-vous?
SPEAKER_01: Très bien, merci!

--- ENGLISH TRANSLATION ---
SPEAKER_00: Hello, how are you?
SPEAKER_01: Very well, thank you!
```

### File Naming

Transcripts are saved to `media/transcripts/` with unique filenames:
```
audio_20260426_083000.txt
```

---

## 🧹 Code Quality

We use **Ruff** for linting, **Black** for formatting, **MyPy** for type checking, and **Bandit** for security scanning.

```bash
# Check for issues
make lint           # Ruff linting
make type-check     # MyPy type checking
make format         # Black formatting
make check-all      # Run all checks
make fix-all        # Auto-fix all issues

# Run full CI locally
make ci             # Run pre-commit, security scan, build
```

### Pre-commit Hooks

```bash
# Install hooks (runs before every commit)
make pre-commit-install

# Run manually
make pre-commit-run
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

# Start with debug logging
make dev-verbose
```

### Available Commands

```bash
make help           # Show all available commands
make build          # Build Docker image
make rebuild        # Rebuild without cache
make server         # Start API server
make docs           # Open Swagger UI
make logs           # View Docker logs
make stop           # Stop services
```

---

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build production image
make docker-build

# Run with Docker Compose
docker compose -f compose.yaml up -d

# View logs
make docker-logs

# Stop services
make stop
```

### Docker Compose Services

- **api**: FastAPI server with automatic transcription

---

## 📚 Documentation

- **[`.claude/CLAUDE.md`](.claude/CLAUDE.md)** - Comprehensive AI development guide
- **API Documentation** - Available at http://localhost:8000/docs when server is running

---

## 🔧 Troubleshooting

### Model Issues

```bash
# Clear model cache (local)
rm -rf model-cache/

# Clear Docker volume
docker volume rm voice-to-text_model-cache

# Test with smaller model
WHISPER_MODEL=tiny make dev
```

### Docker Issues

```bash
# View container logs
make docker-logs

# Restart services
make stop
make server

# Clean rebuild
make docker-rebuild
```

### Import Errors

```bash
# Reinstall dependencies
make setup

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes following conventions
4. Run `make check-all` before committing
5. Submit a pull request

### Code Style

- **Type hints**: Required for all functions
- **Docstrings**: Google style for functions/classes
- **Formatting**: Auto-applied by pre-commit hooks
- **Naming**: `snake_case` for files/functions, `PascalCase` for classes

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🎯 Roadmap

- [ ] Add more Whisper model variants
- [ ] Support for batch processing
- [ ] Real-time streaming transcription
- [ ] Additional language models
- [ ] Web UI for easy transcription
- [ ] Cloud deployment options

---

**Made with ❤️ using OpenAI Whisper, FastAPI, and UV**
