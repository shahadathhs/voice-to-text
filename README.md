# Voice-to-Text: CLI + FastAPI Server

[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/release/python-3140/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Fully local** AI-powered voice transcription using OpenAI's Whisper. No API keys, no cloud calls, complete privacy.

---

## Features

- **CLI & API**: Command-line tool or REST API server
- **Translation**: Translate to English
- **Speaker Diarization**: Identify and label different speakers
- **Multiple Backends**: OpenAI Whisper or Hugging Face Transformers
- **Docker Support**: One-command deployment
- **Fully Local**: Your audio never leaves your machine

---

## Quick Start

### Docker (Recommended)

```bash
# Start API server
make server

# Access API docs at http://localhost:8000/docs
```

### Local Installation

```bash
# Install dependencies
./setup.sh

# Run CLI
uv run python cli.py media/audio/sample.wav --translate --diarize

# Run API server
make dev
```

### Example Usage

```bash
# CLI
uv run python cli.py media/audio/sample.wav --diarize

# API
curl -X POST "http://localhost:8000/transcribe?diarize=true" \
  -F "file=@media/audio/sample.mp3"
```

---

## Project Structure

```
voice-to-text/
├── app/                    # Application code
│   ├── api/               # REST API endpoints
│   ├── cli/               # CLI interface
│   ├── core/              # Configuration & errors
│   ├── schemas/           # Pydantic models
│   ├── services/          # Business logic
│   └── whisper/           # Whisper implementations
├── media/                 # Audio input & transcript output
├── cli.py                 # CLI entry point
├── server.py              # API server entry point
└── Makefile              # Automation commands
```

---

## Documentation

| Document | Description |
|----------|-------------|
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | How to contribute |
| **[SECURITY.md](SECURITY.md)** | Security policy |
| **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** | Community guidelines |
| **[.claude/CLAUDE.md](.claude/CLAUDE.md)** | Development guide |

**API Documentation**: http://localhost:8000/docs (when server is running)

---

## Configuration

### Environment Variables

```bash
# .env file
WHISPER_MODEL=base           # tiny, base, small, medium, large
WHISPER_BACKEND=openai       # openai, transformers
WHISPER_DEVICE=cpu           # cpu, cuda
ENABLE_TRANSLATION=false
ENABLE_DIARIZATION=false
```

### CLI Options

```bash
uv run python cli.py audio.wav [OPTIONS]

Options:
  --model {tiny,base,small,medium,large}  Model size (default: base)
  --translate                            Translate to English
  --diarize                              Enable speaker diarization
  --backend {openai,transformers}         Whisper backend
  --verbose, -v                          Enable verbose output
```

---

## Development

```bash
# Setup development environment
make setup

# Run development server with hot reload
make dev

# Run code quality checks
make check-all          # Run all checks
make fix-all           # Auto-fix issues
make lint              # Ruff linting
make format            # Black formatting
make type-check        # MyPy type checking

# Run tests
make test

# Build Docker image
make docker-build
```

---

## Docker

```bash
# Start server
make server

# Stop server
make stop

# View logs
make logs

# Rebuild
make docker-rebuild
```

---

## Contributing

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines.

Quick steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check-all`
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with OpenAI Whisper, FastAPI, and UV**
