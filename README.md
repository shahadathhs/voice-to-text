# Voice-to-Text

AI-powered voice transcription service using OpenAI's Whisper model. Provides both a command-line interface and REST API server with support for transcription, translation, and speaker diarization.

## Features

- **CLI and API**: Use as a command-line tool or deploy as a REST API server
- **Translation**: Translate non-English audio to English
- **Speaker Diarization**: Identify and label different speakers in conversations
- **Multiple Backends**: OpenAI Whisper or Hugging Face Transformers
- **Fully Local**: Complete privacy - your audio never leaves your machine
- **Python 3.14+ Support**: Modern Python with UV package manager
- **Docker Ready**: One-command deployment with Docker Compose

## Quick Start

### Docker (Recommended)

```bash
# Start the API server
make server

# Access API documentation at http://localhost:8000/docs
```

### Local Installation

```bash
# Install system dependencies (UV, FFmpeg) and Python packages
./setup.sh

# Run CLI
voice-to-text media/audio/sample.wav

# Or use the short alias
vtt media/audio/sample.wav

# Run API server
make dev
```

## Usage

### CLI Usage

The command-line interface provides direct access to transcription features:

```bash
# Basic transcription
voice-to-text media/audio/sample.wav

# With translation to English
voice-to-text media/audio/sample.wav --translate

# With speaker diarization
voice-to-text media/audio/meeting.mp3 --diarize

# All features combined
voice-to-text media/audio/conversation.wav --translate --diarize --max-speakers 2

# Use different model
voice-to-text media/audio/sample.wav --model tiny

# Custom output location
voice-to-text media/audio/sample.wav --output transcript.txt
```

#### CLI Commands via Makefile

The Makefile provides convenient shortcuts for common tasks:

```bash
# Transcription commands
make transcribe FILE=media/audio/sample.wav
make transcribe-tiny FILE=media/audio/sample.mp3
make transcribe-translate FILE=media/audio/meeting.wav
make transcribe-diarize FILE=media/audio/conversation.mp3
make transcribe-all FILE=media/audio/file.wav

# File management
make list-audio          # List available audio files
make list-transcripts    # List transcript files
make clean-transcripts   # Clean all transcripts

# CLI management
make cli-info            # Show configuration and status
make cli-dirs            # Ensure all directories exist
```

### API Usage

Start the server and upload audio files via the REST API:

```bash
# Start server
make server

# Transcribe with curl
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@media/audio/sample.mp3"

# With translation and diarization
curl -X POST "http://localhost:8000/transcribe?translate=true&diarize=true" \
  -F "file=@media/audio/meeting.mp3"
```

Access interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Response Format

The API returns full URLs for accessing uploaded audio and transcripts:

```json
{
  "transcript": "SPEAKER_00: Hello world\nSPEAKER_01: Hi there",
  "saved_to": "media/transcripts/file_20260427_123456.txt",
  "metadata": {
    "model": "base",
    "backend": "openai",
    "device": "cpu",
    "translated": false,
    "diarized": true,
    "audio_file": "/uploads/filename.mp3",
    "audio_url": "http://localhost:8000/uploads/filename.mp3",
    "transcript_file": "filename_20260427_123456.txt",
    "transcript_url": "http://localhost:8000/transcripts/filename_20260427_123456.txt"
  }
}
```

## Project Structure

```
voice-to-text/
├── app/                       # Application code
│   ├── api/                   # REST API endpoints and routes
│   ├── cli/                   # Command-line interface
│   ├── core/                  # Configuration, errors, logging
│   ├── schemas/               # Pydantic validation models
│   ├── services/              # Business logic (transcription, diarization)
│   ├── utils/                 # Utilities and helpers
│   └── whisper/               # Whisper model implementations
├── media/                     # Media files directory
│   ├── audio/                 # Default audio files (CLI usage)
│   ├── uploads/               # Uploaded audio files (API usage)
│   └── transcripts/           # Transcription output
├── compose.yaml               # Docker Compose configuration
├── Dockerfile                 # Production container image
├── Makefile                   # Development automation
├── pyproject.toml            # Project configuration and dependencies
└── setup.sh                   # System dependency installer
```

## Configuration

### Environment Variables

Create a `.env` file from `.env.example` to configure the application:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Whisper Configuration
WHISPER_MODEL=base           # tiny, base, small, medium, large
WHISPER_BACKEND=openai       # openai, transformers
WHISPER_DEVICE=cpu           # cpu, cuda

# Feature Flags
ENABLE_TRANSLATION=false
ENABLE_DIARIZATION=false

# Diarization Settings
DIARIZE_THRESHOLD=0.35       # Clustering threshold (0.0-1.0)
MAX_SPEAKERS=2               # Maximum number of speakers (optional)
USE_SILHOUETTE=false         # Use silhouette analysis

# Media Directories
AUDIO_DIR=media/audio         # Default audio files (CLI)
UPLOADS_DIR=media/uploads     # Uploaded files (API)
TRANSCRIPT_DIR=media/transcripts
MODEL_CACHE_DIR=model-cache

# API Configuration
MAX_FILE_SIZE=524288000      # 500 MB
ALLOWED_FORMATS=wav,mp3,ogg,m4a,flac,aac
API_HOST=http://localhost:8000  # For constructing full URLs (optional)
```

### CLI Options

```bash
voice-to-text audio.wav [OPTIONS]

Positional Arguments:
  input                   Path to audio file

Options:
  --model {tiny,base,small,medium,large}
                          Whisper model size (default: base)
  --backend {openai,transformers}
                          Whisper backend (default: openai)
  --translate             Translate non-English audio to English
  --diarize               Enable speaker diarization
  --diarize-threshold N   Clustering threshold (0.0-1.0, default: 0.35)
  --max-speakers N        Fixed number of speakers
  --use-silhouette        Estimate speakers from embeddings
  --output, -o PATH       Output file path
  --media-dir PATH        Custom media directory
  --ensure-dirs           Create directories if needed (default: enabled)
  --no-ensure-dirs        Disable directory creation
  --verbose, -v           Enable verbose output
  --debug                 Enable debug mode
```

## Development

### Setup

```bash
# Install all dependencies (system and Python)
make install-deps

# Install pre-commit hooks
make pre-commit-install
```

### Code Quality

The project uses modern Python tooling for code quality:

```bash
# Run all checks
make ci                    # Full CI pipeline

# Individual checks
make lint                 # Ruff linting
make type-check           # MyPy type checking
make security             # Bandit security scan
make format               # Black code formatting
make fix-all              # Auto-fix all issues
```

### Building

```bash
# Build distribution packages
make build

# Build Docker image
make docker-build

# Rebuild without cache
make docker-rebuild
```

### Docker Commands

```bash
make server               # Start API server (Docker)
make stop                 # Stop server
make restart              # Restart server
make logs                 # View logs
make ps                   # Check container status
```

## Tooling and Stack

### Core Technologies
- **Python 3.14+**: Modern Python with latest features
- **UV**: Ultra-fast Python package manager (10-100x faster than pip)
- **FastAPI 0.115+**: Modern, fast web framework for building APIs
- **Pydantic v2**: Data validation using Python type annotations
- **OpenAI Whisper**: State-of-the-art speech recognition model

### Audio Processing
- **Librosa**: Audio loading and processing (Python 3.13+ compatible)
- **SpeechBrain**: Speaker diarization and embeddings
- **SoundFile**: Audio I/O operations

### Code Quality Tools
- **Ruff**: Extremely fast Python linter and formatter
- **Black**: Code formatting (PEP 8 compliant)
- **MyPy**: Static type checking
- **Bandit**: Security linter
- **Pre-commit**: Git hooks for automated quality checks

### Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD pipeline with semantic-release

## Installation Options

### Option 1: Docker (Recommended for Production)

```bash
git clone https://github.com/shahadathhs/voice-to-text.git
cd voice-to-text
make server
```

### Option 2: Local Development

```bash
# Prerequisites: Python 3.14+, FFmpeg

git clone https://github.com/shahadathhs/voice-to-text.git
cd voice-to-text
./setup.sh
make dev
```

### Option 3: CLI Only (No Server)

```bash
git clone https://github.com/shahadathhs/voice-to-text.git
cd voice-to-text
./setup.sh
voice-to-text path/to/audio.wav
```

## API Endpoints

### Core Endpoints

- `GET /` - API information and endpoints
- `GET /health` - Health check and service status
- `POST /transcribe` - Transcribe audio file

### File Serving

- `GET /uploads/{filename}` - Retrieve uploaded audio file
- `GET /transcripts/{filename}` - Retrieve transcript file

### Documentation

- `GET /docs` - Swagger UI (interactive API documentation)
- `GET /redoc` - ReDoc (alternative documentation)
- `GET /openapi.json` - OpenAPI specification

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run quality checks: `make ci`
5. Commit with conventional commit format: `git commit -m "feat: add amazing feature"`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Commit Convention

Follow conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [SECURITY.md](SECURITY.md) - Security policy
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines
- [.claude/CLAUDE.md](.claude/CLAUDE.md) - Development guide

## License

MIT License - see [LICENSE](LICENSE) for details.

## Requirements

- Python 3.14 or higher
- FFmpeg (for audio processing)
- 4GB RAM minimum (8GB recommended for larger models)
- 1GB disk space for models

## Support

- GitHub Issues: https://github.com/shahadathhs/voice-to-text/issues
- Documentation: http://localhost:8000/docs (when server is running)
