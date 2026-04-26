# Voice-to-Text - AI Development Guide

**Version**: 1.0.0
**Last Updated**: 2026-04-26
**Framework**: FastAPI 0.115.0+, Python 3.14+

---

## 🎯 Project Overview

Voice-to-Text is an AI-powered voice transcription service using OpenAI's Whisper model. The system uses FastAPI for the REST API with support for transcription, translation, and speaker diarization using SpeechBrain.

### Core Architecture
- **API Layer**: FastAPI with Pydantic v2 for validation
- **Transcription Engine**: OpenAI Whisper (multiple model sizes)
- **Speaker Diarization**: SpeechBrain for speaker identification
- **Package Manager**: UV for ultra-fast dependency management
- **Code Quality**: Ruff, Black, MyPy, Bandit, Pre-commit hooks
- **CI/CD**: GitHub Actions with semantic-release

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
uv                        # Package manager (10-100x faster than pip)
ruff==0.15.11            # Linting and formatting
black==26.3.1            # Code formatting
mypy==1.20.1             # Type checking
bandit>=1.9.4            # Security scanning
pre-commit>=4.5.1        # Git hooks
python-semantic-release  # Automated versioning
```

---

## 🏗️ Project Structure

```
voice-to-text/
├── voice_to_text/              # Package directory
│   ├── __init__.py            # Package init with __version__
│   ├── backends/              # Whisper backends
│   ├── cli.py                 # CLI interface (legacy)
│   ├── config.py              # Configuration settings
│   ├── diarization.py         # Speaker diarization
│   ├── io_utils.py            # File I/O utilities
│   └── pipeline.py            # Transcription pipeline
├── audio/                      # Input audio files
├── transcripts/                # Output transcripts
├── tests/                      # Test suite (future)
├── docs/                       # Documentation
├── pyproject.toml             # Project configuration
├── Makefile                   # Development automation
├── compose.yaml               # Docker compose setup
├── Dockerfile                 # Production container image
├── .pre-commit-config.yaml    # Git hooks
├── .env.example               # Environment template
└── CLAUDE.md                  # This file
```

---

## 🔧 Development Workflow

### Initial Setup
```bash
# Clone and setup
git clone https://github.com/shahadathhs/voice-to-text.git
cd voice-to-text
make setup              # Creates venv + installs dependencies
make pre-commit-install # Install git hooks (one-time)
```

### Daily Development
```bash
# Start development server with hot reload
make dev                # http://localhost:8000
make dev-verbose        # With debug logging

# Run code quality checks
make lint               # Ruff linting
make type-check         # MyPy type checking
make format             # Black formatting
make check-all          # All quality checks
make fix-all            # Auto-fix issues
```

### Docker Development
```bash
# Start with Docker (recommended)
make server             # Start API server
make docs               # Open Swagger UI

# Build and rebuild
make docker-build       # Build Docker image
make docker-rebuild     # Rebuild without cache

# Logs and management
make docker-logs        # View logs
make stop               # Stop services
```

### Code Quality Standards
```bash
# Before committing
make fix-all            # Fix formatting and linting
make check-all          # Verify all checks pass

# Git hooks (pre-commit) run automatically
make pre-commit-run     # Run manually if needed
```

---

## 📝 Code Conventions

### File Organization
- **Routes**: Inline in `server.py` for simple API
- **Models**: Pydantic models in `voice_to_text/`
- **Services**: Business logic in `voice_to_text/pipeline.py`
- **Config**: Settings in `voice_to_text/config.py`

### Naming Conventions
```python
# Files and modules: snake_case
transcriber_service.py
audio_processor.py

# Classes: PascalCase
class TranscriptionService:
    class AudioProcessor:

# Functions and variables: snake_case
def transcribe_audio():
    audio_file = "test.wav"

# Constants: UPPER_SNAKE_CASE
MAX_FILE_SIZE = 500
DEFAULT_MODEL = "base"

# Private: leading underscore
def _internal_helper():
    _private_var = "internal"
```

### Type Hints
```python
# Always include type hints
from typing import Optional
from pathlib import Path

def transcribe_file(
    audio_path: Path,
    model: str = "base",
    language: Optional[str] = None
) -> dict:
    """Transcribe an audio file.

    Args:
        audio_path: Path to audio file
        model: Whisper model size
        language: Language code (auto-detect if None)

    Returns:
        Transcription result dict
    """
    pass
```

### Error Handling
```python
# Use FastAPI exception handlers
from fastapi import HTTPException
from loguru import logger

# Raise HTTP exceptions for API errors
if not audio_file.exists():
    raise HTTPException(
        status_code=404,
        detail=f"Audio file not found: {audio_file}"
    )

# Log errors appropriately
logger.error(f"Failed to transcribe: {e}", exc_info=True)
```

### Async Patterns
```python
# Use async for I/O operations
from fastapi import UploadFile

@app.post("/transcribe")
async def transcribe_endpoint(
    audio_file: UploadFile
) -> dict:
    # Process file asynchronously
    result = await transcribe_audio(audio_file)
    return result
```

---

## 🏛️ Architecture Patterns

### Configuration Management
```python
# Use environment variables
import os
from pathlib import Path

# Get configuration from environment
model_size = os.getenv("WHISPER_MODEL", "base")
device = os.getenv("WHISPER_DEVICE", "cpu")

# Path handling
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/app/audio"))
TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "/app/transcripts"))
```

### Response Formatting
```python
# Use standardized response format
from fastapi import Response
from typing import Any

def success_response(
    data: Any,
    message: str = "Success",
    status_code: int = 200
) -> dict:
    return {
        "status": "success",
        "message": message,
        "data": data
    }

def error_response(
    message: str,
    details: Any = None,
    status_code: int = 400
) -> dict:
    return {
        "status": "error",
        "message": message,
        "details": details
    }
```

### File Processing
```python
# Handle file uploads safely
from fastapi import UploadFile
import shutil

async def save_upload_file(
    upload_file: UploadFile,
    destination: Path
) -> Path:
    """Save uploaded file to destination."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return destination
    finally:
        upload_file.file.close()
```

---

## 🧪 Testing Guidelines (Future)

### Test Organization
```python
# Test structure mirrors app structure
tests/
├── test_api.py              # API endpoint tests
├── test_pipeline.py         # Transcription pipeline tests
├── test_diarization.py      # Diarization tests
└── conftest.py              # Pytest configuration
```

### Test Patterns
```python
# Use pytest with async support
import pytest
from httpx import AsyncClient
from server import app

@pytest.mark.asyncio
async def test_transcribe_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test transcription endpoint
        response = await client.post(
            "/api/transcribe",
            files={"file": ("test.wav", open("test.wav", "rb"), "audio/wav")}
        )
    assert response.status_code == 200
    assert "text" in response.json()
```

---

## 🚀 Deployment Considerations

### Environment Variables
```bash
# Required environment variables
ENVIRONMENT=production
DEBUG=false
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
SECRET_KEY=your-production-secret-key

# Optional
WHISPER_LANGUAGE=en
ENABLE_DIARIZATION=false
ENABLE_TRANSLATION=false
```

### Production Checklist
- [ ] Set `DEBUG=false` in production
- [ ] Use strong `SECRET_KEY`
- [ ] Configure proper file size limits
- [ ] Set up log aggregation
- [ ] Enable health checks
- [ ] Set up monitoring/alerting
- [ ] Use production ASGI server (uvicorn with workers)

### Docker Production
```bash
# Build production image
make docker-build

# Run production stack
docker compose -f compose.yaml up -d

# Or using docker compose directly
docker compose -f compose.yaml up -d
```

---

## 📚 Documentation Standards

### Docstring Format
```python
def complex_function(param1: str, param2: int) -> dict:
    """Brief description of function.

    Longer description if needed. Explain the why, not just the what.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param1 is invalid

    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        {'status': 'success'}
    """
    pass
```

### API Documentation
- Use FastAPI's automatic OpenAPI docs
- Add detailed descriptions to endpoints
- Include request/response examples
- Document authentication requirements
- Tag endpoints by feature/domain

---

## 🎯 Common Tasks

### Adding a New Endpoint
1. Define endpoint in `server.py`
2. Create Pydantic request/response models
3. Add business logic in `voice_to_text/pipeline.py`
4. Add tests in `tests/test_api.py`
5. Update API documentation

### Updating Dependencies
```bash
# Add new dependency
make add PKG=package-name

# Update all dependencies
make update

# Update lock file
make freeze

# Remove dependency
make remove PKG=package-name
```

### Creating a Release
```bash
# Follow conventional commit format
git commit -m "feat: add new feature"

# Push to main branch
git push origin main

# Semantic release will automatically:
# - Bump version
# - Create git tag
# - Generate changelog
# - Create GitHub release
```

---

## 🐛 Debugging Tips

### Enable Debug Logging
```bash
# Run with debug logging
make dev-verbose

# Or set environment variable
export LOG_LEVEL=DEBUG
make dev
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

### Model Issues
```bash
# Clear model cache
rm -rf model-cache/

# Test with different model
WHISPER_MODEL=tiny make dev
```

---

## 📋 Pre-Commit Checklist

Before committing code, ensure:
- [ ] Code formatted with black (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] Type checking passes (`make type-check`)
- [ ] New features include tests
- [ ] Documentation updated
- [ ] Environment variables documented in `.env.example`

---

## 🔐 Security Considerations

### File Upload Security
- Validate file types and sizes
- Sanitize filenames
- Use secure temporary directories
- Scan uploaded files for malware

### API Security
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Use environment variables for secrets

### Secrets Management
- Never commit `.env` files
- Use environment variables for secrets
- Rotate secrets regularly
- Use different secrets for dev/prod

---

## 📞 Support and Contribution

### Getting Help
- Check existing documentation
- Review existing code patterns
- Run `make help` for available commands
- Check logs with `make docker-logs`

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes following conventions
4. Run `make check-all` before committing
5. Submit a pull request with description

### Code Review Focus
- Adherence to code conventions
- Test coverage
- Documentation completeness
- Security considerations
- Performance implications

---

## 🎓 Learning Resources

### Project-Specific
- [Whisper Documentation](https://github.com/openai/whisper)
- [SpeechBrain Documentation](https://speechbrain.readthedocs.io/)
- [FastAPI Official Docs](https://fastapi.tiangolo.com/)

### Tooling
- [UV Documentation](https://github.com/astral-sh/uv)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Semantic Release](https://python-semantic-release.readthedocs.io/)

---

## 🔄 Version Management

### Semantic Release
This project uses semantic-release for automated versioning based on commit messages:

- `feat:` - Minor version bump (new features)
- `fix:` - Patch version bump (bug fixes)
- `BREAKING CHANGE:` - Major version bump

### Example Commits
```bash
git commit -m "feat: add speaker diarization support"
git commit -m "fix: handle missing audio files gracefully"
git commit -m "feat: add translation support

BREAKING CHANGE: API endpoint structure changed"
```

---

## 🎯 Migration from Legacy Setup

### From pip to UV
```bash
# Old way (deprecated)
pip install -r requirements.txt

# New way
make setup  # or manually: uv sync
```

### From docker-compose.yml to compose.yaml
```bash
# Old way
docker-compose up

# New way
make server  # or manually: docker compose -f compose.yaml up
```

### From Manual Versioning to Semantic Release
```bash
# Old way (manual)
# Edit version in files, create tags manually

# New way (automatic)
# Just follow conventional commit format
git commit -m "feat: add new feature"
git push
# Version is automatically bumped and tagged
```

---

**Note**: This document is a living guide. Update it as the project evolves and new patterns emerge.

**Reference**: Based on EcoRoute Atlas tooling standards
