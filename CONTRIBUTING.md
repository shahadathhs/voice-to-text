# Contributing to Voice-to-Text

First off, thank you for considering contributing to Voice-to-Text! It's people
like you that make Voice-to-Text such a great tool.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Adding Features](#adding-features)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.
When you create a bug report, include as many details as possible:

**Bug Report Template:**

```markdown
**Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Linux, macOS, Windows]
 - Python Version: [e.g. 3.14.0]
 - Project Version: [e.g. 1.0.0]

**Additional Context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an
enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List some examples of how this feature would be used**
- **Include mockups or screenshots if applicable**

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/voice-to-text.git
cd voice-to-text
```

### 2. Install Dependencies

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make setup

# Install pre-commit hooks
make pre-commit-install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 4. Make Changes

```bash
# Run development server
make dev

# Run tests (when available)
make test

# Run linting and formatting
make check-all
```

## Pull Request Process

### 1. Before Submitting

Ensure your code passes all checks:

```bash
# Format code
make format

# Fix linting issues
make lint-fix

# Run type checking
make type-check

# Run all checks
make check-all
```

### 2. Submitting Your PR

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features or bug fixes
3. **Update README.md** if needed
4. **Push to your fork** and create a Pull Request

### 3. PR Description

Include the following in your PR description:

- **Summary**: Brief description of changes
- **Type**: Feature / Bug Fix / Refactoring / Documentation
- **Related Issues**: Link to related issues
- **Breaking Changes**: List any breaking changes
- **Testing**: Describe how you tested your changes

### 4. Code Review

- All PRs must be reviewed by at least one maintainer
- Address review comments promptly
- Keep the PR focused and small if possible
- Add commits to address review feedback (don't squash)

## Coding Standards

### Python Style Guide

We follow standard Python conventions:

```python
# Type hints are required
def transcribe_audio(
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

# Use snake_case for functions and variables
def process_transcript():
    transcript_text = "example"

# Use PascalCase for classes
class TranscriptionService:
    pass

# Use UPPER_SNAKE_CASE for constants
MAX_FILE_SIZE = 500 * 1024 * 1024
DEFAULT_MODEL = "base"
```

### File Organization

```
app/
├── api/           # API endpoints
├── cli/           # CLI interface
├── core/          # Configuration, errors
├── schemas/       # Pydantic models
├── services/      # Business logic
├── utils/         # Helper functions
└── whisper/       # Whisper implementations
```

### Import Order

```python
# 1. Standard library imports
import os
from pathlib import Path

# 2. Third-party imports
from fastapi import HTTPException
from pydantic import BaseModel

# 3. Local imports
from app.core.config import settings
from app.services.transcriber import TranscriptionService
```

### Documentation

- All public functions need docstrings
- Use Google style docstrings
- Include type hints
- Add inline comments for complex logic

```python
def process_segments(segments: list) -> dict:
    """Process transcription segments.

    This function takes raw segments from Whisper and processes them
    to add speaker labels and translations.

    Args:
        segments: List of transcription segments

    Returns:
        Processed segments with metadata

    Raises:
        ValueError: If segments are empty or invalid

    Example:
        >>> segments = [{"start": 0, "end": 1, "text": "Hello"}]
        >>> process_segments(segments)
        {"text": "Hello", "speakers": ["SPEAKER_00"]}
    """
    if not segments:
        raise ValueError("Segments cannot be empty")

    # Process each segment
    processed = []
    for segment in segments:
        processed.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"]
        })

    return {"segments": processed}
```

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process or auxiliary tool changes
- `ci`: CI/CD changes

### Examples

```bash
# Feature
git commit -m "feat(api): add speaker diarization endpoint"

# Bug fix
git commit -m "fix(transcriber): handle missing audio files gracefully"

# Documentation
git commit -m "docs(readme): update Docker setup instructions"

# Refactoring
git commit -m "refactor(services): extract validation logic"

# Breaking change
git commit -m "feat(api)!: change response format

BREAKING CHANGE: API responses now use a new format"
```

## Adding Features

### 1. Discuss First

Open an issue to discuss the feature before implementing:

- **Why** do we need this feature?
- **What** problem does it solve?
- **How** should it work?

### 2. Implementation Checklist

- [ ] Feature discussed and approved
- [ ] Branch created from `main`
- [ ] Code written with type hints
- [ ] Docstrings added
- [ ] Tests written (if applicable)
- [ ] Documentation updated
- [ ] All checks passing
- [ ] PR submitted for review

### 3. Feature Templates

**API Endpoints:**

```python
from fastapi import APIRouter, UploadFile
from app.schemas.transcription import TranscriptionResponse

router = APIRouter()

@router.post("/your-endpoint", response_model=TranscriptionResponse)
async def your_endpoint(
    file: UploadFile,
    param: str = "default"
) -> TranscriptionResponse:
    """Your endpoint description.

    Args:
        file: Uploaded file
        param: Parameter description

    Returns:
        Transcription response with results
    """
    # Implementation here
    pass
```

**CLI Commands:**

```python
import click

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--your-option', default='default', help='Option description')
def your_command(audio_file: str, your_option: str):
    """Your command description."""
    # Implementation here
    pass
```

## Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run with verbose output
uv run pytest -v
```

## Getting Help

- **Documentation**: Check [README.md](README.md) and [CLAUDE.md](.claude/CLAUDE.md)
- **Issues**: Search or create a GitHub issue
- **Discussions**: Use GitHub Discussions for questions

## Recognition

Contributors who add significant features or fix critical bugs will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in the README

## License

By contributing, you agree that your contributions will be licensed under
the [MIT License](LICENSE).

---

Thank you for your contributions! 🎉
