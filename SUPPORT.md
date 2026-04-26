# Support

## Table of Contents

- [Getting Help](#getting-help)
- [Documentation](#documentation)
- [Common Issues](#common-issues)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Reporting Issues](#reporting-issues)

## Getting Help

### Quick Help

- **Documentation**: [README.md](README.md)
- **Development Guide**: [.claude/CLAUDE.md](.claude/CLAUDE.md)
- **API Docs**: http://localhost:8000/docs (when server is running)

### Community Support

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions and share ideas
- **Pull Requests**: Contribute fixes or improvements

## Documentation

### Quick Start Guide

```bash
# Clone the repository
git clone https://github.com/shahadathhs/voice-to-text.git
cd voice-to-text

# Setup (installs UV, FFmpeg, dependencies)
./setup.sh

# Start API server
make server

# Or use CLI
uv run python cli.py media/audio/sample.wav
```

### Available Commands

```bash
make help           # Show all available commands
make dev            # Start development server
make docs           # Open API documentation
make check-all      # Run all quality checks
```

## Common Issues

### Issue: "Module not found" Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'whisper'
```

**Solution:**
```bash
# Reinstall dependencies
make setup

# Or manually
uv sync
```

### Issue: "FFmpeg not found"

**Symptoms:**
```
Error: FFmpeg is not installed
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Or run the full setup
./setup.sh
```

### Issue: Docker container fails to start

**Symptoms:**
```
Error: Cannot connect to Docker daemon
```

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify Docker is running
docker --version
docker compose version
```

### Issue: "Permission denied" errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/app/media'
```

**Solution:**
```bash
# Fix permissions
chmod -R 755 media/

# Or reset virtual environment
make reset-venv
```

### Issue: Models download every time

**Symptoms:**
Models are re-downloaded on every run (slow startup).

**Solution:**
```bash
# Check model cache exists
ls -la model-cache/

# Clear and redownload
rm -rf model-cache/
make dev  # Models will download once

# For Docker, check volume
docker volume ls | grep model-cache
```

### Issue: Out of memory errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Use smaller model
WHISPER_MODEL=tiny make dev

# Or use CPU
WHISPER_DEVICE=cpu make dev
```

## Troubleshooting

### Debug Mode

Enable debug logging to see detailed error messages:

```bash
# Development
make dev-verbose

# Or set environment variable
export LOG_LEVEL=DEBUG
make dev
```

### Health Check

Check if the server is running:

```bash
# Check server status
make status

# Or manually
curl http://localhost:8000/health

# Check Docker logs
make docker-logs
```

### Reset Everything

If nothing else works, do a clean reset:

```bash
# Stop all services
make stop

# Clean everything (including models)
make clean

# Rebuild from scratch
make docker-rebuild

# Or local reset
make reset-venv
make setup
```

### Check Dependencies

Verify all dependencies are installed:

```bash
# List installed packages
make list

# Check UV installation
uv --version

# Check Python version
python3 --version
```

## FAQ

### General Questions

**Q: Is this really free and offline?**
A: Yes! OpenAI Whisper is open-source and runs entirely on your machine. No API keys needed.

**Q: What audio formats are supported?**
A: WAV, MP3, OGG, M4A, FLAC, AAC

**Q: How accurate is the transcription?**
A: Whisper is very accurate, especially with clear audio. Accuracy depends on audio quality, background noise, and speaker clarity.

**Q: Can I use this for commercial purposes?**
A: Yes, it's licensed under MIT. See [LICENSE](LICENSE) for details.

### Performance Questions

**Q: Which model should I use?**
A:
- `tiny`: Fastest, least accurate (~1GB RAM)
- `base`: Good balance (~1GB RAM)
- `small`: Better accuracy (~2GB RAM)
- `medium`: High accuracy (~5GB RAM)
- `large`: Best accuracy (~10GB RAM)

**Q: Can I use GPU?**
A: Yes! Set `WHISPER_DEVICE=cuda` if you have NVIDIA GPU with CUDA.

**Q: How long does transcription take?**
A: Roughly 10-20% of real-time on CPU, faster on GPU. A 10-minute file takes ~1-2 minutes.

### Feature Questions

**Q: What languages are supported?**
A: Whisper supports 99 languages. See [OpenAI's docs](https://github.com/openai/whisper) for full list.

**Q: Can I identify different speakers?**
A: Yes! Use the `--diarize` flag or `diarize=true` query parameter.

**Q: Can I translate to English?**
A: Yes! Use the `--translate` flag or `translate=true` query parameter.

**Q: Is there a batch processing mode?**
A: Not yet, but it's on the roadmap. You can script it:
```bash
for file in media/audio/*.wav; do
    uv run python cli.py "$file" --diarize
done
```

### Deployment Questions

**Q: Can I deploy this to a server?**
A: Yes! Use Docker:
```bash
docker compose -f compose.yaml up -d
```

**Q: What are the server requirements?**
A:
- **Minimum**: 2GB RAM, 2 CPU cores
- **Recommended**: 4GB RAM, 4 CPU cores
- **For large models**: 8GB+ RAM

**Q: Can I run this behind a reverse proxy?**
A: Yes! Configure nginx/Apache to proxy to `http://localhost:8000`

### Development Questions

**Q: How do I add a new feature?**
A: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Q: How do I run tests?**
A: `make test` (tests are a work in progress)

**Q: What's the development workflow?**
A:
1. Fork and clone
2. Create a feature branch
3. Make changes
4. Run `make check-all`
5. Submit a PR

## Reporting Issues

### Before Reporting

1. **Search existing issues**: Check if your issue was already reported
2. **Try debug mode**: `make dev-verbose`
3. **Check documentation**: [README.md](README.md) and [CLAUDE.md](.claude/CLAUDE.md)

### When Reporting

Include this information:

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Error occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happens (include error messages)

## Environment
- OS: [e.g. Ubuntu 22.04]
- Python Version: [e.g. 3.14.0]
- Project Version: [e.g. 1.0.0]
- Installation Method: [Docker / Local]

## Logs
Include relevant logs from:
- make dev-verbose
- make docker-logs
```

### Issue Templates

Use the GitHub issue templates:
- **Bug Report**: For bugs and errors
- **Feature Request**: For new features
- **Documentation**: For docs improvements

## Additional Resources

### Official Documentation

- [Whisper GitHub](https://github.com/openai/whisper)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [UV Documentation](https://github.com/astral-sh/uv)
- [SpeechBrain Documentation](https://speechbrain.readthedocs.io/)

### Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

### Professional Support

For enterprise support or custom development, please open a discussion on GitHub.

---

Still need help? [Open an issue](https://github.com/shahadathhs/voice-to-text/issues/new)
