# Default values
AUDIO ?= audio.wav
MODEL ?= base

.PHONY: help build rebuild server stop docs lint format run translate diarize all clean

# Default target - show help
help:
	@echo "Voice-to-Text Server - Available Commands:"
	@echo ""
	@echo "🚀 Server Commands:"
	@echo "  make server    - Start the FastAPI server (recommended)"
	@echo "  make stop      - Stop the running server"
	@echo "  make docs      - Open Swagger UI documentation"
	@echo ""
	@echo "🔨 Build Commands:"
	@echo "  make build     - Build the Docker image"
	@echo "  make rebuild   - Force rebuild (no cache)"
	@echo ""
	@echo "🧹 Code Quality:"
	@echo "  make lint      - Check code with ruff"
	@echo "  make format    - Auto-format code with ruff"
	@echo ""
	@echo "💻 CLI Commands (Legacy):"
	@echo "  make run AUDIO=file.mp3       - Basic transcription"
	@echo "  make translate AUDIO=file.mp3 - Transcription + Translation"
	@echo "  make diarize AUDIO=file.mp3   - Transcription + Diarization"
	@echo "  make all AUDIO=file.mp3       - All features"
	@echo ""
	@echo "🗑️  Cleanup:"
	@echo "  make clean     - Remove all transcripts"

# ============================================
# Server Commands (Primary Workflow)
# ============================================

# Build the Docker image
build:
	sudo docker compose build

# Force rebuild (no cache)
rebuild:
	sudo docker compose build --no-cache

# Start the API Server (FastAPI)
server:
	@echo "🚀 Starting FastAPI server..."
	@echo "📖 Swagger UI will be available at: http://localhost:8000/docs"
	sudo docker compose up whisper

# Stop the running server
stop:
	sudo docker compose down

# Open API documentation
docs:
	@echo "Opening Swagger UI..."
	xdg-open http://localhost:8000/docs || echo "Please open http://localhost:8000/docs in your browser"

# ============================================
# Code Quality
# ============================================

# Linter (ruff)
lint:
	sudo docker compose run --rm whisper ruff check .

# Formatter (ruff)
format:
	sudo docker compose run --rm whisper ruff format .

# ============================================
# CLI Commands (Legacy - Use API Instead)
# ============================================

# Basic transcription
run:
	sudo docker compose run --rm whisper python transcribe.py $(AUDIO) --model $(MODEL)

# Transcription + Translation
translate:
	sudo docker compose run --rm whisper python transcribe.py $(AUDIO) --model $(MODEL) --translate

# Transcription + Diarization
diarize:
	sudo docker compose run --rm whisper python transcribe.py $(AUDIO) --model $(MODEL) --diarize

# All-in-one: Transcribe + Translate + Diarize
all:
	sudo docker compose run --rm whisper python transcribe.py $(AUDIO) --model $(MODEL) --translate --diarize

# ============================================
# Cleanup
# ============================================

# Clean transcripts folder
clean:
	rm -rf transcripts/*
