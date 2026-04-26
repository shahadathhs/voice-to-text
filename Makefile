# Default values for legacy commands
AUDIO ?= audio.wav
MODEL ?= base

# Docker Variables
DOCKER_IMAGE := voice-to-text:latest
COMPOSE_FILE := compose.yaml

# Python Variables
VENV := .venv
PYTHON_BIN := uv run python

# Phony targets
.PHONY: help setup venv install reset-venv pre-commit-install pre-commit-run pre-commit-update
.PHONY: build dev dev-verbose prod release-dry-run release-changelog release-publish
.PHONY: lint lint-fix format format-check type-check check-all fix-all
.PHONY: docker-build docker-up docker-down docker-logs docker-rebuild
.PHONY: clean shell logs update freeze list add add-dev remove ci security info
.PHONY: server stop docs run translate diarize all

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Voice-to-Text - Available Commands:"
	@echo ""
	@echo "🚀 Setup Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;32m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📖 Quick Start:"
	@echo "  make setup              - Full setup (recommended first step)"
	@echo "  make server             - Start API server"
	@echo "  make docs               - Open API documentation"
	@echo ""
	@echo "💻 Legacy CLI Commands (deprecated - use API instead):"
	@echo "  make run AUDIO=file.mp3 - Basic transcription"
	@echo "  make all AUDIO=file.mp3 - All features (transcribe + translate + diarize)"

# =============================================================================
# SETUP
# =============================================================================
venv: ## Create virtual environment
	@echo "Creating virtual environment..."
	@rm -rf .venv 2>/dev/null || true
	@uv venv
	@echo "✓ Virtual environment created"

reset-venv: ## Force reset virtual environment (fix permission issues)
	@echo "Resetting virtual environment..."
	@rm -rf .venv .uv uv.lock 2>/dev/null || true
	@uv venv
	@uv sync
	@echo "✓ Virtual environment reset complete"

install: ## Install dependencies
	@echo "Installing dependencies..."
	@uv sync
	@echo "✓ Dependencies installed"

setup: venv install ## Full setup (venv + install)
	@echo "✓ Setup complete!"
	@echo "Run 'make server' to start the API server"

pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@uv run pre-commit install
	@echo "✓ Pre-commit hooks installed"

pre-commit-run: ## Run all pre-commit hooks manually
	@echo "Running pre-commit hooks..."
	@uv run pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "Updating pre-commit hooks..."
	@uv run pre-commit autoupdate
	@echo "✓ Pre-commit hooks updated"

# =============================================================================
# RUNNING
# =============================================================================
build: ## Build distribution packages
	@echo "Building distribution packages..."
	@uv run python -m build
	@echo "✓ Built packages in dist/"

dev: ## Run in development mode with hot reload
	$(PYTHON_BIN) -m uvicorn server:app --reload --host 0.0.0.0 --port 8000

dev-verbose: ## Run in development mode with verbose logging
	$(PYTHON_BIN) -m uvicorn server:app --reload --log-level debug

prod: ## Run in production mode
	$(PYTHON_BIN) -m uvicorn server:app --host 0.0.0.0 --port $${PORT:-8000} --workers 4

release-dry-run: ## Preview release without publishing (semantic-release)
	@echo "Previewing release..."
	@uv run semantic-release version --no-commit --no-tag

release-changelog: ## Generate changelog only (semantic-release)
	@echo "Generating changelog..."
	@uv run semantic-release changelog

release-publish: ## Publish release (semantic-release - used by CI)
	@echo "Publishing release..."
	@uv run semantic-release version --vcs-release

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linter (ruff)
	@echo "Running linter..."
	$(PYTHON_BIN) -m ruff check voice_to_text/ server.py transcribe.py
	@echo "✓ Linting complete"

lint-fix: ## Fix linting issues automatically
	@echo "Fixing linting issues..."
	$(PYTHON_BIN) -m ruff check --fix voice_to_text/ server.py transcribe.py

format: ## Format code with black
	@echo "Formatting code..."
	$(PYTHON_BIN) -m black voice_to_text/ server.py transcribe.py
	@echo "✓ Code formatted"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	$(PYTHON_BIN) -m black --check voice_to_text/ server.py transcribe.py

type-check: ## Run type checker (mypy)
	@echo "Running type checks..."
	$(PYTHON_BIN) -m mypy voice_to_text/

check-all: lint type-check ## Run all quality checks
	@echo "Running all quality checks..."

fix-all: lint-fix format ## Fix all auto-fixable issues
	@echo "Fixing all auto-fixable issues..."

# =============================================================================
# DOCKER
# =============================================================================
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker compose -f $(COMPOSE_FILE) build

docker-rebuild: ## Rebuild Docker image without cache
	@echo "Rebuilding Docker image (no cache)..."
	docker compose -f $(COMPOSE_FILE) build --no-cache

docker-up: ## Start Docker services
	@echo "Starting Docker services..."
	docker compose -f $(COMPOSE_FILE) up -d

docker-down: ## Stop Docker services
	@echo "Stopping Docker services..."
	docker compose -f $(COMPOSE_FILE) down

docker-logs: ## Show Docker logs
	docker compose -f $(COMPOSE_FILE) logs -f

# =============================================================================
# SERVER (Primary Workflow)
# =============================================================================
server: ## Start the API server (recommended)
	@echo "🚀 Starting FastAPI server..."
	@echo "📖 Swagger UI: http://localhost:8000/docs"
	docker compose -f $(COMPOSE_FILE) up

stop: ## Stop the running server
	@echo "Stopping server..."
	docker compose -f $(COMPOSE_FILE) down

docs: ## Open API documentation in browser
	@echo "Opening Swagger UI..."
	@xdg-open http://localhost:8000/docs 2>/dev/null || echo "Please open http://localhost:8000/docs in your browser"

# =============================================================================
# LEGACY CLI COMMANDS (Deprecated - Use API Instead)
# =============================================================================
run: ## Legacy: Basic transcription (use API instead)
	docker compose -f $(COMPOSE_FILE) run --rm whisper $(PYTHON_BIN) transcribe.py $(AUDIO) --model $(MODEL)

translate: ## Legacy: Transcription + Translation (use API instead)
	docker compose -f $(COMPOSE_FILE) run --rm whisper $(PYTHON_BIN) transcribe.py $(AUDIO) --model $(MODEL) --translate

diarize: ## Legacy: Transcription + Diarization (use API instead)
	docker compose -f $(COMPOSE_FILE) run --rm whisper $(PYTHON_BIN) transcribe.py $(AUDIO) --model $(MODEL) --diarize

all: ## Legacy: All features (use API instead)
	docker compose -f $(COMPOSE_FILE) run --rm whisper $(PYTHON_BIN) transcribe.py $(AUDIO) --model $(MODEL) --translate --diarize

# =============================================================================
# UTILITIES
# =============================================================================
clean: ## Clean up generated files
	@echo "Cleaning up generated files..."
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" \) -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -o -name ".coverage" \) -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ logs/ .venv/ .uv/ uv.lock media/transcripts/* 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-transcripts: ## Clean transcripts folder only
	@echo "Cleaning transcripts..."
	rm -rf media/transcripts/*
	@echo "✓ Transcripts cleaned"

shell: ## Open Python shell with app context
	@echo "Loading Python shell..."
	$(PYTHON_BIN) -i -c "from voice_to_text import transcribe; print('Voice-to-Text loaded!')"

logs: ## Show application logs
	@tail -f logs/app.log 2>/dev/null || echo "No log file found"

update: ## Update dependencies
	@echo "Updating dependencies..."
	uv sync --upgrade

freeze: ## Update lock file
	@echo "Updating lock file..."
	uv lock

list: ## List installed dependencies
	uv pip list

add: ## Add a new package (use PKG=name)
	@echo "Adding package: $(PKG)..."
	uv add $(PKG)

add-dev: ## Add a new dev package (use PKG=name)
	@echo "Adding dev package: $(PKG)..."
	uv add --dev $(PKG)

remove: ## Remove a package (use PKG=name)
	@echo "Removing package: $(PKG)..."
	uv remove $(PKG)

# =============================================================================
# CI/CD
# =============================================================================
security: ## Run security scan with bandit
	@echo "Running security scan..."
	@uv run bandit -r voice_to_text/ -f screen -v
	@echo "✓ Security scan complete"

ci: pre-commit-run security build ## Run CI pipeline checks
	@echo "Running CI pipeline..."

# =============================================================================
# INFO
# =============================================================================
info: ## Show project information
	@echo "Name: Voice-to-Text"
	@echo "Version: $$(grep __version__ voice_to_text/__init__.py | cut -d'"' -f2)"
	@echo "Python: $$(python --version)"
	@echo "Environment: $${ENVIRONMENT:-development}"
