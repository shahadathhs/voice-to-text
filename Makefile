# Voice-to-Text API - Makefile
# Modern API-focused build system

# Docker Variables
DOCKER_IMAGE := voice-to-text:latest
COMPOSE_FILE := compose.yaml

# Python Variables
VENV := .venv

# Detect if UV is available
HAS_UV := $(shell command -v uv 2> /dev/null && echo "yes" || echo "no")

# Use UV if available, otherwise use standard Python
ifeq ($(HAS_UV),yes)
    PYTHON_BIN := uv run python
    PIP_INSTALL := uv sync
    VENV_CMD := uv venv
    PACKAGE_CMD := uv add
    PACKAGE_DEV_CMD := uv add --dev
    RUN_CMD := uv run
else
    PYTHON_BIN := $(VENV)/bin/python
    PIP_INSTALL := $(VENV)/bin/pip install -e .
    VENV_CMD := python3 -m venv
    PACKAGE_CMD := $(VENV)/bin/pip install
    PACKAGE_DEV_CMD := $(VENV)/bin/pip install
    RUN_CMD := $(VENV)/bin/python -m
endif

# Phony targets
.PHONY: help setup install-deps venv install reset-venv
.PHONY: pre-commit-install pre-commit-run pre-commit-update
.PHONY: build dev dev-verbose prod
.PHONY: lint lint-fix format format-check type-check check-all fix-all
.PHONY: docker-build docker-down docker-rebuild docker-ps
.PHONY: clean shell logs update freeze list add add-dev remove ci security info
.PHONY: server stop restart logs docs status

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "🎙️  Voice-to-Text API - Available Commands"
	@echo ""
	@echo "🚀 Setup Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;32m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📖 Quick Start:"
	@echo "  ./setup.sh              - Complete setup (recommended)"
	@echo "  make server             - Start API server"
	@echo "  make docs               - Open API documentation"
	@echo ""
	@echo "💡 Tip: Run ./setup.sh to install UV, FFmpeg, and all dependencies"

# =============================================================================
# SETUP
# =============================================================================
install-deps: ## Install all system dependencies (UV, FFmpeg, Python packages)
	@echo "🚀 Running complete setup..."
	@./setup.sh

venv: ## Create virtual environment
	@echo "Creating virtual environment..."
	@rm -rf $(VENV) 2>/dev/null || true
	@echo "Using: $(VENV_CMD)"
	@$(VENV_CMD) $(VENV)
	@echo "✓ Virtual environment created at $(VENV)"
	@echo "Note: Using $(PYTHON_BIN)"

reset-venv: ## Force reset virtual environment (fix permission issues)
	@echo "Resetting virtual environment..."
	@rm -rf $(VENV) .uv uv.lock 2>/dev/null || true
	@$(VENV_CMD) $(VENV)
	@$(PIP_INSTALL)
	@echo "✓ Virtual environment reset complete"

install: ## Install Python dependencies only
	@echo "Installing Python dependencies..."
	@echo "Using: $(PIP_INSTALL)"
	@$(PIP_INSTALL)
	@echo "✓ Dependencies installed"

setup: venv install pre-commit-install ## Full setup (venv + install + hooks)
	@echo "✓ Setup complete!"
	@echo "Python: $(PYTHON_BIN)"
	@echo "Virtual Environment: $(VENV)"
	@echo ""
	@echo "🎉 Ready to use!"
	@echo ""
	@echo "Next steps:"
	@echo "  make server             - Start API server"
	@echo "  make dev                - Run in development mode"
	@echo "  make docs               - Open API documentation"

pre-commit-install: ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	@$(RUN_CMD) pre-commit install
	@echo "✓ Pre-commit hooks installed"

pre-commit-run: ## Run all pre-commit hooks manually
	@echo "Running pre-commit hooks..."
	@$(RUN_CMD) pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	@echo "Updating pre-commit hooks..."
	@$(RUN_CMD) pre-commit autoupdate
	@echo "✓ Pre-commit hooks updated"

# =============================================================================
# RUNNING
# =============================================================================
build: ## Build distribution packages
	@echo "Building distribution packages..."
	@$(RUN_CMD) build
	@echo "✓ Built packages in dist/"

dev: ## Run in development mode with hot reload
	@echo "Starting development server..."
	@$(RUN_CMD) uvicorn server:app --reload --host 0.0.0.0 --port 8000

dev-verbose: ## Run in development mode with verbose logging
	@echo "Starting development server (verbose)..."
	@$(RUN_CMD) uvicorn server:app --reload --log-level debug

prod: ## Run in production mode
	@echo "Starting production server..."
	@$(RUN_CMD) uvicorn server:app --host 0.0.0.0 --port $${PORT:-8000} --workers 4

# =============================================================================
# CODE QUALITY
# =============================================================================
lint: ## Run linter (ruff)
	@echo "Running linter..."
	@$(RUN_CMD) ruff check app/
	@echo "✓ Linting complete"

lint-fix: ## Fix linting issues automatically
	@echo "Fixing linting issues..."
	@$(RUN_CMD) ruff check --fix app/
	@echo "✓ Linting issues fixed"

format: ## Format code with black
	@echo "Formatting code..."
	@$(RUN_CMD) black app/
	@echo "✓ Code formatted"

format-check: ## Check if code needs formatting
	@echo "Checking code formatting..."
	@$(RUN_CMD) black --check app/

type-check: ## Run type checker (mypy)
	@echo "Running type checks..."
	@$(RUN_CMD) mypy app/

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

docker-down: ## Stop Docker services
	@echo "Stopping Docker services..."
	docker compose -f $(COMPOSE_FILE) down

docker-rebuild: ## Rebuild Docker image without cache
	@echo "Rebuilding Docker image (no cache)..."
	docker compose -f $(COMPOSE_FILE) build --no-cache

docker-ps: ## Show Docker containers
	@echo "Docker containers:"
	docker compose -f $(COMPOSE_FILE) ps

# =============================================================================
# SERVER (Primary Workflow)
# =============================================================================
server: ## Start the API server (recommended)
	@echo "🚀 Starting FastAPI server..."
	@echo "📖 Swagger UI: http://localhost:8000/docs"
	@echo "🎙️  Voice-to-Text API is ready!"
	docker compose -f $(COMPOSE_FILE) up

stop: ## Stop the running server
	@echo "Stopping server..."
	docker compose -f $(COMPOSE_FILE) down

restart: ## Restart the server
	@echo "Restarting server..."
	docker compose -f $(COMPOSE_FILE) down
	docker compose -f $(COMPOSE_FILE) up

logs: ## Show server logs
	docker compose -f $(COMPOSE_FILE) logs -f

docs: ## Open API documentation in browser
	@echo "📖 Opening Swagger UI..."
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "Alternative docs: http://localhost:8000/redoc"
	@xdg-open http://localhost:8000/docs 2>/dev/null || echo "Please open http://localhost:8000/docs in your browser"

status: ## Check server status
	@echo "🔍 Checking server status..."
	docker compose -f $(COMPOSE_FILE) ps

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
	@$(PYTHON_BIN) -i -c "from app import transcribe; print('Voice-to-Text loaded!')"

update: ## Update dependencies
	@echo "Updating dependencies..."
ifeq ($(HAS_UV),yes)
	uv sync --upgrade
else
	$(VENV)/bin/pip install --upgrade -r requirements.txt
endif

freeze: ## Update lock file
	@echo "Updating lock file..."
ifeq ($(HAS_UV),yes)
	uv lock
else
	$(VENV)/bin/pip freeze > requirements.txt
endif

list: ## List installed dependencies
	@$(PYTHON_BIN) -m pip list

add: ## Add a new package (use PKG=name)
	@echo "Adding package: $(PKG)..."
	@$(PACKAGE_CMD) $(PKG)

add-dev: ## Add a new dev package (use PKG=name)
	@echo "Adding dev package: $(PKG)..."
	@$(PACKAGE_DEV_CMD) $(PKG)

remove: ## Remove a package (use PKG=name)
	@echo "Removing package: $(PKG)..."
ifeq ($(HAS_UV),yes)
	uv remove $(PKG)
else
	$(VENV)/bin/pip uninstall $(PKG)
endif

# =============================================================================
# CI/CD
# =============================================================================
security: ## Run security scan with bandit
	@echo "Running security scan..."
	@$(RUN_CMD) bandit -r app/ -f screen -v
	@echo "✓ Security scan complete"

ci: pre-commit-run security build ## Run CI pipeline checks
	@echo "Running CI pipeline..."

# =============================================================================
# INFO
# =============================================================================
info: ## Show project information
	@echo "🎙️  Voice-to-Text API"
	@echo "Version: $$(grep __version__ app/__init__.py | cut -d'"' -f2)"
	@echo "Python: $$(python3 --version)"
	@echo "Environment: $${ENVIRONMENT:-development}"
	@echo ""
	@echo "Package Manager:"
	@echo "  UV Available: $(HAS_UV)"
	@echo "  Python Bin: $(PYTHON_BIN)"
	@echo "  Virtual Env: $(VENV)"
	@echo ""
	@echo "Quick Start:"
	@echo "  ./setup.sh              - Complete setup (recommended)"
	@echo "  make install-deps       - Install UV, FFmpeg, and dependencies"
	@echo "  make setup              - Quick setup (Python only)"
	@echo "  make server             - Start API server"
	@echo "  make dev                - Development mode"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs               - Open documentation hub"
	@echo "  /docs-hub              - Choose documentation viewer"
	@echo "  /rapidoc              - RapiDoc (recommended)"
	@echo "  /docs                 - Swagger UI"
	@echo "  /redoc                - ReDoc"
	@echo "  make docs               - Open API documentation"
