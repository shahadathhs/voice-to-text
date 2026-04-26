#!/bin/bash
set -e

echo "🚀 Voice-to-Text - Complete Setup"
echo "================================"
echo ""

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# Install UV package manager
echo ""
echo "📦 Installing UV package manager..."
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    # Verify UV installation
    if command -v uv &> /dev/null; then
        echo "✓ UV installed successfully"
        uv --version
    else
        echo "❌ UV installation failed"
        echo "Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
else
    echo "✓ UV already installed"
    uv --version
fi

# Install FFmpeg (required for audio processing)
echo ""
echo "🎵 Installing FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing..."

    if [[ "$OS" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            echo "Installing FFmpeg via apt..."
            sudo apt-get update
            sudo apt-get install -y ffmpeg libsndfile1
        elif command -v yum &> /dev/null; then
            echo "Installing FFmpeg via yum..."
            sudo yum install -y ffmpeg libsndfile
        elif command -v pacman &> /dev/null; then
            echo "Installing FFmpeg via pacman..."
            sudo pacman -S --noconfirm ffmpeg libsndfile
        else
            echo "❌ Unable to detect package manager. Please install FFmpeg manually:"
            echo "  Ubuntu/Debian: sudo apt-get install ffmpeg libsndfile1"
            echo "  Fedora/RHEL: sudo yum install ffmpeg libsndfile"
            echo "  Arch: sudo pacman -S ffmpeg libsndfile"
            exit 1
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            echo "Installing FFmpeg via Homebrew..."
            brew install ffmpeg libsndfile
        else
            echo "❌ Homebrew not found. Please install:"
            echo "  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    fi

    # Verify FFmpeg installation
    if command -v ffmpeg &> /dev/null; then
        echo "✓ FFmpeg installed successfully"
        ffmpeg -version | head -n 1
    else
        echo "❌ FFmpeg installation failed"
        exit 1
    fi
else
    echo "✓ FFmpeg already installed"
    ffmpeg -version | head -n 1
fi

# Create virtual environment and install dependencies
echo ""
echo "🐍 Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install Python dependencies
echo ""
echo "📚 Installing Python dependencies..."
uv sync
echo "✓ Python dependencies installed"

# Install pre-commit hooks
echo ""
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install
echo "✓ Pre-commit hooks installed"

# Setup media directories
echo ""
echo "📁 Setting up media directories..."
mkdir -p media/audio
mkdir -p media/transcripts
echo "✓ Media directories created"

# Setup environment file
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example"
    echo "  ⚠️  Please review and update .env with your settings"
else
    echo "✓ .env already exists"
fi

# Run basic checks
echo ""
echo "🔍 Running basic checks..."
uv run python -c "import fastapi; import whisper; import speechbrain" 2>/dev/null && echo "✓ All core dependencies working" || echo "⚠️  Some dependencies may need attention"

# Final summary
echo ""
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "Installed components:"
echo "  • UV package manager: $(uv --version)"
echo "  • FFmpeg: $(ffmpeg -version | head -n 1)"
echo "  • Python: $(python3 --version)"
echo "  • Virtual environment: .venv"
echo ""
echo "🚀 Ready to use!"
echo ""
echo "Quick start commands:"
echo "  make server    - Start API server with Docker"
echo "  make dev       - Run in development mode"
echo "  make docs      - Open API documentation"
echo ""
echo "For more commands: make help"
