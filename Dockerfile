FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./

# Install dependencies using uv
RUN uv sync --no-dev

# Copy application source
COPY . .

# Create media directories
RUN mkdir -p /app/media/audio /app/media/transcripts /app/model-cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Start command
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
