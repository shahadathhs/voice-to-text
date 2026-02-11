# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (FFmpeg is required for Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to keep the image size small
# We install torch first to ensure it's cached properly if needed
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir openai-whisper setuptools-rust

# Copy the resten of the application code
COPY transcribe.py .

# Define the entry point for the container
ENTRYPOINT ["python", "transcribe.py"]

# Default command (can be overridden)
CMD ["--help"]
