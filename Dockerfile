# Use a standard Python image (non-slim) for better build stability with large libraries
FROM python:3.10-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version to prevent installation bugs
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install dependencies in stages to reduce memory pressure
RUN pip install --no-cache-dir "numpy<2.0" torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Create transcripts directory
RUN mkdir transcripts

# Copy the rest of the application code
COPY transcribe.py .

# Define the entry point for the container
ENTRYPOINT ["python", "transcribe.py"]

# Default command (can be overridden)
CMD ["--help"]
