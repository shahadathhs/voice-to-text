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

# Install dependencies from requirements.txt
ARG CACHEBUST=1
RUN pip install --no-cache-dir -r requirements.txt

# Create transcripts directory
RUN mkdir transcripts

# Copy the rest of the application code
COPY transcribe.py .
COPY server.py .
COPY voice_to_text/ ./voice_to_text/
COPY pyproject.toml .

EXPOSE 8000

# Define the default command (can be overridden)
CMD ["python", "transcribe.py", "--help"]

