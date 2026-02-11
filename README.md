# Simple Voice-to-Text System

A standalone, local voice-to-text system using OpenAI's Whisper. This version is containerized with **Docker** for ease of use and to avoid local environment issues.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **FFmpeg** (only if running locally without Docker).

---

## 🚀 Quick Start with Docker

The easiest way to run the system without worrying about Python dependencies or system libraries.

### 1. Build the Image

```bash
docker compose build
```

### 2. Run Transcription

Place your audio file (e.g., `audio.wav`) in the project folder and run:

```bash
docker compose run --rm whisper audio.wav
```

---

## 🛠️ Local Installation (Manual)

If you prefer to run it directly on your host machine:

1. **Install FFmpeg**:

   ```bash
   sudo apt update && sudo apt install ffmpeg
   ```

2. **Setup Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   python3 transcribe.py audio.wav
   ```

---

## ⚙️ Options

You can specify the model size (`tiny`, `base`, `small`, `medium`, `large`):

```bash
# Docker
docker compose run --rm whisper audio.wav --model medium

# Local
python3 transcribe.py audio.wav --model medium
```

## 📄 Output

The transcript will be printed directly to your terminal.
