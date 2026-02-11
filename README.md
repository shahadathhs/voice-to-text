# Simple Voice-to-Text System

A standalone, local voice-to-text system using OpenAI's Whisper. This version supports **Dual Output (Original + Translation)** and **Local-First Speaker Diarization** (no tokens required).

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **FFmpeg** (only if running locally without Docker).
- **Make** (installed by default on most Linux systems).

---

## 🚀 Quick Start with Makefile (Easiest)

We provide a `Makefile` to simplify the Docker commands.

### 1. Build the Image

```bash
make build
```

### 2. Run Basic Transcription

```bash
make run AUDIO=audio.wav
```

### 3. Translate to English (Includes Original Transcript)

```bash
make translate AUDIO=audio.wav
```

### 4. Speaker Diarization

To identify different speakers and label the transcript (`SPEAKER_00: Hello`):

```bash
make diarize AUDIO=audio.wav
```

### 5. All-in-one (Transcribe + Translate + Diarize)

```bash
make all AUDIO=audio.wav
```

---

## 🐳 Quick Start with Docker (Manual)

If you don't have `make` installed:

### 1. Build

```bash
docker compose build
```

### 2. Run

```bash
# Basic
docker compose run --rm whisper audio.wav

# Dual Output (Original + translation)
docker compose run --rm whisper audio.wav --translate

# Diarization (No token required!)
docker compose run --rm whisper audio.wav --diarize
```

---

## 📂 Output

- Transcripts are saved to `transcripts/`.
- If diarization is enabled, output follows the format: `SPEAKER_XX: [Text segment]`
- If translation is enabled, files contain both the **Original language** and the **English translation**.

---

## 💾 Offline Mode & Model Caching

The system automatically caches models in the `models/` directory. Once a model is downloaded, you can run the system without an internet connection.

### How to use Offline:

1.  **First Run**: Run the system once while connected to the internet to download the models.
2.  **Verify**: Check that the `models/` folder contains Whisper and SpeechBrain data.
3.  **Run Offline**: Subsequent runs will use the cached files automatically.

---

## 🛠️ Local Installation (Manual)

1. **Install FFmpeg**: `sudo apt update && sudo apt install ffmpeg libsndfile1`
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   python3 transcribe.py audio.wav --translate --diarize
   ```

---

## ⚙️ Options

- `--model`: Choose model size (`tiny`, `base`, `small`, `medium`, `large`). Default: `base`.
- `--translate`: Translate non-English audio to English.
- `--diarize`: Enable speaker diarization (uses SpeechBrain, no token needed).

---

## ❓ Troubleshooting

### 403 Client Error during download

If you see a 403 error, ensure you have a stable internet connection for the first run. The system uses open-source models from SpeechBrain and OpenAI. No Hugging Face tokens are required.
