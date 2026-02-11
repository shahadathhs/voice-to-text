# Simple Voice-to-Text System

A standalone, local voice-to-text system using OpenAI's Whisper. This version supports **Dual Output (Original + Translation)**, **Speaker Diarization**, and simplified usage with **Makefile**.

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

1. **Get a Hugging Face Token**:
   - Create an account on [Hugging Face](https://huggingface.co/).
   - Go to [Settings -> Access Tokens](https://huggingface.co/settings/tokens) and create a "Read" token.
   - **Important**: Accept the user conditions for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).
2. **Setup .env file**:
   - Create a `.env` file from the example:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and paste your Hugging Face token:
     ```bash
     HF_TOKEN=your_token_here
     ```
3. **Run with Diarization**:

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

# Diarization
docker compose run --rm whisper audio.wav --diarize --hf-token YOUR_TOKEN
```

---

## 📂 Output

- Transcripts are saved to `transcripts/`.
- If diarization is enabled, output follows the format: `SPEAKER_XX: [Text segment]`
- If translation is enabled, files contain both the **Original language** and the **English translation**, both labeled with speakers if requested.

---

## 🛠️ Local Installation (Manual)

1. **Install FFmpeg**: `sudo apt update && sudo apt install ffmpeg`
2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Run**:
   ```bash
   python3 transcribe.py audio.wav --translate --diarize --hf-token YOUR_TOKEN
   ```

---

## ⚙️ Options

- `--model`: Choose model size (`tiny`, `base`, `small`, `medium`, `large`). Default: `base`.
- `--translate`: Translate non-English audio to English.
- `--diarize`: Enable speaker diarization.
- `--hf-token`: Your Hugging Face access token (required for diarization).
