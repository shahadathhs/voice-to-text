# Simple Voice-to-Text System

A standalone, local voice-to-text system using OpenAI's Whisper. This version supports **Translation**, **Speaker Diarization**, and automatic **Output Persistence**.

## Prerequisites

- **Docker** and **Docker Compose** installed.
- **FFmpeg** (only if running locally without Docker).

---

## 🚀 Quick Start with Docker

### 1. Build the Image

```bash
docker compose build
```

### 2. Run Basic Transcription

```bash
docker compose run --rm whisper audio.wav
```

### 3. Translate to English

If the audio is in another language and you want English text:

```bash
docker compose run --rm whisper audio.wav --translate
```

### 4. Speaker Diarization (Who is saying what?)

To identify different speakers in the audio:

1. **Get a Hugging Face Token**:
   - Create an account on [Hugging Face](https://huggingface.co/).
   - Go to [Settings -> Access Tokens](https://huggingface.co/settings/tokens) and create a "Read" token.
   - **Important**: Accept the user conditions for the following models on Hugging Face:
     - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
     - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. **Run with Token**:

```bash
docker compose run --rm whisper audio.wav --diarize --hf-token YOUR_HF_TOKEN
```

---

## 📂 Output

All transcripts are automatically saved to the `transcripts/` folder with a unique timestamp:
`transcripts/audio_20231027_120000.txt`

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
