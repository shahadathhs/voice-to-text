"""Configuration constants for transcription and diarization."""

# Whisper backend: "openai-whisper" (default) or "transformers"
WHISPER_BACKEND_DEFAULT = "openai-whisper"

# Diarization: sub-segment sliding window
SUBSEGMENT_WINDOW_S = 1.5
SUBSEGMENT_STRIDE_S = 0.5
SUBSEGMENT_MIN_DURATION_S = 3.0
SMOOTHING_MAX_DURATION_S = 2.0

# SpeechBrain ECAPA: 16 kHz mono; minimum duration for reliable embedding (ms)
DIARIZE_SAMPLE_RATE = 16000
MIN_SEGMENT_MS = 1500
MIN_CHUNK_MS = int(SUBSEGMENT_WINDOW_S * 1000)
