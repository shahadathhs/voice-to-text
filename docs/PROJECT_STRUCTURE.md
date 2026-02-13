# Project structure

Quick reference for where to find things.

| Path | Purpose |
|------|--------|
| `voice_to_text/` | Main Python package |
| `voice_to_text/config.py` | Constants (WHISPER_BACKEND_DEFAULT, MIN_SEGMENT_MS, etc.) |
| `voice_to_text/io_utils.py` | `check_file`, `get_unique_filename`, `save_transcript` |
| `voice_to_text/diarization.py` | `overlap`, `assign_speaker_by_overlap`, `perform_diarization`, chunk building, smoothing |
| `voice_to_text/pipeline.py` | `_run_whisper`, `transcribe()` (orchestration) |
| `voice_to_text/cli.py` | `main()` — argparse and run |
| `voice_to_text/backends/` | Whisper backends (openai, transformers) |
| `transcribe.py` | CLI entrypoint (thin wrapper) |
| `server.py` | FastAPI app and `/transcribe` endpoint |
| `docs/` | ARCHITECTURE.md, PROJECT_STRUCTURE.md |
| `transcripts/` | Output directory for saved transcripts |
| `README.md` | User-facing quick start and options |

## File size

- **config.py** — small; constants only.
- **io_utils.py** — small; three functions.
- **diarization.py** — ~200 lines; all diarization logic.
- **pipeline.py** — ~100 lines; load + transcribe + translate/diarize.
- **cli.py** — ~60 lines; argparse and one transcribe call.
- **backends/** — two small modules (~30–50 lines each).
