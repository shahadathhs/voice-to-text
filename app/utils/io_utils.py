"""File I/O and path utilities."""

import os
import sys
from datetime import datetime
from pathlib import Path


def check_file(path: str) -> str:
    """Validate that the file exists; exit with error message if not."""
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        sys.exit(1)
    return path


def get_unique_filename(audio_path: str) -> str:
    """Generate a unique transcript filename from the audio path and current time."""
    base_name = Path(audio_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.txt"


def save_transcript(text: str, filename: str) -> Path:
    """Save transcript text to transcripts/<filename>; return the file path."""
    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / filename
    file_path.write_text(text, encoding="utf-8")
    return file_path
