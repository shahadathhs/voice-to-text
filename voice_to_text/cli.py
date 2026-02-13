"""CLI entrypoint for voice-to-text."""

import argparse

from voice_to_text.config import WHISPER_BACKEND_DEFAULT
from voice_to_text.io_utils import check_file, get_unique_filename, save_transcript
from voice_to_text.pipeline import transcribe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Voice-to-Text using Whisper & SpeechBrain (all local)."
    )
    parser.add_argument("input", help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size (tiny, base, small, medium, large). Default: base",
    )
    parser.add_argument("--translate", action="store_true", help="Translate to English")
    parser.add_argument("--diarize", action="store_true", help="Speaker diarization (SpeechBrain)")
    parser.add_argument(
        "--diarize-threshold",
        type=float,
        default=0.35,
        help="Clustering distance (lower=more speakers). Default: 0.35",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        metavar="N",
        help="Fix number of speakers (overrides --diarize-threshold)",
    )
    parser.add_argument(
        "--use-silhouette",
        action="store_true",
        help="Estimate number of speakers from embeddings",
    )
    parser.add_argument(
        "--whisper-backend",
        choices=("openai-whisper", "transformers"),
        default=WHISPER_BACKEND_DEFAULT,
        help="Whisper backend. Default: openai-whisper",
    )
    args = parser.parse_args()

    audio_file = check_file(args.input)
    transcript = transcribe(
        audio_file,
        model=args.model,
        translate=args.translate,
        diarize=args.diarize,
        diarize_threshold=args.diarize_threshold,
        max_speakers=args.max_speakers,
        whisper_backend=args.whisper_backend,
        use_silhouette=args.use_silhouette,
    )

    print("\n" + "=" * 20 + " TRANSCRIPT " + "=" * 20)
    print(transcript)
    print("=" * 52 + "\n")

    output_filename = get_unique_filename(audio_file)
    saved_path = save_transcript(transcript, output_filename)
    print(f"[*] Transcript saved to: {saved_path}")
