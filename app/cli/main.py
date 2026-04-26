"""CLI interface for voice-to-text transcription."""

import argparse
import sys
from pathlib import Path

from app.core.config import settings
from app.core.errors import AudioFileError, TranscriptionError
from app.core.logger import logger, setup_logging
from app.services.transcriber import transcription_service


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Voice-to-Text: AI-powered audio transcription using OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription (audio from media/audio/)
  %(prog)s media/audio/sample.wav

  # With translation
  %(prog)s media/audio/sample.wav --translate

  # With speaker diarization
  %(prog)s media/audio/meeting.mp3 --diarize

  # All features
  %(prog)s media/audio/conversation.wav --translate --diarize --max-speakers 2

  # Use custom media directory
  %(prog)s /path/to/audio.wav --media-dir /custom/media

  # Use different model
  %(prog)s media/audio/sample.wav --model small

For more information, see: https://github.com/shahadathhs/voice-to-text
        """,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Path to audio file (supported: wav, mp3, ogg, m4a, flac, aac)",
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,  # Use settings default
        help="Whisper model size (default: base from settings)",
    )

    parser.add_argument(
        "--backend",
        choices=["openai", "transformers"],
        default=None,  # Use settings default
        help="Whisper backend (default: openai from settings)",
    )

    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate non-English audio to English",
    )

    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (identify different speakers)",
    )

    parser.add_argument(
        "--diarize-threshold",
        type=float,
        default=None,
        metavar="N",
        help="Clustering threshold for diarization (0.0-1.0, default: 0.35). Lower = more speakers",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        metavar="N",
        help="Fixed number of speakers (overrides --diarize-threshold)",
    )

    parser.add_argument(
        "--use-silhouette",
        action="store_true",
        help="Estimate number of speakers from embeddings",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output file path (default: auto-generated in media/transcripts/)",
    )

    parser.add_argument(
        "--media-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Media directory path (default: media/). Will be created if it doesn't exist.",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    if not args.input.exists():
        raise ValueError(f"Input file not found: {args.input}")

    # Check if input file is a file (not directory)
    if not args.input.is_file():
        raise ValueError(f"Input is not a file: {args.input}")

    # Validate diarize threshold
    if args.diarize_threshold is not None:
        if not 0.0 <= args.diarize_threshold <= 1.0:
            raise ValueError("--diarize-threshold must be between 0.0 and 1.0")

    # Validate max speakers
    if args.max_speakers is not None:
        if args.max_speakers < 1:
            raise ValueError("--max-speakers must be at least 1")


async def transcribe_async(args: argparse.Namespace) -> int:
    """Transcribe audio file asynchronously.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        # Validate arguments
        validate_args(args)

        if args.media_dir:
            media_path = args.media_dir
            media_path.mkdir(parents=True, exist_ok=True)
            (media_path / "audio").mkdir(exist_ok=True)
            (media_path / "transcripts").mkdir(exist_ok=True)

            # Update settings
            settings.media_dir = media_path
            settings.audio_dir = media_path / "audio"
            settings.transcript_dir = media_path / "transcripts"

            logger.info(f"Using media directory: {media_path}")

        # Initialize service
        logger.info("Initializing transcription service...")
        transcription_service.initialize()

        # Perform transcription
        logger.info(f"Transcribing: {args.input}")

        result = await transcription_service.transcribe_file(
            audio_file=args.input,
            translate=args.translate or settings.enable_translation,
            diarize=args.diarize or settings.enable_diarization,
            diarize_threshold=args.diarize_threshold or settings.diarize_threshold,
            max_speakers=args.max_speakers or settings.max_speakers,
            use_silhouette=args.use_silhouette or settings.use_silhouette,
        )

        # Print result
        print("\n" + "=" * 80)
        print("TRANSCRIPTION RESULT")
        print("=" * 80)
        print(result["transcript"])
        print("=" * 80)
        print(f"\nSaved to: {result['saved_to']}")
        print(f"Metadata: {result['metadata']}")
        print("=" * 80)

        return 0

    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        return 1

    except AudioFileError as e:
        logger.error(f"Audio file error: {e.message}")
        return 1

    except TranscriptionError as e:
        logger.error(f"Transcription error: {e.message}")
        return 1

    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    # Parse arguments
    args = parse_args()

    # Configure logging
    if args.debug or args.verbose:
        settings.log_level = "DEBUG"
        settings.debug = True

    setup_logging()

    # Print banner
    logger.info(f"{settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Run transcription
    import asyncio

    exit_code = asyncio.run(transcribe_async(args))

    # Cleanup
    try:
        transcription_service.cleanup()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
