"""CLI interface for voice-to-text transcription."""

import argparse
import sys
from pathlib import Path

from app.core.config import settings
from app.core.errors import AudioFileError, TranscriptionError
from app.core.logger import logger, setup_logging
from app.services.transcriber import transcription_service


def cmd_list(args: argparse.Namespace) -> int:
    """List audio or transcript files.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from app.core.config import ensure_directories

    ensure_directories()

    if args.type == "audio" or args.type == "all":
        print("📁 Audio files in media/audio/:")
        audio_dir = Path(settings.audio_dir)
        if audio_dir.exists():
            files = list(audio_dir.glob("*"))
            if files:
                for f in sorted(files):
                    if f.is_file():
                        size = f.stat().st_size / (1024 * 1024)  # MB
                        print(f"  {f.name} ({size:.1f} MB)")
            else:
                print("  No audio files found")
        else:
            print("  Audio directory not found")
        print()

    if args.type == "transcripts" or args.type == "all":
        print("📄 Transcript files in media/transcripts/:")
        transcript_dir = Path(settings.transcript_dir)
        if transcript_dir.exists():
            files = list(transcript_dir.glob("*.txt"))
            if files:
                for f in sorted(files)[-10:]:  # Last 10 files
                    size = f.stat().st_size / 1024  # KB
                    mtime = f.stat().st_mtime
                    from datetime import datetime

                    mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"  {f.name} ({size:.1f} KB, {mod_time})")
            else:
                print("  No transcript files found")
        else:
            print("  Transcript directory not found")

    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Clean transcript files.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from app.core.config import ensure_directories

    ensure_directories()

    transcript_dir = Path(settings.transcript_dir)
    if not transcript_dir.exists():
        print("No transcript directory found")
        return 1

    if args.all:
        # Clean all transcripts
        files = list(transcript_dir.glob("*.txt"))
        if not files:
            print("No transcript files to clean")
            return 0

        print(f"Found {len(files)} transcript file(s)")
        if args.force or input("Delete all transcripts? (y/N): ").lower() == "y":
            for f in files:
                f.unlink()
                print(f"  Deleted: {f.name}")
            print("✓ All transcripts cleaned")
    else:
        # Clean by pattern
        pattern = args.pattern
        matching_files = list(transcript_dir.glob(f"*{pattern}*.txt"))
        if not matching_files:
            print(f"No transcripts matching '{pattern}' found")
            return 0

        print(f"Found {len(matching_files)} transcript(s) matching '{pattern}'")
        if (
            args.force
            or input(f"Delete {len(matching_files)} file(s)? (y/N): ").lower() == "y"
        ):
            for f in matching_files:
                f.unlink()
                print(f"  Deleted: {f.name}")
            print(f"✓ Cleaned {len(matching_files)} transcript(s)")

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show CLI configuration and status.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from app.core.config import ensure_directories

    ensure_directories()

    print("📊 CLI Configuration")
    print("=" * 50)
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Log Level: {settings.log_level}")
    print()
    print("📁 Directories")
    print("-" * 50)
    print(f"Audio:      {settings.audio_dir}")
    print(f"Uploads:    {settings.uploads_dir}")
    print(f"Transcripts:{settings.transcript_dir}")
    print(f"Cache:      {settings.model_cache_dir}")
    print()
    print("🎙️  Whisper Settings")
    print("-" * 50)
    print(f"Model:      {settings.whisper_model}")
    print(f"Backend:    {settings.whisper_backend}")
    print(f"Device:     {settings.whisper_device}")
    print(f"Translation:{settings.enable_translation}")
    print(f"Diarization:{settings.enable_diarization}")
    print()
    print("📈 Directory Status")
    print("-" * 50)
    for dir_name, dir_path in [
        ("Audio", settings.audio_dir),
        ("Uploads", settings.uploads_dir),
        ("Transcripts", settings.transcript_dir),
        ("Cache", settings.model_cache_dir),
    ]:
        path = Path(dir_path)
        if path.exists():
            count = len(list(path.glob("*")))
            print(f"✓ {dir_name:12} ({count} files)")
        else:
            print(f"✗ {dir_name:12} (missing)")

    return 0


def cmd_dirs(args: argparse.Namespace) -> int:
    """Ensure all directories exist.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    from app.core.config import ensure_directories

    ensure_directories()
    print("✓ All CLI directories created/verified")

    if args.verbose:
        print("\nCreated directories:")
        print(f"  {settings.audio_dir}")
        print(f"  {settings.uploads_dir}")
        print(f"  {settings.transcript_dir}")
        print(f"  {settings.model_cache_dir}")

    return 0


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

  # Use custom media directory (auto-creates subdirectories)
  %(prog)s /path/to/audio.wav --media-dir /custom/media

  # Disable automatic directory creation
  %(prog)s media/audio/sample.wav --no-ensure-dirs

  # Use different model
  %(prog)s media/audio/sample.wav --model small

Directory Structure:
  The CLI automatically creates media directories if needed:
  - media/audio/      - Default audio files (for CLI usage)
  - media/uploads/    - Uploaded audio files (for API usage)
  - media/transcripts/ - Transcription output

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
        help="Media directory path (default: media/). Creates audio/, uploads/, and transcripts/ subdirectories.",
    )

    parser.add_argument(
        "--ensure-dirs",
        action="store_true",
        default=True,
        help="Ensure media directories exist (default: enabled). Use --no-ensure-dirs to disable.",
    )

    parser.add_argument(
        "--no-ensure-dirs",
        dest="ensure_dirs",
        action="store_false",
        help="Disable automatic directory creation",
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
    if args.diarize_threshold is not None and not 0.0 <= args.diarize_threshold <= 1.0:
        raise ValueError("--diarize-threshold must be between 0.0 and 1.0")

    # Validate max speakers
    if args.max_speakers is not None and args.max_speakers < 1:
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

        # Ensure directories exist (always run unless explicitly disabled)
        if args.ensure_dirs:
            from app.core.config import ensure_directories

            ensure_directories()
            logger.info("✓ Ensured all required directories exist")

            if args.media_dir:
                media_path = args.media_dir
                media_path.mkdir(parents=True, exist_ok=True)
                (media_path / "audio").mkdir(exist_ok=True)
                (media_path / "uploads").mkdir(exist_ok=True)
                (media_path / "transcripts").mkdir(exist_ok=True)

                # Update settings
                settings.media_dir = media_path
                settings.audio_dir = media_path / "audio"
                settings.uploads_dir = media_path / "uploads"
                settings.transcript_dir = media_path / "transcripts"

                logger.info(f"Using custom media directory: {media_path}")

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
    if hasattr(args, "debug") and args.debug:
        settings.log_level = "DEBUG"
        settings.debug = True

    setup_logging()

    # Route to appropriate command handler
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "clean":
        return cmd_clean(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "dirs":
        return cmd_dirs(args)
    elif args.command == "transcribe":
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
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
