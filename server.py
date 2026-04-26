"""FastAPI server entry point for voice-to-text."""

from voice_to_text.api.routes import app

# Export the FastAPI app
__all__ = ["app"]

# This allows running with: uvicorn server:app
# or: uvicorn voice_to_text.api.routes:app
# or: python -m uvicorn server:app

