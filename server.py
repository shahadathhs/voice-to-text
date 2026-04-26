"""FastAPI server entry point - delegates to app.main."""

from app.main import app

__all__ = ["app"]

# This allows running with:
#   uvicorn server:app --reload
#   python -m uvicorn server:app --reload
#   make dev


