"""API routes for Voice-to-Text application."""

from app.api.routes import router
from app.api import docs  # Documentation routes

__all__ = ["router", "docs"]
