"""API routes for Voice-to-Text application."""

from app.api import docs  # Documentation routes
from app.api.routes import router

__all__ = ["docs", "router"]
