"""Middleware for enhanced error handling and request processing."""

from collections.abc import Callable
from typing import cast

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.errors import AppError
from app.core.logger import logger
from app.core.response import ResponseBuilder


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and handle any uncaught exceptions.

        Args:
            request: Incoming request
            call_next: Next middleware or route handler

        Returns:
            Response with proper error handling
        """
        try:
            response = await call_next(request)
            return cast(Response, response)
        except AppError as e:
            # Handle known application exceptions
            logger.warning(f"Application error: {e.message}")
            return cast(
                Response,
                JSONResponse(
                    content=ResponseBuilder.error(
                        message=e.message,
                        status_code=e.status_code,
                        details=e.details,
                        errors=e.errors if e.errors else None,
                    ).model_dump(),
                    status_code=e.status_code,
                ),
            )
        except Exception as e:
            # Handle unexpected exceptions
            logger.exception(f"Unhandled exception: {e}")
            return cast(
                Response,
                JSONResponse(
                    content=ResponseBuilder.internal_server_error(
                        message="Internal server error",
                        details={"error": str(e)} if settings.debug else None,
                    ).model_dump(),
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                ),
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Log request and response information.

        Args:
            request: Incoming request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        logger.info(f"{request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Log response status
        logger.info(
            f"{request.method} {request.url.path} - Status: {response.status_code}"
        )

        return cast(Response, response)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for adding request context."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Add request context for better error tracking.

        Args:
            request: Incoming request
            call_next: Next middleware or route handler

        Returns:
            Response
        """
        request_id = request.headers.get("X-Request-ID", "unknown")
        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return cast(Response, response)
