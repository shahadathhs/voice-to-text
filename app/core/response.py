"""Response Builder - Standardized API response creation."""

from typing import Any, TypeVar

from fastapi import status

from app.schemas.base import DataResponse, ErrorResponse, MetaData, PaginatedResponse

T = TypeVar("T")


class ResponseBuilder:
    """Helper class for building standardized API responses."""

    @staticmethod
    def success(
        data: T,
        message: str = "Success",
        status_code: int = status.HTTP_200_OK,
        metadata: MetaData | None = None,
    ) -> DataResponse[T]:
        """
        Create a success response.

        Args:
            data: Response data
            message: Success message
            status_code: HTTP status code
            metadata: Optional metadata

        Returns:
            DataResponse: Formatted response
        """
        return DataResponse[T](
            status_code=status_code,
            success=True,
            message=message,
            data=data,
            metadata=metadata,
        )

    @staticmethod
    def created(
        data: T,
        message: str = "Resource created successfully",
    ) -> DataResponse[T]:
        """Create a 201 Created response."""
        return ResponseBuilder.success(
            data=data,
            message=message,
            status_code=status.HTTP_201_CREATED,
        )

    @staticmethod
    def no_content(
        message: str = "Request processed successfully",
    ) -> DataResponse[None]:
        """Create a 204 No Content response."""
        return DataResponse[None](
            status_code=status.HTTP_204_NO_CONTENT,
            success=True,
            message=message,
            data=None,
            metadata=None,
        )

    @staticmethod
    def error(
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        errors: list[Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ErrorResponse:
        """
        Create an error response.

        Args:
            message: Error message
            status_code: HTTP status code
            errors: Optional list of errors
            details: Optional error details

        Returns:
            ErrorResponse: Formatted error response
        """
        return ErrorResponse(
            status_code=status_code,
            success=False,
            message=message,
            errors=errors,
            details=details,
        )

    @staticmethod
    def bad_request(
        message: str = "Bad request",
        errors: list[Any] | None = None,
    ) -> ErrorResponse:
        """Create a 400 Bad Request response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            errors=errors,
        )

    @staticmethod
    def unauthorized(
        message: str = "Unauthorized",
    ) -> ErrorResponse:
        """Create a 401 Unauthorized response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    @staticmethod
    def forbidden(
        message: str = "Forbidden",
    ) -> ErrorResponse:
        """Create a 403 Forbidden response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
        )

    @staticmethod
    def not_found(
        message: str = "Resource not found",
        details: dict[str, Any] | None = None,
    ) -> ErrorResponse:
        """Create a 404 Not Found response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )

    @staticmethod
    def validation_error(
        errors: list[Any],
        message: str = "Validation error",
    ) -> ErrorResponse:
        """Create a 422 Validation Error response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            errors=errors,
        )

    @staticmethod
    def internal_server_error(
        message: str = "Internal server error",
        details: dict[str, Any] | None = None,
    ) -> ErrorResponse:
        """Create a 500 Internal Server Error response."""
        return ResponseBuilder.error(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )

    @staticmethod
    def paginated(
        data: list[T],
        total: int,
        page: int,
        page_size: int,
        message: str = "Data retrieved successfully",
    ) -> PaginatedResponse[T]:
        """
        Create a paginated response.

        Args:
            data: List of items
            total: Total number of items
            page: Current page number
            page_size: Items per page
            message: Success message

        Returns:
            PaginatedResponse: Formatted paginated response
        """
        return PaginatedResponse[T](
            status_code=status.HTTP_200_OK,
            success=True,
            message=message,
            data=data,
            metadata=MetaData(
                total=total,
                page=page,
                page_size=page_size,
                has_next=page * page_size < total,
                has_prev=page > 1,
            ),
        )
