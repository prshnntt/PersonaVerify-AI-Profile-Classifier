"""
predictor/utils.py

Custom utilities for the predictor app.
"""

from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status


def custom_exception_handler(exc, context):
    """
    Custom DRF exception handler that ensures all errors
    return a consistent JSON format.
    """
    response = exception_handler(exc, context)

    if response is not None:
        return Response(
            {
                'error': response.status_text if hasattr(response, 'status_text') else 'Error',
                'details': response.data,
            },
            status=response.status_code,
        )

    # Unhandled exceptions
    return Response(
        {'error': 'Internal server error', 'details': str(exc)},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
