# src/utils/api/__init__.py
# Created: 2025-01-29 20:59:38
# Author: Genterr

"""
API utilities for making HTTP requests and handling responses.
"""

from .api_client import (
    APIClient,
    APIConfig,
    APIResponse,
    APIError,
    RequestError,
    ResponseError,
    AuthenticationError,
    RequestMethod
)

from .response_handler import (
    ResponseHandler,
    ProcessedResponse,
    ResponseHandlerError,
    ValidationError
)

__all__ = [
    'APIClient',
    'APIConfig',
    'APIResponse',
    'APIError',
    'RequestError',
    'ResponseError',
    'AuthenticationError',
    'RequestMethod',
    'ResponseHandler',
    'ProcessedResponse',
    'ResponseHandlerError',
    'ValidationError'
]