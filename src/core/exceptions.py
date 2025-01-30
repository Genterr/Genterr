from typing import Any, Dict, Optional

class GenterrError(Exception):
    """Base exception class for all Genterr exceptions"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ConfigError(GenterrError):
    """Raised when there is a configuration error"""
    pass

class LoggerError(GenterrError):
    """Raised when there is a logging error"""
    pass

class ValidationError(GenterrError):
    """Raised when data validation fails"""
    pass

class ProcessingError(GenterrError):
    """Raised when processing of data fails"""
    pass

class NetworkError(GenterrError):
    """Raised when network operations fail"""
    pass

class ResourceNotFoundError(GenterrError):
    """Raised when a requested resource is not found"""
    pass

class AuthenticationError(GenterrError):
    """Raised when authentication fails"""
    pass