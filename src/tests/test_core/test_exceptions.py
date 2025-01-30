import pytest
from src.core.exceptions import (
    GenterrError,
    ConfigError,
    LoggerError,
    ValidationError,
    ProcessingError,
    NetworkError,
    ResourceNotFoundError,
    AuthenticationError
)

def test_base_exception():
    """Test GenterrError base exception"""
    with pytest.raises(GenterrError) as exc_info:
        raise GenterrError("Base error message")
    assert str(exc_info.value) == "Base error message"
    assert isinstance(exc_info.value, Exception)

def test_config_error():
    """Test ConfigError"""
    with pytest.raises(ConfigError) as exc_info:
        raise ConfigError("Invalid configuration")
    assert str(exc_info.value) == "Invalid configuration"
    assert isinstance(exc_info.value, GenterrError)

def test_logger_error():
    """Test LoggerError"""
    with pytest.raises(LoggerError) as exc_info:
        raise LoggerError("Logging failed")
    assert str(exc_info.value) == "Logging failed"
    assert isinstance(exc_info.value, GenterrError)

def test_validation_error():
    """Test ValidationError"""
    with pytest.raises(ValidationError) as exc_info:
        raise ValidationError("Invalid input data")
    assert str(exc_info.value) == "Invalid input data"
    assert isinstance(exc_info.value, GenterrError)

def test_processing_error():
    """Test ProcessingError"""
    with pytest.raises(ProcessingError) as exc_info:
        raise ProcessingError("Processing failed")
    assert str(exc_info.value) == "Processing failed"
    assert isinstance(exc_info.value, GenterrError)

def test_network_error():
    """Test NetworkError"""
    with pytest.raises(NetworkError) as exc_info:
        raise NetworkError("Network connection failed")
    assert str(exc_info.value) == "Network connection failed"
    assert isinstance(exc_info.value, GenterrError)

def test_resource_not_found_error():
    """Test ResourceNotFoundError"""
    with pytest.raises(ResourceNotFoundError) as exc_info:
        raise ResourceNotFoundError("Resource not found")
    assert str(exc_info.value) == "Resource not found"
    assert isinstance(exc_info.value, GenterrError)

def test_authentication_error():
    """Test AuthenticationError"""
    with pytest.raises(AuthenticationError) as exc_info:
        raise AuthenticationError("Authentication failed")
    assert str(exc_info.value) == "Authentication failed"
    assert isinstance(exc_info.value, GenterrError)

def test_error_with_details():
    """Test exception with additional details"""
    details = {"code": 404, "path": "/some/path"}
    with pytest.raises(ResourceNotFoundError) as exc_info:
        raise ResourceNotFoundError("Resource not found", details=details)
    assert exc_info.value.details == details

def test_error_inheritance():
    """Test proper exception inheritance"""
    exceptions = [
        ConfigError, LoggerError, ValidationError,
        ProcessingError, NetworkError, ResourceNotFoundError,
        AuthenticationError
    ]
    
    for exception_class in exceptions:
        exc = exception_class("Test")
        assert isinstance(exc, GenterrError)
        assert isinstance(exc, Exception)