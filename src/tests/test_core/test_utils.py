import pytest
from datetime import datetime
from src.core.utils import (
    format_datetime,
    validate_string,
    validate_datetime,
    validate_path,
    sanitize_filename,
    is_valid_email,
    parse_datetime,
    create_directory,
    get_file_extension,
    merge_dicts
)
from src.core.exceptions import ValidationError
from pathlib import Path
import os

def test_format_datetime():
    """Test datetime formatting"""
    dt = datetime(2025, 1, 30, 21, 11, 14)
    assert format_datetime(dt) == "2025-01-30 21:11:14"
    assert format_datetime(dt, format="%Y-%m-%d") == "2025-01-30"
    assert format_datetime(dt, format="%H:%M:%S") == "21:11:14"

def test_validate_string():
    """Test string validation"""
    assert validate_string("test") == "test"
    assert validate_string("  test  ") == "test"
    
    with pytest.raises(ValidationError):
        validate_string("")
    with pytest.raises(ValidationError):
        validate_string(None)
    with pytest.raises(ValidationError):
        validate_string("   ")

def test_validate_datetime():
    """Test datetime validation"""
    dt = datetime.now()
    assert validate_datetime(dt) == dt
    
    with pytest.raises(ValidationError):
        validate_datetime("not a datetime")
    with pytest.raises(ValidationError):
        validate_datetime(None)

def test_validate_path():
    """Test path validation"""
    valid_path = Path("test_dir/test_file.txt")
    assert validate_path(valid_path) == valid_path
    assert validate_path("test_dir/test_file.txt") == Path("test_dir/test_file.txt")
    
    with pytest.raises(ValidationError):
        validate_path("")
    with pytest.raises(ValidationError):
        validate_path(None)

def test_sanitize_filename():
    """Test filename sanitization"""
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test\\file.txt") == "test_file.txt"
    assert sanitize_filename("test:*?\"<>|file.txt") == "test_file.txt"
    assert sanitize_filename(" test file.txt ") == "test_file.txt"

def test_is_valid_email():
    """Test email validation"""
    assert is_valid_email("test@example.com") is True
    assert is_valid_email("test.name@example.co.uk") is True
    assert is_valid_email("test+label@example.com") is True
    
    assert is_valid_email("invalid") is False
    assert is_valid_email("invalid@") is False
    assert is_valid_email("@invalid.com") is False
    assert is_valid_email("") is False
    assert is_valid_email(None) is False

def test_parse_datetime():
    """Test datetime parsing"""
    assert parse_datetime("2025-01-30 21:11:14") == datetime(2025, 1, 30, 21, 11, 14)
    assert parse_datetime("2025-01-30") == datetime(2025, 1, 30)
    
    with pytest.raises(ValidationError):
        parse_datetime("invalid")
    with pytest.raises(ValidationError):
        parse_datetime("")
    with pytest.raises(ValidationError):
        parse_datetime(None)

def test_create_directory(tmp_path):
    """Test directory creation"""
    test_dir = tmp_path / "test_dir"
    created_dir = create_directory(test_dir)
    assert created_dir.exists()
    assert created_dir.is_dir()
    
    # Test nested directory creation
    nested_dir = tmp_path / "parent" / "child"
    created_nested = create_directory(nested_dir)
    assert created_nested.exists()
    assert created_nested.is_dir()

def test_get_file_extension():
    """Test file extension extraction"""
    assert get_file_extension("test.txt") == ".txt"
    assert get_file_extension("test.tar.gz") == ".gz"
    assert get_file_extension("test") == ""
    assert get_file_extension(".hidden") == ""
    assert get_file_extension("") == ""
    assert get_file_extension(None) == ""

def test_merge_dicts():
    """Test dictionary merging"""
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    
    merged = merge_dicts(dict1, dict2)
    assert merged == {
        "a": 1,
        "b": {"c": 2, "d": 3},
        "e": 4
    }
    
    # Test that original dicts are not modified
    assert dict1 == {"a": 1, "b": {"c": 2}}
    assert dict2 == {"b": {"d": 3}, "e": 4}