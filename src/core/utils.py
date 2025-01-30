from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any, Optional
import re
import os
from .exceptions import ValidationError

def format_datetime(dt: datetime, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime object to string."""
    return dt.strftime(format)

def validate_string(value: Optional[str]) -> str:
    """Validate and clean a string value."""
    if not value or not isinstance(value, str) or not value.strip():
        raise ValidationError("String value is required")
    return value.strip()

def validate_datetime(value: Any) -> datetime:
    """Validate a datetime object."""
    if not isinstance(value, datetime):
        raise ValidationError("Valid datetime object is required")
    return value

def validate_path(path: Union[str, Path]) -> Path:
    """Validate and convert a path to Path object."""
    if not path:
        raise ValidationError("Path is required")
    return Path(path) if isinstance(path, str) else path

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Replace multiple underscores with a single underscore
    filename = re.sub(r'_+', '_', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename.strip())
    return filename

def is_valid_email(email: Optional[str]) -> bool:
    """Check if a string is a valid email address."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def parse_datetime(value: Optional[str]) -> datetime:
    """Parse a string to datetime object."""
    if not value:
        raise ValidationError("Datetime string is required")
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    
    raise ValidationError(f"Invalid datetime format: {value}")

def create_directory(path: Union[str, Path]) -> Path:
    """Create a directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_extension(filename: Optional[str]) -> str:
    """Get the file extension from a filename."""
    if not filename:
        return ""
    return os.path.splitext(filename)[1]

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
            
    return result