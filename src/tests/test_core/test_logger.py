import pytest
import logging
import os
from pathlib import Path
from src.core.config import Config
from src.core.logger import Logger, LoggerError

@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file"""
    return tmp_path / "test.log"

@pytest.fixture
def config_with_custom_logging(temp_log_file):
    """Create a config with custom logging settings"""
    config = Config()
    config.update({
        "logging": {
            "level": "DEBUG",
            "file": str(temp_log_file),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    })
    return config

@pytest.fixture
def logger(config_with_custom_logging):
    """Create a logger instance with custom config"""
    return Logger(config_with_custom_logging)

def test_logger_initialization(logger):
    """Test basic logger initialization"""
    assert logger.logger.level == logging.DEBUG
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.name == "genterr"

def test_logger_file_handler(logger, temp_log_file):
    """Test if file handler is properly configured"""
    assert os.path.exists(temp_log_file)
  
    logger.debug("Test message")
    
    with open(temp_log_file, 'r') as f:
        log_content = f.read()
    
    assert "Test message" in log_content

def test_logger_levels(logger):
    """Test different logging levels"""
    test_messages = {
        "debug": "Debug message",
        "info": "Info message",
        "warning": "Warning message",
        "error": "Error message",
        "critical": "Critical message"
    }
    
    for level, message in test_messages.items():
        getattr(logger, level)(message)

def test_invalid_log_level():
    """Test logger initialization with invalid log level"""
    config = Config()
    config.update({"logging": {"level": "INVALID_LEVEL"}})
    
    with pytest.raises(LoggerError):
        Logger(config)

def test_invalid_log_file():
    """Test logger initialization with invalid log file path"""
    config = Config()
    config.update({"logging": {"file": "/invalid/path/log.txt"}})
    
    with pytest.raises(LoggerError):
        Logger(config)

def test_logger_format(config_with_custom_logging, temp_log_file):
    """Test custom log format"""
    logger = Logger(config_with_custom_logging)
    test_message = "Test format message"
    logger.info(test_message)
    
    with open(temp_log_file, 'r') as f:
        log_content = f.read()
    
    assert test_message in log_content
    assert " - INFO - " in log_content

def test_logger_context(logger, temp_log_file):
    """Test logger context information"""
    context = {"user": "test_user", "action": "test_action"}
    logger.info("Test with context", extra=context)
    
    with open(temp_log_file, 'r') as f:
        log_content = f.read()
    
    assert "test_user" in log_content
    assert "test_action" in log_content

def test_multiple_handlers(config_with_custom_logging, temp_log_file):
    """Test logger with multiple handlers"""
    # Add console handler
    config_with_custom_logging.update({
        "logging": {
            "console_output": True
        }
    })
    
    logger = Logger(config_with_custom_logging)
    assert len(logger.logger.handlers) == 2  # File and console handlers

def test_log_rotation(tmp_path):
    """Test log file rotation"""
    config = Config()
    config.update({
        "logging": {
            "file": str(tmp_path / "rotating.log"),
            "max_size": 1024,  # 1KB
            "backup_count": 3
        }
    })
    
    logger = Logger(config)
    
    # Write enough logs to trigger rotation
    large_message = "x" * 512  # 512 bytes
    for _ in range(10):
        logger.info(large_message)
    
    # Check if backup files were created
    log_files = list(tmp_path.glob("rotating.log*"))
    assert len(log_files) > 1
