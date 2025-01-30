import pytest
import json
import os
from pathlib import Path
from datetime import datetime, UTC
from src.core.config import Config, ConfigError

@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before and after each test"""
    # Backup all GENTERR_ environment variables
    saved_vars = {k: v for k, v in os.environ.items() if k.startswith("GENTERR_")}
    
    # Clear all GENTERR_ environment variables
    for key in list(saved_vars.keys()):
        del os.environ[key]
    
    yield
    
    # Clean up any test variables
    for key in list(os.environ.keys()):
        if key.startswith("GENTERR_"):
            del os.environ[key]
    
    # Restore original variables
    for key, value in saved_vars.items():
        os.environ[key] = value

@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration"""
    return {
        "app": {
            "name": "genterr",
            "version": "1.0.0",
            "debug": False,
            "created_at": "2025-01-30T17:36:03+00:00"
        },
        "model": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_memory_usage": 0.9,
            "device": "cpu"
        },
        "logging": {
            "level": "INFO",
            "file": "genterr.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "security": {
            "token_expiration": 3600,
            "max_retries": 3
        }
    }

@pytest.fixture
def config_file(tmp_path, sample_config):
    """Fixture creating a temporary config file"""
    config_path = tmp_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(sample_config, f)
    return config_path

def test_config_loading(config_file, sample_config):
    """Test basic configuration loading from file"""
    config = Config(config_file)
    assert config.get("app.name") == "genterr"
    assert config.get("model.batch_size") == 32
    assert config.get("logging.level") == "INFO"
    assert config.get("security.token_expiration") == 3600

def test_environment_variables(clean_env):
    """Test environment variable overrides"""
    # Set environment variables before creating config
    os.environ["GENTERR_MODEL_BATCH_SIZE"] = "64"
    os.environ["GENTERR_LOGGING_LEVEL"] = "DEBUG"
    
    # Create new config instance
    config = Config()
    
    # Check if environment variables are properly loaded
    assert config.get("model.batch_size") == 64, "Environment variable GENTERR_MODEL_BATCH_SIZE not properly loaded"
    assert config.get("logging.level") == "DEBUG", "Environment variable GENTERR_LOGGING_LEVEL not properly loaded"

def test_config_validation():
    """Test configuration validation rules"""
    with pytest.raises(ConfigError):
        Config().validate({
            "model": {
                "batch_size": -1,
                "learning_rate": 0
            }
        })

    with pytest.raises(ConfigError):
        Config().validate({
            "model": {
                "max_memory_usage": 1.5
            }
        })

def test_config_defaults(clean_env):
    """Test default configuration values"""
    config = Config()
    assert config.get("app.debug") == False
    assert config.get("model.batch_size") == 32
    assert config.get("logging.level") == "INFO"
    assert config.get("nonexistent.key", default="default") == "default"

def test_config_update():
    """Test configuration updates"""
    config = Config()
    config.update({
        "model": {
            "batch_size": 128,
            "learning_rate": 0.01
        }
    })
    assert config.get("model.batch_size") == 128
    assert config.get("model.learning_rate") == 0.01

def test_nested_config_access():
    """Test accessing nested configuration values"""
    config = Config()
    config.set("deep.nested.value", 42)
    assert config.get("deep.nested.value") == 42
    
    config.update({"another": {"nested": {"key": "value"}}})
    assert config.get("another.nested.key") == "value"

def test_config_type_conversion(clean_env):
    """Test configuration value type conversion"""
    os.environ["GENTERR_APP_DEBUG"] = "true"
    os.environ["GENTERR_MODEL_BATCH_SIZE"] = "128"
    os.environ["GENTERR_MODEL_LEARNING_RATE"] = "0.001"
    
    config = Config()
    assert isinstance(config.get("app.debug"), bool)
    assert isinstance(config.get("model.batch_size"), int)
    assert isinstance(config.get("model.learning_rate"), float)

def test_invalid_config_file():
    """Test handling of invalid configuration file"""
    with pytest.raises(ConfigError):
        Config(Path("nonexistent_config.json"))

def test_config_serialization(sample_config, tmp_path):
    """Test configuration serialization and deserialization"""
    config = Config()
    config.update(sample_config)
    
    save_path = tmp_path / "saved_config.json"
    config.save(save_path)
    
    loaded_config = Config(save_path)
    assert loaded_config.get("app.name") == config.get("app.name")
    assert loaded_config.get("model.batch_size") == config.get("model.batch_size")