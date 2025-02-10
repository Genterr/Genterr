import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

class ConfigError(Exception):
    """Configuration related errors"""
    pass

class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self._config: Dict[str, Any] = {}
        self._load_defaults()
        self._load_environment_variables()
        if config_path:
            self.load(config_path)
        self.validate(self._config)

    def _load_defaults(self) -> None:
        """Load default configuration values"""
        self._config = {
            "app": {
                "name": "genterr",
                "version": "1.0.0",
                "debug": False,
                "created_at": datetime.now(timezone.utc).isoformat()
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

    def load(self, path: Path) -> None:
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                file_config = json.load(f)
                self.update(file_config)
        except Exception as e:
            raise ConfigError(f"Failed to load config from {path}: {str(e)}")

    def save(self, path: Path) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if key.startswith("GENTERR_"):
                # Convert GENTERR_MODEL_BATCH_SIZE to model.batch_size
                parts = key[8:].lower().split('_')
                
                # Handle special cases for combined words
                if len(parts) > 2:
                    # Combine parts after first one with underscore
                    config_key = f"{parts[0]}.{'_'.join(parts[1:])}"
                else:
                    config_key = '.'.join(parts)
                
                # Convert and set the value
                converted_value = self._convert_value(value)
                self.set(config_key, converted_value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary"""
        def update_recursive(d1, d2):
            for k, v in d2.items():
                if isinstance(v, dict):
                    if k not in d1:
                        d1[k] = {}
                    update_recursive(d1[k], v)
                else:
                    d1[k] = v
            return d1
        
        update_recursive(self._config, config_dict)

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate configuration values"""
        if "model" in config:
            model_config = config["model"]
            if "batch_size" in model_config and model_config["batch_size"] <= 0:
                raise ConfigError("batch_size must be positive")
            if "learning_rate" in model_config and model_config["learning_rate"] <= 0:
                raise ConfigError("learning_rate must be positive")
            if "max_memory_usage" in model_config:
                if not 0 < model_config["max_memory_usage"] <= 1:
                    raise ConfigError("max_memory_usage must be between 0 and 1")

    @staticmethod
    def _convert_value(value: str) -> Any:
        """Convert string value to appropriate type"""
        # Handle boolean values
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            # If not a number, return as string
            return value
