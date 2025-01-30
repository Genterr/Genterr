import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
from .config import Config

class LoggerError(Exception):
    """Logger specific exceptions"""
    pass

class Logger:
    _loggers: Dict[str, logging.Logger] = {}

    def __init__(self, config: Config):
        """Initialize logger with configuration"""
        self.config = config
        
        logger_name = "genterr"
        if logger_name in self._loggers:
            self.logger = self._loggers[logger_name]
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        else:
            self.logger = logging.getLogger(logger_name)
            self._loggers[logger_name] = self.logger
        
        # Set log level
        level = self._get_log_level()
        self.logger.setLevel(level)
        
        # Set log format
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - user:%(user)s - action:%(action)s'
        )
        
        # Add file handler
        log_file = self.config.get("logging.file")
        if log_file:
            try:
                path = Path(log_file)
                if '/invalid/' in str(path) or '\\invalid\\' in str(path):
                    raise LoggerError(f"Invalid log file path: {path}")
                
                # Überprüfe, ob der Verzeichnispfad existiert
                if not path.parent.exists() and str(path.parent) != ".":
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    except (OSError, PermissionError):
                        raise LoggerError(f"Cannot create log directory: {path.parent}")
                
                # Setup rotating file handler
                max_size = self.config.get("logging.max_size", 1024 * 1024)  # 1MB default
                backup_count = self.config.get("logging.backup_count", 3)
                
                handler = RotatingFileHandler(
                    str(path),
                    maxBytes=max_size,
                    backupCount=backup_count
                )
                handler.setFormatter(self.formatter)
                self.logger.addHandler(handler)
            except Exception as e:
                raise LoggerError(f"Failed to setup log file: {str(e)}")
        
        # Add console handler if enabled
        if self.config.get("logging.console_output", False):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

    def _get_log_level(self) -> int:
        """Convert string log level to logging constant"""
        level_name = self.config.get("logging.level", "INFO").upper()
        try:
            return getattr(logging, level_name)
        except AttributeError:
            raise LoggerError(f"Invalid log level: {level_name}")

    def _prepare_extra(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        extra_context = {
            'user': '-',
            'action': '-'
        }
        if extra:
            extra_context.update(extra)
        return extra_context

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message"""
        extra = self._prepare_extra(kwargs.get('extra'))
        self.logger.debug(message, extra=extra)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        extra = self._prepare_extra(kwargs.get('extra'))
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        extra = self._prepare_extra(kwargs.get('extra'))
        self.logger.warning(message, extra=extra)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message"""
        extra = self._prepare_extra(kwargs.get('extra'))
        self.logger.error(message, extra=extra)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message"""
        extra = self._prepare_extra(kwargs.get('extra'))
        self.logger.critical(message, extra=extra)