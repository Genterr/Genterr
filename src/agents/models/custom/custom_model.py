from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import hashlib
import psutil
import torch
import gc
from contextlib import contextmanager

class ModelError(Exception):
    """Base exception for model-related errors"""
    pass

class ModelInitializationError(ModelError):
    """Raised when model initialization fails"""
    pass

class ModelInputError(ModelError):
    """Raised when input validation fails"""
    pass

class ModelStateError(ModelError):
    """Raised when model state is invalid"""
    pass

class ModelStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    INFERENCING = "inferencing"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

@dataclass
class ModelConfig:
    """Configuration settings for custom models"""
    batch_size: int = 32
    learning_rate: float = 0.001
    max_sequence_length: int = 512
    model_dimension: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"  # or "cuda" for GPU
    max_memory_usage: float = 0.9  # 90% of available memory
    enable_profiling: bool = False
    recovery_attempts: int = 3
    timeout_seconds: int = 300

class ModelMetrics:
    """Tracks model performance metrics"""
    def __init__(self):
        self.training_loss: List[float] = []
        self.validation_loss: List[float] = []
        self.inference_times: List[float] = []
        self.accuracy: float = 0.0
        self.total_training_time: float = 0.0
        self.total_inference_samples: int = 0
        self.last_updated: datetime = datetime.utcnow()
        self.memory_usage: List[float] = []
        self.gpu_usage: List[float] = []
        self.error_count: int = 0
        self.recovery_attempts: int = 0

    def update_training_metrics(self, train_loss: float, val_loss: float, epoch_time: float) -> None:
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)
        self.total_training_time += epoch_time
        self.update_resource_usage()
        self.last_updated = datetime.utcnow()

    def update_inference_metrics(self, inference_time: float, accuracy: float = None) -> None:
        self.inference_times.append(inference_time)
        self.total_inference_samples += 1
        if accuracy is not None:
            self.accuracy = (self.accuracy * (self.total_inference_samples - 1) + accuracy) / self.total_inference_samples
        self.update_resource_usage()
        self.last_updated = datetime.utcnow()

    def update_resource_usage(self) -> None:
        """Update memory and GPU usage metrics"""
        process = psutil.Process()
        self.memory_usage.append(process.memory_percent())
        
        if torch.cuda.is_available():
            self.gpu_usage.append(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_training_loss": np.mean(self.training_loss) if self.training_loss else 0.0,
            "avg_validation_loss": np.mean(self.validation_loss) if self.validation_loss else 0.0,
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "accuracy": self.accuracy,
            "total_training_time": self.total_training_time,
            "total_inference_samples": self.total_inference_samples,
            "avg_memory_usage": np.mean(self.memory_usage) if self.memory_usage else 0.0,
            "avg_gpu_usage": np.mean(self.gpu_usage) if self.gpu_usage else 0.0,
            "error_count": self.error_count,
            "recovery_attempts": self.recovery_attempts,
            "last_updated": self.last_updated.isoformat()
        }

class CustomModel(ABC):
    """
    Base class for all custom GENTERR AI models.
    Provides common functionality and interfaces that all custom models must implement.
    """

    def __init__(
        self,
        name: str,
        version: str,
        config: Optional[ModelConfig] = None,
        description: str = None
    ):
        self.model_id: UUID = uuid4()
        self.name: str = name
        self.version: str = version
        self.description: str = description or ""
        self.created_at: datetime = datetime.utcnow()
        self.config: ModelConfig = config or ModelConfig()
        self.status: ModelStatus = ModelStatus.INITIALIZING
        self.metrics: ModelMetrics = ModelMetrics()
        
        # Setup logging
        self.logger = logging.getLogger(f"model.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Resource management
        self._resources_initialized = False
        self._profile_data = {}

    @contextmanager
    def resource_manager(self):
        """Context manager for handling model resources"""
        try:
            if not self._resources_initialized:
                self._initialize_resources()
            yield
        finally:
            if self.config.enable_profiling:
                self._update_profile_data()
            self._check_resource_limits()

    def _initialize_resources(self):
        """Initialize model resources"""
        if torch.cuda.is_available() and self.config.device == "cuda":
            torch.cuda.empty_cache()
        self._resources_initialized = True

    def _check_resource_limits(self):
        """Check if resource usage is within limits"""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        if memory_percent > self.config.max_memory_usage * 100:
            self.logger.warning(f"Memory usage ({memory_percent}%) exceeds limit")
            gc.collect()

    def _update_profile_data(self):
        """Update profiling data"""
        if self.config.enable_profiling:
            self._profile_data[datetime.utcnow()] = {
                "memory": psutil.Process().memory_info().rss,
                "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }

    @abstractmethod
    async def train(self, training_data: Any, validation_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Train the model on provided data."""
        raise NotImplementedError("Subclasses must implement train method")

    @abstractmethod
    async def predict(self, input_data: Any) -> Tuple[Any, float]:
        """Generate predictions for input data."""
        raise NotImplementedError("Subclasses must implement predict method")

    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data before processing"""
        # Implementation depends on data type
        # Should be overridden by subclasses
        return input_data

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format and dimensions"""
        try:
            sanitized_data = self.sanitize_input(input_data)
            # Implement validation logic here
            return True
        except Exception as e:
            raise ModelInputError(f"Input validation failed: {str(e)}")

    async def save(self, path: Optional[Path] = None) -> bool:
        """Save model state and configuration"""
        try:
            if path is None:
                path = Path(self.config.checkpoint_dir) / f"{self.name}_v{self.version}"
            
            path.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            self.save_config(path)
            
            # Save metrics
            metrics_path = path / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            
            # Generate and save checksum
            self._save_checksum(path)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    async def load(self, path: Path) -> bool:
        """Load model state and configuration"""
        try:
            # Verify checksum
            if not self._verify_checksum(path):
                raise ModelStateError("Model checksum verification failed")
            
            # Load configuration and metrics
            self.load_config(path)
            
            metrics_path = path / "metrics.json"
            with open(metrics_path, 'r') as f:
                metrics_dict = json.load(f)
                # Update metrics (implementation depends on specific needs)
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

    def _save_checksum(self, path: Path) -> None:
        """Generate and save checksum for model files"""
        checksums = {}
        for file_path in path.glob('**/*'):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    checksums[str(file_path.relative_to(path))] = hashlib.sha256(f.read()).hexdigest()
        
        with open(path / "checksums.json", 'w') as f:
            json.dump(checksums, f, indent=2)

    def _verify_checksum(self, path: Path) -> bool:
        """Verify checksums of model files"""
        try:
            with open(path / "checksums.json", 'r') as f:
                stored_checksums = json.load(f)
            
            for file_path, stored_hash in stored_checksums.items():
                full_path = path / file_path
                if full_path.exists():
                    with open(full_path, 'rb') as f:
                        current_hash = hashlib.sha256(f.read()).hexdigest()
                    if current_hash != stored_hash:
                        return False
                else:
                    return False
            return True
        except Exception:
            return False

    async def shutdown(self) -> None:
        """Cleanup and shutdown model resources"""
        try:
            self.logger.info(f"Shutting down model {self.name}")
            
            # Save final metrics
            if self.config.enable_profiling:
                profile_path = Path(self.config.checkpoint_dir) / f"{self.name}_profile.json"
                with open(profile_path, 'w') as f:
                    json.dump(self._profile_data, f, indent=2)
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up other resources
            gc.collect()
            
            self.status = ModelStatus.OFFLINE
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.status = ModelStatus.ERROR

    def __repr__(self) -> str:
        return f"CustomModel(name='{self.name}', version='{self.version}', id={self.model_id}, status={self.status})"

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()