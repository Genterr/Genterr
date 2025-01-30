from datetime import datetime, UTC, timedelta
from typing import Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
import logging
import psutil
import os
import gc
import json
import hashlib
import numpy as np
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

# Zuerst alle Hilfsklassen
class ModelInputError(Exception):
    """Exception for input validation errors"""
    pass

class ModelStateError(Exception):
    """Exception for model state errors"""
    pass

class ModelStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    INFERENCING = "inferencing"
    ERROR = "error"
    OFFLINE = "offline"

class ModelError(Exception):
    """Base exception for model errors"""
    pass

class ModelConfig:
    def __init__(
        self,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        device: str = "cpu",
        enable_profiling: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_memory_usage: float = 0.9
    ):
        # Validate parameters
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_memory_usage <= 0 or max_memory_usage > 1:
            raise ValueError("max_memory_usage must be between 0 and 1")
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be either 'cpu' or 'cuda'")
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.enable_profiling = enable_profiling
        self.checkpoint_dir = checkpoint_dir
        self.max_memory_usage = max_memory_usage

class ModelMetrics:
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.inference_times = []
        self.memory_usage = []
        self.gpu_usage = []
        self.error_count = 0
        self.total_training_time = 0
        self.total_inference_samples = 0
        self.accuracy = 0.0
        self.recovery_attempts = 0
        self.last_updated = datetime.now(UTC)

    def update_training_metrics(self, loss: float, accuracy: float, time_taken: float):
        self.training_loss.append(loss)
        self.total_training_time += time_taken
        self.accuracy = accuracy
        self.last_updated = datetime.now(UTC)
        self.update_resource_usage()

    def update_inference_metrics(self, time_taken: float, accuracy: float = None):
        self.inference_times.append(time_taken)
        if accuracy is not None:
            self.accuracy = accuracy
        self.total_inference_samples += 1
        self.update_resource_usage()
        self.last_updated = datetime.now(UTC)

    def update_resource_usage(self) -> None:
        """Update memory and GPU usage metrics"""
        process = psutil.Process()
        self.memory_usage.append(process.memory_percent())
        
        if torch.cuda.is_available():
            self.gpu_usage.append(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_loss": self.training_loss,  
            "validation_loss": self.validation_loss,  
            "inference_times": self.inference_times,  
            "memory_usage": self.memory_usage,  
            "gpu_usage": self.gpu_usage,  
            "avg_training_loss": float(np.mean(self.training_loss)) if self.training_loss else 0.0,
            "avg_validation_loss": float(np.mean(self.validation_loss)) if self.validation_loss else 0.0,
            "avg_inference_time": float(np.mean(self.inference_times)) if self.inference_times else 0.0,
            "accuracy": float(self.accuracy),
            "total_training_time": float(self.total_training_time),
            "total_inference_samples": int(self.total_inference_samples),
            "error_count": int(self.error_count),
            "recovery_attempts": int(self.recovery_attempts),
            "last_updated": self.last_updated.isoformat()
        }
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load metrics from dictionary"""
        if "training_loss" in data:
            self.training_loss = data["training_loss"]
        if "validation_loss" in data:
            self.validation_loss = data["validation_loss"]
        if "inference_times" in data:
            self.inference_times = data["inference_times"]
        if "memory_usage" in data:
            self.memory_usage = data["memory_usage"]
        if "gpu_usage" in data:
            self.gpu_usage = data["gpu_usage"]
        if "error_count" in data:
            self.error_count = int(data["error_count"])
        if "total_training_time" in data:
            self.total_training_time = float(data["total_training_time"])
        if "total_inference_samples" in data:
            self.total_inference_samples = int(data["total_inference_samples"])
        if "accuracy" in data:
            self.accuracy = float(data["accuracy"])
        if "recovery_attempts" in data:
            self.recovery_attempts = int(data["recovery_attempts"])
        if "last_updated" in data:
            self.last_updated = datetime.fromisoformat(data["last_updated"])

class CustomModel(ABC):
    """Base class for all custom GENTERR AI models."""
    
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
        self.created_at: datetime = datetime.now(UTC)
        self.config: ModelConfig = config or ModelConfig()
        self.status: ModelStatus = ModelStatus.INITIALIZING
        self.metrics: ModelMetrics = ModelMetrics()
        
        # Setup logging
        self.logger = logging.getLogger(f"model.{self.name}")
        self.logger.setLevel(logging.INFO)
        
        # Resource management
        self._resources_initialized = False
        self._profile_data = {}

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


    @abstractmethod
    async def train(self, training_data: Any, validation_data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """Train the model on provided data."""
        raise NotImplementedError("Subclasses must implement train method")

    @abstractmethod
    async def predict(self, input_data: Any) -> Tuple[Any, float]:
        """Generate predictions for input data."""
        raise NotImplementedError("Subclasses must implement predict method")

    async def save(self, path: Optional[Path] = None) -> bool:
        """Save model state and configuration"""
        try:
            if path is None:
                path = Path(self.config.checkpoint_dir) / f"{self.name}_v{self.version}"
            
            path.mkdir(parents=True, exist_ok=True)
            
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
        
            # Load metrics
            metrics_path = path / "metrics.json"
            with open(metrics_path, 'r') as f:
                metrics_dict = json.load(f)
                self.metrics.from_dict(metrics_dict)  # Benutze die neue from_dict Methode
        
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Cleanup and shutdown model resources"""
        try:
            self.logger.info(f"Shutting down model {self.name}")
            
            
            if self.config.enable_profiling:
                checkpoint_dir = Path(self.config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                profile_path = checkpoint_dir / f"{self.name}_profile.json"
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