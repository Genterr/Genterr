# src/tests/test_models/test_custom_model.py
import pytest
from pathlib import Path
from datetime import datetime, UTC
from uuid import UUID
import torch
import gc
import psutil
from contextlib import contextmanager 
from src.agents.models.custom.custom_model import (
    CustomModel,
    ModelConfig,
    ModelStatus,
    ModelError,
    ModelMetrics,
    ModelInputError
)

class TestCustomModel(CustomModel):
    """Test implementation of CustomModel"""
    def __init__(self, name: str, version: str, config=None, description: str = None):
        super().__init__(name, version, config, description)
        self.status = ModelStatus.INITIALIZING
        self.metrics = ModelMetrics()

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
            self._profile_data[datetime.now(UTC)] = {
                "memory": psutil.Process().memory_info().rss,
                "gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }

    async def train(self, training_data, validation_data=None, **kwargs):
        self.status = ModelStatus.TRAINING
        self.metrics.update_training_metrics(0.1, 0.2, 1.0)
        self.status = ModelStatus.READY
        return {"loss": 0.1, "accuracy": 0.95}

    async def predict(self, input_data):
        """Generate predictions for input data"""
        if input_data is None:
            raise ModelInputError("Input data cannot be None")
            
        self.status = ModelStatus.INFERENCING
        result = input_data
        confidence = 0.95
        self.metrics.update_inference_metrics(0.1, accuracy=0.95)
        self.status = ModelStatus.READY
        return result, confidence

    async def save_config(self, path: Path) -> bool:
        """Save model configuration"""
        return True

    async def load_config(self, path: Path) -> bool:
        """Load model configuration"""
        return True

@pytest.fixture
def model_config():
    """Create a test model configuration"""
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    return ModelConfig(
        batch_size=16,
        learning_rate=0.001,
        checkpoint_dir="checkpoints",
        enable_profiling=True
    )

@pytest.fixture
def test_model(model_config):
    """Create a test model instance"""
    return TestCustomModel(
        name="test_model",
        version="1.0",
        config=model_config,
        description="Test model for unit tests"
    )

def test_model_config_validation():
    """Test model configuration validation"""
    # Test invalid batch size
    with pytest.raises(ValueError):
        ModelConfig(batch_size=-1)
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        ModelConfig(learning_rate=0)
    
    # Test invalid memory usage
    with pytest.raises(ValueError):
        ModelConfig(max_memory_usage=1.5)
    
    # Test invalid device
    with pytest.raises(ValueError):
        ModelConfig(device="invalid_device")
    
    # Test valid configuration
    config = ModelConfig(
        batch_size=32,
        learning_rate=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_profiling=True
    )
    assert config.batch_size == 32
    assert config.learning_rate == 0.01

@pytest.mark.asyncio
async def test_model_training(test_model):
    """Test model training process"""
    training_data = ["test_data"]
    result = await test_model.train(training_data)
    
    assert "loss" in result
    assert "accuracy" in result
    assert len(test_model.metrics.training_loss) > 0
    assert test_model.status == ModelStatus.READY

@pytest.mark.asyncio
async def test_model_prediction(test_model):
    """Test model prediction"""
    input_data = {"test": "data"}
    prediction, confidence = await test_model.predict(input_data)
    
    assert prediction == input_data
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
    assert len(test_model.metrics.inference_times) > 0

@pytest.mark.asyncio
async def test_model_save_load(test_model, tmp_path):
    """Test model save and load functionality"""
    # Save model
    save_path = tmp_path / "test_model"
    success = await test_model.save(save_path)
    assert success == True
    assert save_path.exists()
    
    # Load model
    new_model = TestCustomModel("test_model", "1.0")
    success = await new_model.load(save_path)
    assert success == True

def test_model_metrics(test_model):
    """Test metrics tracking"""
    metrics = test_model.metrics
    
    # Initial state
    assert metrics.training_loss == []
    assert metrics.inference_times == []
    assert metrics.error_count == 0
    
    # Update metrics
    metrics.update_training_metrics(0.5, 0.4, 1.0)
    assert len(metrics.training_loss) == 1
    assert metrics.total_training_time > 0

@pytest.mark.asyncio
async def test_model_shutdown(test_model):
    """Test model shutdown process"""
    await test_model.shutdown()
    assert test_model.status == ModelStatus.OFFLINE

@pytest.mark.asyncio
async def test_resource_management(test_model):
    """Test resource management functionality"""
    with test_model.resource_manager():
        # Simulate some work
        await test_model.predict({"test": "data"})
        
    assert test_model._resources_initialized
    if test_model.config.enable_profiling:
        assert len(test_model._profile_data) > 0

@pytest.mark.asyncio
async def test_model_error_handling(test_model):
    """Test error handling in model operations"""
    # Test invalid input handling
    with pytest.raises(ModelInputError):
        await test_model.predict(None)
    
    # Test state transitions during errors
    test_model.status = ModelStatus.READY
    try:
        raise Exception("Simulated error")
    except:
        test_model.status = ModelStatus.ERROR
        test_model.metrics.error_count += 1
    assert test_model.status == ModelStatus.ERROR
    assert test_model.metrics.error_count > 0

@pytest.mark.asyncio
async def test_model_recovery(test_model):
    """Test model recovery from error state"""
    test_model.status = ModelStatus.ERROR
    test_model.metrics.recovery_attempts = 0
    
    # Simulate recovery
    test_model.status = ModelStatus.INITIALIZING
    test_model.metrics.recovery_attempts += 1
    test_model.status = ModelStatus.READY
    
    assert test_model.status == ModelStatus.READY
    assert test_model.metrics.recovery_attempts == 1

@pytest.mark.asyncio
async def test_memory_limit_handling(test_model):
    """Test handling of memory limits"""
    original_memory_usage = test_model.config.max_memory_usage
    test_model.config.max_memory_usage = 0.1  # Set very low memory limit
    
    with test_model.resource_manager():
        # Should trigger memory warning and cleanup
        large_data = ["x" * 1000000 for _ in range(1000)]
        await test_model.predict({"large_data": large_data})
    
    # Cleanup and restore original limit
    del large_data
    gc.collect()
    test_model.config.max_memory_usage = original_memory_usage
    assert test_model.status != ModelStatus.ERROR

@pytest.mark.asyncio
async def test_gpu_resource_management(test_model):
    """Test GPU resource management if available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    with test_model.resource_manager():
        # Simulate GPU operations
        test_model.config.device = "cuda"
        tensor = torch.cuda.FloatTensor(1000, 1000)
        await test_model.predict({"tensor": tensor.cpu().numpy()})
        
        # Check GPU memory management
        assert torch.cuda.memory_allocated() > 0
        
    # After context manager, memory should be cleared
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() == 0

def test_metrics_serialization(test_model):
    """Test metrics serialization and deserialization"""
    # Add some test metrics
    test_model.metrics.update_training_metrics(0.5, 0.8, 1.0)
    test_model.metrics.update_inference_metrics(0.1, 0.9)
    
    # Convert to dict
    metrics_dict = test_model.metrics.to_dict()
    
    # Verify dictionary contents
    assert isinstance(metrics_dict["avg_training_loss"], float)
    assert isinstance(metrics_dict["accuracy"], float)
    assert isinstance(metrics_dict["total_training_time"], float)
    assert isinstance(metrics_dict["last_updated"], str)
    
    # Create new metrics from dict
    new_metrics = ModelMetrics()
    for key, value in metrics_dict.items():
        if hasattr(new_metrics, key):
            if key == "last_updated":
                new_metrics.last_updated = datetime.fromisoformat(value)
            else:
                setattr(new_metrics, key, value)
    
    # Verify values match
    assert new_metrics.accuracy == test_model.metrics.accuracy
    assert new_metrics.total_training_time == test_model.metrics.total_training_time

@pytest.mark.asyncio
async def test_metrics_persistence(test_model, tmp_path):
    """Test metrics persistence through save/load cycle"""
    # Set some metrics
    test_model.metrics.update_training_metrics(0.5, 0.8, 1.0)
    test_model.metrics.error_count = 2
    test_model.metrics.recovery_attempts = 1
    
    # Save model
    save_path = tmp_path / "metrics_test"
    await test_model.save(save_path)
    
    # Load into new model
    new_model = TestCustomModel("test_model", "1.0")
    await new_model.load(save_path)
    
    # Verify metrics maintained
    assert new_model.metrics.training_loss == test_model.metrics.training_loss
    assert new_model.metrics.error_count == test_model.metrics.error_count
    assert new_model.metrics.recovery_attempts == test_model.metrics.recovery_attempts

def test_metrics_resource_tracking(test_model):
    """Test resource usage tracking in metrics"""
    # Initial state
    assert len(test_model.metrics.memory_usage) == 0
    assert len(test_model.metrics.gpu_usage) == 0
    
    # Update metrics
    test_model.metrics.update_resource_usage()
    
    # Verify memory tracking
    assert len(test_model.metrics.memory_usage) == 1
    assert isinstance(test_model.metrics.memory_usage[0], float)
    
    # Verify GPU tracking
    if torch.cuda.is_available():
        assert len(test_model.metrics.gpu_usage) == 1
        assert isinstance(test_model.metrics.gpu_usage[0], float)