"""Test module for custom model implementation."""
import pytest
import asyncio
import sys
from pathlib import Path
import torch
import gc
import psutil
from src.agents.models.custom.custom_model import (
    CustomModel,
    ModelConfig,
    ModelStatus,
    ModelError,
    ModelMetrics,
    ModelInputError
)

# Event Loop Setup für Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class ConcreteTestModel(CustomModel):
    """Concrete implementation of CustomModel for testing."""
    
    def __init__(self, name: str, version: str, config: ModelConfig = None, description: str = None):
        """Initialize the test model."""
        super().__init__(
            name=name,
            version=version,
            config=config or ModelConfig(),
            description=description
        )
        self.status = ModelStatus.INITIALIZING
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model resources."""
        self._resources_initialized = True
        self.status = ModelStatus.READY

    async def train(self, training_data, validation_data=None, **kwargs):
        """Implement required train method."""
        if training_data is None:
            raise ModelInputError("Training data cannot be None")
        
        if not self._resources_initialized:
            raise ModelStateError("Model not initialized")

        self.status = ModelStatus.TRAINING
        try:
            self.metrics.update_training_metrics(0.1, 0.2, 1.0)
            return {"loss": 0.1, "accuracy": 0.95}
        finally:
            self.status = ModelStatus.READY

    async def predict(self, input_data):
        """Implement required predict method."""
        if input_data is None:
            raise ModelInputError("Input data cannot be None")
        
        if not self._resources_initialized:
            raise ModelStateError("Model not initialized")

        self.status = ModelStatus.INFERENCING
        try:
            result = input_data
            confidence = 0.95
            self.metrics.update_inference_metrics(0.1, accuracy=0.95)
            return result, confidence
        finally:
            self.status = ModelStatus.READY

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the session."""
    if sys.platform.startswith('win'):
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    
    asyncio.set_event_loop(loop)
    yield loop
    
    # Cleanup
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()
        asyncio.set_event_loop(None)

@pytest.fixture(scope="function")
def checkpoint_dir():
    """Create and manage test checkpoint directory."""
    path = Path("test_checkpoints")
    path.mkdir(parents=True, exist_ok=True)
    yield path
    # Cleanup
    if path.exists():
        for file in path.glob("*"):
            file.unlink()
        path.rmdir()

@pytest.fixture(scope="function")
def model_config(checkpoint_dir):
    """Create a test model configuration."""
    return ModelConfig(
        batch_size=16,
        learning_rate=0.001,
        checkpoint_dir=str(checkpoint_dir),
        enable_profiling=True
    )

@pytest.fixture
async def test_model(model_config):
    """Create and manage a test model instance."""
    model = ConcreteTestModel(
        name="test_model",
        version="1.0",
        config=model_config,
        description="Test model for unit tests"
    )
    
    try:
        yield model
    finally:
        await model.shutdown()

@pytest.mark.asyncio
class TestCustomModel:
    """Test cases for CustomModel class."""

    async def test_custom_model_initialization(self, test_model):
        """Test model initialization."""
        model = await anext(test_model)
        assert isinstance(model, CustomModel)
        assert model.name == "test_model"
        assert model.version == "1.0"
        assert isinstance(model.config, ModelConfig)
        assert model.status == ModelStatus.READY
        assert model._resources_initialized

    async def test_training_workflow(self, test_model):
        """Test complete training workflow."""
        model = await anext(test_model)
        training_data = ["test_data"]
        result = await model.train(training_data)
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"] == 0.1
        assert result["accuracy"] == 0.95
        assert model.status == ModelStatus.READY
        assert len(model.metrics.training_loss) > 0

    async def test_prediction_workflow(self, test_model):
        """Test complete prediction workflow."""
        model = await anext(test_model)
        input_data = {"test": "data"}
        prediction, confidence = await model.predict(input_data)
        
        assert prediction == input_data
        assert confidence == 0.95
        assert model.status == ModelStatus.READY
        assert len(model.metrics.inference_times) > 0

    async def test_model_error_handling(self, test_model):
        """Test error handling in the model."""
        model = await anext(test_model)
        with pytest.raises(ModelInputError) as exc_info:
            await model.predict(None)
        assert "Input data cannot be None" in str(exc_info.value)

        with pytest.raises(ModelInputError) as exc_info:
            await model.train(None)
        assert "Training data cannot be None" in str(exc_info.value)

    async def test_cleanup(self, test_model):
        """Test proper cleanup of resources."""
        model = await anext(test_model)
        assert model.status == ModelStatus.READY
        await model.shutdown()
        assert model.status == ModelStatus.OFFLINE

    async def test_resource_management(self, test_model):
        """Test resource management."""
        model = await anext(test_model)
        initial_memory = psutil.Process().memory_info().rss
        
        # Create some load
        large_data = ["x" * 1000000 for _ in range(100)]
        await model.predict({"data": large_data})
        
        current_memory = psutil.Process().memory_info().rss
        del large_data
        await model.shutdown()
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        assert final_memory < current_memory