"""Test module for custom model implementation."""
import pytest
import asyncio
from pathlib import Path
from datetime import datetime, UTC
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

class ConcreteTestModel(CustomModel):
    """Concrete implementation of CustomModel for testing."""
    
    async def train(self, training_data, validation_data=None, **kwargs):
        """Implement required train method."""
        self.status = ModelStatus.TRAINING
        self.metrics.update_training_metrics(0.1, 0.2, 1.0)
        self.status = ModelStatus.READY
        return {"loss": 0.1, "accuracy": 0.95}

    async def predict(self, input_data):
        """Implement required predict method."""
        if input_data is None:
            raise ModelInputError("Input data cannot be None")
        self.status = ModelStatus.INFERENCING
        result = input_data
        confidence = 0.95
        self.metrics.update_inference_metrics(0.1, accuracy=0.95)
        self.status = ModelStatus.READY
        return result, confidence

@pytest.fixture(scope="session")
def event_loop():
    """Create and provide a session-scoped event loop."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def model_config():
    """Create a test model configuration."""
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    return ModelConfig(
        batch_size=16,
        learning_rate=0.001,
        checkpoint_dir="checkpoints",
        enable_profiling=True
    )

@pytest.fixture
def test_model(model_config):
    """Create and provide a test model instance."""
    model = ConcreteTestModel(
        name="test_model",
        version="1.0",
        config=model_config,
        description="Test model for unit tests"
    )
    return model

@pytest.mark.asyncio
class TestCustomModel:
    """Test cases for CustomModel class."""
    
    @pytest.fixture(autouse=True)
    def setup(self, test_model):
        """Set up test fixtures."""
        self.model = test_model
    
    async def test_custom_model_initialization(self):
        """Test that the model initializes correctly."""
        assert isinstance(self.model, CustomModel)
        assert self.model.name == "test_model"
        assert self.model.version == "1.0"

    async def test_training_workflow(self):
        """Test the complete training workflow."""
        training_data = ["test_data"]
        result = await self.model.train(training_data)
        assert "loss" in result
        assert "accuracy" in result
        assert self.model.status == ModelStatus.READY

    async def test_prediction_workflow(self):
        """Test the complete prediction workflow."""
        input_data = {"test": "data"}
        prediction, confidence = await self.model.predict(input_data)
        assert prediction == input_data
        assert 0 <= confidence <= 1
        assert self.model.status == ModelStatus.READY

    async def test_model_error_handling(self):
        """Test error handling in the model."""
        with pytest.raises(ModelInputError):
            await self.model.predict(None)

    async def test_cleanup(self):
        """Test proper cleanup of resources."""
        await self.model.shutdown()
        assert self.model.status == ModelStatus.OFFLINE