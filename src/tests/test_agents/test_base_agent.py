import pytest
from src.agents.core.base.base_agent import BaseAgent, AgentStatus, AgentConfig

@pytest.fixture
def test_agent():
    """Create a test agent for each test"""
    return BaseAgent("test_agent", "Test description")

def test_agent_initialization(test_agent):
    """Test if agent initializes with correct values"""
    assert test_agent.name == "test_agent"
    assert test_agent.description == "Test description"
    assert test_agent.status == AgentStatus.INITIALIZED

def test_agent_status_change(test_agent):
    """Test if agent status can be changed"""
    test_agent.status = AgentStatus.ACTIVE
    assert test_agent.status == AgentStatus.ACTIVE