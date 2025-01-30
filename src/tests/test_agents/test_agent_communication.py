# src/tests/test_agents/test_agent_communication.py
import pytest
from uuid import uuid4
from datetime import datetime, UTC
from src.agents.core.collaboration.agent_communication import (
    AgentCommunication,
    MessageType,
    MessageStatus,
    MessageError
)

@pytest.fixture
def comm_system():
    """Create a fresh communication system for each test"""
    return AgentCommunication()

def test_validate_message_content(comm_system):
    """Test message content validation"""
    # Valid content
    valid_content = {
        "action": "test",
        "data": "test_data"
    }
    assert comm_system.validate_message_content(valid_content) == True
    
    # Invalid content (missing required field)
    invalid_content = {
        "action": "test"
    }
    with pytest.raises(MessageError) as exc_info:
        comm_system.validate_message_content(invalid_content)
    assert "Missing required fields" in str(exc_info.value)

@pytest.mark.asyncio
async def test_create_channel(comm_system):
    """Test channel creation"""
    channel_id = "test_channel"
    participants = [uuid4(), uuid4()]
    
    success = await comm_system.create_collaboration_channel(
        channel_id,
        participants
    )
    
    assert success == True
    assert channel_id in comm_system.active_channels
    assert len(comm_system.active_channels[channel_id]) == 2

@pytest.mark.asyncio
async def test_send_message(comm_system):
    """Test message sending"""
    from_agent = uuid4()
    to_agent = uuid4()
    content = {
        "action": "test_action",
        "data": "test_data"
    }
    
    success = await comm_system.send_message(
        from_agent,
        to_agent,
        MessageType.TASK,
        content
    )
    
    assert success == True
    messages = comm_system.get_pending_messages(to_agent)
    assert len(messages) == 1
    assert messages[0].content == content
    assert messages[0].status == MessageStatus.PENDING