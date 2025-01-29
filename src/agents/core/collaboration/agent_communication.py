from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from uuid import UUID
import logging
import json
from enum import Enum
from dataclasses import dataclass

class MessageStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"

class MessageType(Enum):
    TASK = "task"
    RESPONSE = "response"
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"

@dataclass
class Message:
    """Message data structure"""
    id: UUID
    timestamp: datetime
    from_agent: UUID
    to_agent: UUID
    type: MessageType
    content: Dict[str, Any]
    status: MessageStatus
    channel_id: Optional[str] = None

class AgentCommunication:
    """
    Handles communication between multiple agents in the GENTERR platform.
    Enables agent collaboration and task sharing.
    """

    def __init__(self):
        self.active_channels: Dict[str, Set[UUID]] = {}
        self.message_queue: List[Message] = []
        self.logger = logging.getLogger("agent.communication")
        self.channel_metadata: Dict[str, Dict[str, Any]] = {}

    def validate_message_content(self, content: Dict[str, Any]) -> bool:
        """Validate message content structure"""
        required_fields = ["action", "data"]
        return all(field in content for field in required_fields)

    async def send_message(
        self,
        from_agent: UUID,
        to_agent: UUID,
        message_type: MessageType,
        content: Dict[str, Any],
        channel_id: Optional[str] = None
    ) -> bool:
        """Send a message from one agent to another"""
        try:
            if not self.validate_message_content(content):
                self.logger.error("Invalid message content structure")
                return False

            message = Message(
                id=UUID(),
                timestamp=datetime.utcnow(),
                from_agent=from_agent,
                to_agent=to_agent,
                type=message_type,
                content=content,
                status=MessageStatus.PENDING,
                channel_id=channel_id
            )

            if channel_id and channel_id not in self.active_channels:
                self.logger.error(f"Channel {channel_id} does not exist")
                return False

            self.message_queue.append(message)
            self.logger.info(f"Message sent: {message.id}")
            return True

        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False

    async def create_collaboration_channel(
        self,
        channel_id: str,
        participants: List[UUID],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a new collaboration channel for multiple agents"""
        try:
            if channel_id in self.active_channels:
                self.logger.error(f"Channel {channel_id} already exists")
                return False

            self.active_channels[channel_id] = set(participants)
            self.channel_metadata[channel_id] = metadata or {
                "created_at": datetime.utcnow(),
                "participant_count": len(participants),
                "message_count": 0
            }
            
            self.logger.info(f"Created channel: {channel_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating channel: {str(e)}")
            return False

    def get_pending_messages(self, agent_id: UUID) -> List[Message]:
        """Get all pending messages for a specific agent"""
        return [
            msg for msg in self.message_queue 
            if msg.to_agent == agent_id and msg.status == MessageStatus.PENDING
        ]

    def acknowledge_message(self, message_id: UUID) -> bool:
        """Mark a message as acknowledged"""
        try:
            for msg in self.message_queue:
                if msg.id == message_id:
                    msg.status = MessageStatus.ACKNOWLEDGED
                    self.logger.info(f"Message acknowledged: {message_id}")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging message: {str(e)}")
            return False

    def close_channel(self, channel_id: str) -> bool:
        """Close a collaboration channel"""
        try:
            if channel_id in self.active_channels:
                del self.active_channels[channel_id]
                del self.channel_metadata[channel_id]
                self.logger.info(f"Channel closed: {channel_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error closing channel: {str(e)}")
            return False

    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """Get statistics for a specific channel"""
        if channel_id not in self.active_channels:
            return {}
            
        stats = self.channel_metadata[channel_id].copy()
        stats.update({
            "active_participants": len(self.active_channels[channel_id]),
            "pending_messages": len([
                msg for msg in self.message_queue 
                if msg.channel_id == channel_id and msg.status == MessageStatus.PENDING
            ])
        })
        return stats

    def __str__(self) -> str:
        return f"AgentCommunication(active_channels={len(self.active_channels)}, queued_messages={len(self.message_queue)})"