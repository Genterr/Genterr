from typing import Dict, List, Any, Optional
from datetime import datetime
from uuid import UUID

class AgentCommunication:
    """
    Handles communication between multiple agents in the GENTERR platform.
    Enables agent collaboration and task sharing.
    """

    def __init__(self):
        self.active_channels: Dict[str, List[UUID]] = {}
        self.message_queue: List[Dict[str, Any]] = []

    async def send_message(
        self,
        from_agent: UUID,
        to_agent: UUID,
        message_type: str,
        content: Dict[str, Any]
    ) -> bool:
        """Send a message from one agent to another"""
        try:
            message = {
                "timestamp": datetime.utcnow(),
                "from_agent": str(from_agent),
                "to_agent": str(to_agent),
                "type": message_type,
                "content": content,
                "status": "pending"
            }
            self.message_queue.append(message)
            return True
        except Exception as e:
            return False

    async def create_collaboration_channel(
        self,
        channel_id: str,
        participants: List[UUID]
    ) -> bool:
        """Create a new collaboration channel for multiple agents"""
        if channel_id not in self.active_channels:
            self.active_channels[channel_id] = participants
            return True
        return False

    def get_pending_messages(self, agent_id: UUID) -> List[Dict[str, Any]]:
        """Get all pending messages for a specific agent"""
        return [
            msg for msg in self.message_queue 
            if msg["to_agent"] == str(agent_id) and msg["status"] == "pending"
        ]

    def acknowledge_message(self, message_id: int) -> None:
        """Mark a message as acknowledged"""
        if 0 <= message_id < len(self.message_queue):
            self.message_queue[message_id]["status"] = "acknowledged"