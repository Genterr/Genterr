from datetime import datetime, timedelta, UTC
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from uuid import UUID, uuid4
import logging
import json
from enum import Enum
from dataclasses import dataclass
from threading import Lock
from collections import deque
from logging.handlers import RotatingFileHandler

# Custom exceptions for better error handling
class CommunicationError(Exception):
    """Base exception for communication-related errors"""
    pass

class ChannelError(CommunicationError):
    """Raised when channel operations fail"""
    pass

class MessageError(CommunicationError):
    """Raised when message operations fail"""
    pass

class MessageStatus(Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"

class MessageType(Enum):
    TASK = "task"
    RESPONSE = "response"
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"
    SYSTEM = "system"

@dataclass
class Message:
    """
    Message data structure for agent communication
    
    Attributes:
        id: Unique message identifier
        timestamp: Creation time of the message
        from_agent: Sender agent UUID
        to_agent: Recipient agent UUID
        type: Type of message
        content: Message content
        status: Current message status
        channel_id: Optional channel identifier
        expires_at: Optional expiration timestamp
    """
    id: UUID
    timestamp: datetime
    from_agent: UUID
    to_agent: UUID
    type: MessageType
    content: Dict[str, Any]
    status: MessageStatus
    channel_id: Optional[str] = None
    expires_at: Optional[datetime] = None

@dataclass
class ChannelConfig:
    """Configuration for communication channels"""
    max_participants: int = 100
    message_ttl: int = 3600  # seconds
    max_queue_size: int = 1000
    require_acknowledgment: bool = True

class AgentCommunication:
    """
    Handles communication between multiple agents in the GENTERR platform.
    Enables agent collaboration and task sharing.
    
    Thread-safe implementation with message queue management and error handling.
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        """
        Initialize agent communication system
        
        Args:
            config: Optional channel configuration
        """
        self.config = config or ChannelConfig()
        self.active_channels: Dict[str, Set[UUID]] = {}
        self.message_queue: deque = deque(maxlen=self.config.max_queue_size)
        self.channel_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
        # Enhanced logging setup
        self.logger = logging.getLogger("agent.communication")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = RotatingFileHandler(
                'agent_communication.log',
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def validate_message_content(self, content: Dict[str, Any]) -> bool:
        """
        Validate message content structure
        
        Args:
            content: Message content to validate
            
        Returns:
            bool: True if content is valid
            
        Raises:
            MessageError: If content validation fails
        """
        try:
            required_fields = ["action", "data"]
            if not all(field in content for field in required_fields):
                missing = [f for f in required_fields if f not in content]
                raise MessageError(f"Missing required fields: {missing}")
            
            if not isinstance(content["action"], str):
                raise MessageError("Action must be a string")
                
            return True
        except Exception as e:
            self.logger.error(f"Message validation failed: {str(e)}")
            raise MessageError(f"Content validation failed: {str(e)}")

    async def send_message(
        self,
        from_agent: UUID,
        to_agent: UUID,
        message_type: MessageType,
        content: Dict[str, Any],
        channel_id: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Send a message from one agent to another
        
        Args:
            from_agent: Sender agent UUID
            to_agent: Recipient agent UUID
            message_type: Type of message
            content: Message content
            channel_id: Optional channel identifier
            ttl: Optional time-to-live in seconds
            
        Returns:
            bool: True if message was sent successfully
            
        Raises:
            MessageError: If message sending fails
        """
        try:
            self.validate_message_content(content)

            expires_at = None
            if ttl or self.config.message_ttl:
                expires_at = datetime.now(UTC) + timedelta(
                    seconds=(ttl or self.config.message_ttl)
                )

            message = Message(
                id=uuid4(),
                timestamp=datetime.now(UTC),
                from_agent=from_agent,
                to_agent=to_agent,
                type=message_type,
                content=content,
                status=MessageStatus.PENDING,
                channel_id=channel_id,
                expires_at=expires_at
            )

            with self._lock:
                if channel_id:
                    if channel_id not in self.active_channels:
                        raise ChannelError(f"Channel {channel_id} does not exist")
                    if to_agent not in self.active_channels[channel_id]:
                        raise MessageError(f"Agent {to_agent} not in channel {channel_id}")

                self.message_queue.append(message)
                
                if channel_id:
                    self.channel_metadata[channel_id]["message_count"] += 1

            self.logger.info(f"Message sent: {message.id} via channel: {channel_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            raise MessageError(f"Failed to send message: {str(e)}")

    async def create_collaboration_channel(
        self,
        channel_id: str,
        participants: List[UUID],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collaboration channel for multiple agents
        
        Args:
            channel_id: Unique channel identifier
            participants: List of participant UUIDs
            metadata: Optional channel metadata
            
        Returns:
            bool: True if channel was created successfully
            
        Raises:
            ChannelError: If channel creation fails
        """
        try:
            with self._lock:
                if channel_id in self.active_channels:
                    raise ChannelError(f"Channel {channel_id} already exists")

                if len(participants) > self.config.max_participants:
                    raise ChannelError(
                        f"Maximum participants ({self.config.max_participants}) exceeded"
                    )

                self.active_channels[channel_id] = set(participants)
                self.channel_metadata[channel_id] = {
                    **(metadata or {}),
                    "created_at": datetime.now(UTC),
                    "participant_count": len(participants),
                    "message_count": 0,
                    "last_activity": datetime.now(UTC)
                }
            
            self.logger.info(f"Created channel: {channel_id} with {len(participants)} participants")
            return True

        except Exception as e:
            self.logger.error(f"Error creating channel: {str(e)}")
            raise ChannelError(f"Failed to create channel: {str(e)}")

    def get_pending_messages(self, agent_id: UUID) -> List[Message]:
        """
        Get all pending messages for a specific agent
        
        Args:
            agent_id: Agent UUID to get messages for
            
        Returns:
            List[Message]: List of pending messages
        """
        with self._lock:
            current_time = datetime.now(UTC)
            messages = [
                msg for msg in self.message_queue 
                if msg.to_agent == agent_id 
                and msg.status == MessageStatus.PENDING
                and (not msg.expires_at or msg.expires_at > current_time)
            ]
            
            # Update expired messages
            for msg in self.message_queue:
                if (msg.expires_at 
                    and msg.expires_at <= current_time 
                    and msg.status == MessageStatus.PENDING):
                    msg.status = MessageStatus.EXPIRED

            return messages

    def acknowledge_message(self, message_id: UUID) -> bool:
        """
        Mark a message as acknowledged
        
        Args:
            message_id: UUID of message to acknowledge
            
        Returns:
            bool: True if message was acknowledged
            
        Raises:
            MessageError: If acknowledgment fails
        """
        try:
            with self._lock:
                for msg in self.message_queue:
                    if msg.id == message_id:
                        if msg.status == MessageStatus.EXPIRED:
                            raise MessageError("Cannot acknowledge expired message")
                        msg.status = MessageStatus.ACKNOWLEDGED
                        
                        if msg.channel_id:
                            self.channel_metadata[msg.channel_id]["last_activity"] = datetime.now(UTC)
                            
                        self.logger.info(f"Message acknowledged: {message_id}")
                        return True
                        
                raise MessageError(f"Message {message_id} not found")
                
        except Exception as e:
            self.logger.error(f"Error acknowledging message: {str(e)}")
            raise MessageError(f"Failed to acknowledge message: {str(e)}")

    def close_channel(self, channel_id: str) -> bool:
        """
        Close a collaboration channel
        
        Args:
            channel_id: Channel to close
            
        Returns:
            bool: True if channel was closed
            
        Raises:
            ChannelError: If channel closure fails
        """
        try:
            with self._lock:
                if channel_id not in self.active_channels:
                    raise ChannelError(f"Channel {channel_id} does not exist")
                    
                # Clean up pending messages
                self.message_queue = deque(
                    [msg for msg in self.message_queue if msg.channel_id != channel_id],
                    maxlen=self.config.max_queue_size
                )
                
                del self.active_channels[channel_id]
                del self.channel_metadata[channel_id]
                
            self.logger.info(f"Channel closed: {channel_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing channel: {str(e)}")
            raise ChannelError(f"Failed to close channel: {str(e)}")

    def get_channel_stats(self, channel_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific channel
        
        Args:
            channel_id: Channel to get stats for
            
        Returns:
            Dict[str, Any]: Channel statistics
            
        Raises:
            ChannelError: If channel does not exist
        """
        try:
            with self._lock:
                if channel_id not in self.active_channels:
                    raise ChannelError(f"Channel {channel_id} does not exist")
                
                stats = self.channel_metadata[channel_id].copy()
                current_time = datetime.now(UTC)
                
                stats.update({
                    "active_participants": len(self.active_channels[channel_id]),
                    "pending_messages": len([
                        msg for msg in self.message_queue 
                        if msg.channel_id == channel_id 
                        and msg.status == MessageStatus.PENDING
                        and (not msg.expires_at or msg.expires_at > current_time)
                    ]),
                    "expired_messages": len([
                        msg for msg in self.message_queue
                        if msg.channel_id == channel_id
                        and (
                            msg.status == MessageStatus.EXPIRED
                            or (msg.expires_at and msg.expires_at <= current_time)
                        )
                    ]),
                    "age_seconds": (current_time - stats["created_at"]).total_seconds(),
                    "last_activity_seconds": (
                        current_time - stats["last_activity"]
                    ).total_seconds()
                })
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting channel stats: {str(e)}")
            raise ChannelError(f"Failed to get channel stats: {str(e)}")

    def cleanup_expired_messages(self) -> int:
        """
        Remove expired messages from the queue
        
        Returns:
            int: Number of messages removed
        """
        try:
            with self._lock:
                current_time = datetime.now(UTC)
                original_size = len(self.message_queue)
                
                self.message_queue = deque(
                    [
                        msg for msg in self.message_queue
                        if not msg.expires_at or msg.expires_at > current_time
                    ],
                    maxlen=self.config.max_queue_size
                )
                
                removed = original_size - len(self.message_queue)
                self.logger.info(f"Removed {removed} expired messages")
                return removed
                
        except Exception as e:
            self.logger.error(f"Error cleaning up messages: {str(e)}")
            raise MessageError(f"Failed to cleanup messages: {str(e)}")

    def __str__(self) -> str:
        """String representation of the communication system"""
        return (
            f"AgentCommunication("
            f"active_channels={len(self.active_channels)}, "
            f"queued_messages={len(self.message_queue)}, "
            f"max_queue_size={self.config.max_queue_size}"
            f")"
        )