import logging
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add status enum for type safety
class AgentStatus(Enum):
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    INACTIVE = "inactive"
    BUSY = "busy"

@dataclass
class AgentConfig:
    """Configuration settings for the agent"""
    max_concurrent_tasks: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    logging_level: str = "INFO"

class BaseAgent:
    """
    Base class for all GENTERR AI agents.
    Provides core functionality and interfaces that all agents must implement.

    Examples:
        >>> agent = BaseAgent("test_agent")
        >>> agent.initialize()
        True
        >>> agent.status
        <AgentStatus.ACTIVE>
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = None,
        config: AgentConfig = None
    ):
        self.agent_id: UUID = uuid4()
        self.name: str = name
        self.description: str = description or ""
        self.created_at: datetime = datetime.utcnow()
        self._status: AgentStatus = AgentStatus.INITIALIZED
        self.metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "average_rating": 0.0,
            "errors": 0,
            "total_processing_time": 0.0
        }
        self.capabilities: Dict[str, bool] = {}
        self.config = config or AgentConfig()
        
        # Setup logging
        self.logger = logging.getLogger(f"agent.{self.name}")
        self.logger.setLevel(self.config.logging_level)

    @property
    def status(self) -> AgentStatus:
        """Get current agent status"""
        return self._status

    @status.setter
    def status(self, new_status: AgentStatus) -> None:
        """Set agent status with logging"""
        self._status = new_status
        self.logger.info(f"Agent {self.name} status changed to {new_status}")

    def initialize(self) -> bool:
        """Initialize agent resources and connections"""
        try:
            self.logger.info(f"Initializing agent {self.name}")
            # Add initialization logic here
            self.status = AgentStatus.ACTIVE
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.status = AgentStatus.ERROR
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a given task and return results
        Must be implemented by specific agent types
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            Dictionary containing task results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement process_task method")

    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task input"""
        required_fields = ["task_id", "type", "data"]
        return all(field in task for field in required_fields)

    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """Update agent performance metrics after task completion"""
        self.metrics["tasks_completed"] += 1
        
        if task_result.get("success", False):
            current_success = self.metrics["success_rate"] * (self.metrics["tasks_completed"] - 1)
            self.metrics["success_rate"] = (current_success + 1) / self.metrics["tasks_completed"]
        
        if "processing_time" in task_result:
            self.metrics["total_processing_time"] += task_result["processing_time"]

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "status": self.status.value,
            "metrics": self.metrics,
            "capabilities": self.capabilities,
            "uptime": (datetime.utcnow() - self.created_at).total_seconds()
        }

    def shutdown(self) -> None:
        """Cleanup and shutdown agent"""
        try:
            self.logger.info(f"Shutting down agent {self.name}")
            # Add cleanup logic here
            self.status = AgentStatus.INACTIVE
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.status = AgentStatus.ERROR

    def __repr__(self) -> str:
        return f"BaseAgent(name='{self.name}', id={self.agent_id}, status={self.status})"