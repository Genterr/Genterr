from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime

class BaseAgent:
    """
    Base class for all GENTERR AI agents.
    Provides core functionality and interfaces that all agents must implement.
    """
    
    def __init__(self, name: str, description: str = None):
        self.agent_id: UUID = uuid4()
        self.name: str = name
        self.description: str = description or ""
        self.created_at: datetime = datetime.utcnow()
        self.status: str = "initialized"
        self.metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "average_rating": 0.0
        }
        self.capabilities: Dict[str, bool] = {}

    def initialize(self) -> bool:
        """Initialize agent resources and connections"""
        try:
            self.status = "active"
            return True
        except Exception as e:
            self.status = "error"
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a given task and return results
        Must be implemented by specific agent types
        """
        raise NotImplementedError("Subclasses must implement process_task method")

    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """Update agent performance metrics after task completion"""
        self.metrics["tasks_completed"] += 1
        # Additional metric updates based on task_result

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "status": self.status,
            "metrics": self.metrics,
            "capabilities": self.capabilities
        }

    def shutdown(self) -> None:
        """Cleanup and shutdown agent"""
        self.status = "inactive"