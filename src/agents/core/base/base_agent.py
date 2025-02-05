import logging
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime, UTC
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Custom exceptions for better error handling
class AgentError(Exception):
    """Base exception for agent-related errors"""
    pass

class MetricsError(AgentError):
    """Raised when metrics update fails"""
    pass

class ValidationError(AgentError):
    """Raised when task validation fails"""
    pass

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
    enable_metrics: bool = True
    metrics_update_threshold: float = 0.0001  # Minimum change to update metrics

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
        """
        Initialize a new agent instance.

        Args:
            name: The name of the agent
            description: Optional description of the agent's purpose
            config: Optional configuration settings
        """
        self.agent_id: UUID = uuid4()
        self.name: str = name
        self.description: str = description or ""
        self.created_at: datetime = datetime.now(UTC)
        self._status: AgentStatus = AgentStatus.INITIALIZED
        self.metrics: Dict[str, Any] = {
            "tasks_completed": 0,
            "success_rate": 0.0,
            "average_rating": 0.0,
            "errors": 0,
            "total_processing_time": 0.0,
            "last_updated": datetime.now(UTC).isoformat()
        }
        self.capabilities: Dict[str, bool] = {}
        self.config = config or AgentConfig()
        
        # Setup logging with enhanced format
        self.logger = logging.getLogger(f"agent.{self.name}")
        self.logger.setLevel(self.config.logging_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @property
    def status(self) -> AgentStatus:
        """Get current agent status"""
        return self._status

    @status.setter
    def status(self, new_status: AgentStatus) -> None:
        """
        Set agent status with logging.
        
        Args:
            new_status: New status to set
        """
        if new_status != self._status:
            self._status = new_status
            self.logger.info(f"Agent {self.name} status changed to {new_status}")

    def initialize(self) -> bool:
        """
        Initialize agent resources and connections.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing agent {self.name}")
            
            # Initialize capabilities
            self.capabilities = {
                "text_processing": True,
                "data_analysis": True,
                "task_execution": True,
                "error_handling": True
            }
            
            # Validate configuration
            if not self._validate_config():
                raise ValueError("Invalid configuration")
                
            # Setup resources
            self._setup_resources()
            
            # Set status to active
            self.status = AgentStatus.ACTIVE
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            self.status = AgentStatus.ERROR
            return False

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a given task and return results.
        Must be implemented by specific agent types.
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            Dictionary containing task results
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement process_task method")

    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate task input.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            bool: True if task is valid, False otherwise
            
        Raises:
            ValidationError: If task is invalid
        """
        try:
            required_fields = ["task_id", "type", "data"]
            if not all(field in task for field in required_fields):
                missing_fields = [f for f in required_fields if f not in task]
                raise ValidationError(f"Missing required fields: {missing_fields}")
            return True
        except Exception as e:
            self.logger.error(f"Task validation failed: {str(e)}")
            raise ValidationError(f"Task validation failed: {str(e)}")

    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """
        Update agent performance metrics after task completion.
        
        Args:
            task_result: Dictionary containing task results
                - success (bool): Whether the task was successful
                - processing_time (float): Time taken to process the task
                - rating (float, optional): Rating of task execution
                
        Raises:
            MetricsError: If metrics update fails
        """
        try:
            if not self.config.enable_metrics:
                return

            # Update task count
            self.metrics["tasks_completed"] += 1
            
            # Update success rate
            success = task_result.get("success", False)
            if success:
                total_tasks = self.metrics["tasks_completed"]
                previous_successes = self.metrics["success_rate"] * (total_tasks - 1)
                new_success_rate = (previous_successes + 1) / total_tasks
                
                if abs(new_success_rate - self.metrics["success_rate"]) >= self.config.metrics_update_threshold:
                    self.metrics["success_rate"] = new_success_rate
            else:
                self.metrics["errors"] += 1
            
            # Update processing time
            if "processing_time" in task_result:
                self.metrics["total_processing_time"] += task_result["processing_time"]
            
            # Update rating if provided
            if "rating" in task_result:
                new_rating = task_result["rating"]
                total_tasks = self.metrics["tasks_completed"]
                previous_rating_total = self.metrics["average_rating"] * (total_tasks - 1)
                new_average_rating = (previous_rating_total + new_rating) / total_tasks
                
                if abs(new_average_rating - self.metrics["average_rating"]) >= self.config.metrics_update_threshold:
                    self.metrics["average_rating"] = new_average_rating
            
            # Update timestamp
            self.metrics["last_updated"] = datetime.utcnow().isoformat()
            
            self.logger.debug(f"Updated metrics for agent {self.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {str(e)}")
            raise MetricsError(f"Metrics update failed: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and metrics.
        
        Returns:
            Dictionary containing current status and metrics
        """
        return {
            "agent_id": str(self.agent_id),
            "name": self.name,
            "status": self.status.value,
            "metrics": self.metrics,
            "capabilities": self.capabilities,
            "uptime": (datetime.utcnow() - self.created_at).total_seconds(),
            "last_updated": datetime.utcnow().isoformat()
        }

    def shutdown(self) -> None:
        """
        Cleanup and shutdown agent.
        
        Raises:
            AgentError: If shutdown fails
        """
        try:
            self.logger.info(f"Shutting down agent {self.name}")
            self.cleanup()  # Call the new cleanup method
            self.status = AgentStatus.INACTIVE
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            self.status = AgentStatus.ERROR
            raise AgentError(f"Shutdown failed: {str(e)}")

    def _validate_config(self) -> bool:
        """Validate agent configuration"""
        try:
            if self.config.max_concurrent_tasks < 1:
                return False
            if self.config.timeout_seconds < 0:
                return False
            if self.config.retry_attempts < 0:
                return False
            return True
        except Exception as e:
            self.logger.error(f"Config validation failed: {str(e)}")
            return False

    def _setup_resources(self) -> None:
        """Setup agent resources"""
        try:
            # Create necessary directories
            self.work_dir = Path(f"work/{self.name}")
            self.work_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize metrics storage
            self._init_metrics()
            
        except Exception as e:
            raise AgentError(f"Failed to setup resources: {str(e)}")

    def _init_metrics(self) -> None:
        """Initialize metrics storage"""
        self.metrics.update({
            "initialization_time": datetime.now(UTC).isoformat(),
            "last_active": datetime.now(UTC).isoformat(),
            "tasks_in_progress": 0
        })

    def cleanup(self) -> None:
        """Cleanup resources before shutdown"""
        try:
            self.logger.info(f"Cleaning up agent {self.name}")
            
            # Save final metrics
            self._save_final_metrics()
            
            # Clean up work directory
            if hasattr(self, 'work_dir') and self.work_dir.exists():
                for file in self.work_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                self.work_dir.rmdir()
                
            self.status = AgentStatus.INACTIVE
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            self.status = AgentStatus.ERROR

    def _save_final_metrics(self) -> None:
        """Save final metrics before shutdown"""
        try:
            self.metrics["shutdown_time"] = datetime.now(UTC).isoformat()
            metrics_file = Path(f"logs/metrics_{self.name}.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save final metrics: {str(e)}")

    def __repr__(self) -> str:
        """Return string representation of the agent"""
        return f"BaseAgent(name='{self.name}', id={self.agent_id}, status={self.status})"