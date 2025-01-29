# src/marketplace/collaboration/collaboration_manager.py

from typing import Dict, Any, Optional, List, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import uuid
import asyncio
import logging
from pathlib import Path
import json
import sqlite3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import dataclasses

logger = logging.getLogger(__name__)

class CollaborationError(Exception):
    """Base exception for collaboration-related errors"""
    pass

class CollaborationNotFoundError(CollaborationError):
    """Raised when collaboration cannot be found"""
    pass

class CollaborationValidationError(CollaborationError):
    """Raised when collaboration validation fails"""
    pass

class CollaborationStatus(Enum):
    """Status of a collaboration in the system"""
    INITIATED = "initiated"
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    DISPUTED = "disputed"
    ARCHIVED = "archived"

class CollaborationType(Enum):
    """Types of collaboration supported by the system"""
    DIRECT = "direct"              # Direct collaboration between two parties
    TEAM = "team"                  # Team-based collaboration
    PROJECT = "project"            # Project-based collaboration
    MENTORSHIP = "mentorship"      # Mentorship relationship
    CONSULTATION = "consultation"  # Consultation services
    PARTNERSHIP = "partnership"    # Long-term partnership

class CommunicationChannel(Enum):
    """Available communication channels"""
    CHAT = "chat"
    VIDEO = "video"
    VOICE = "voice"
    EMAIL = "email"
    DOCUMENT = "document"
    CODE = "code"

@dataclass
class CollaborationRules:
    """Rules and constraints for collaboration"""
    max_participants: int
    min_reputation: float
    required_skills: Set[str]
    communication_channels: List[CommunicationChannel]
    max_duration: timedelta
    requires_nda: bool
    auto_extension_allowed: bool
    conflict_resolution_protocol: str
    quality_standards: Dict[str, Any]
    milestone_requirements: List[Dict[str, Any]]

@dataclass
class Collaboration:
    """Individual collaboration entry"""
    id: str
    created_at: datetime
    initiator_id: str
    participants: List[str]
    collab_type: CollaborationType
    status: CollaborationStatus
    title: str
    description: str
    rules: CollaborationRules
    start_date: datetime
    end_date: datetime
    milestones: List[Dict[str, Any]]
    resources: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

@dataclass
class CollaborationConfig:
    """Configuration for collaboration management"""
    database_path: Path
    backup_path: Path
    storage_path: Path
    max_active_collaborations: int = 50
    max_participants_per_collab: int = 20
    default_duration: timedelta = timedelta(days=30)
    auto_archive_after: timedelta = timedelta(days=90)
    enable_analytics: bool = True
    max_workers: int = 4

class CollaborationManager:
    """
    Manages collaboration operations in the marketplace.
    
    This class handles:
    - Collaboration creation and validation
    - Participant management
    - Status tracking and updates
    - Resource allocation and sharing
    - Progress monitoring
    - Conflict resolution
    - Analytics and reporting
    """

    def __init__(self, config: CollaborationConfig):
        """Initialize CollaborationManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _setup_logging(self) -> None:
        """Configure logging for collaboration management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'collaboration_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create collaborations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collaborations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    initiator_id TEXT NOT NULL,
                    participants TEXT NOT NULL,
                    collab_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    rules TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    milestones TEXT NOT NULL,
                    resources TEXT NOT NULL,
                    metadata TEXT,
                    last_updated TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create activity log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collaboration_activity (
                    id TEXT PRIMARY KEY,
                    collaboration_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    details TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (collaboration_id) REFERENCES collaborations(id)
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collab_initiator ON collaborations(initiator_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collab_status ON collaborations(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collab_type ON collaborations(collab_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_collab ON collaboration_activity(collaboration_id)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_collaboration_timestamp 
                AFTER UPDATE ON collaborations
                BEGIN
                    UPDATE collaborations SET updated_timestamp = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)

    async def create_collaboration(
        self,
        initiator_id: str,
        title: str,
        description: str,
        collab_type: CollaborationType,
        rules: CollaborationRules,
        participants: List[str],
        start_date: datetime,
        end_date: datetime,
        milestones: List[Dict[str, Any]],
        resources: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Collaboration:
        """
        Create a new collaboration in the system.
        
        Args:
            initiator_id: ID of the user initiating the collaboration
            title: Title of the collaboration
            description: Detailed description
            collab_type: Type of collaboration
            rules: Collaboration rules and constraints
            participants: List of participant IDs
            start_date: Start date of collaboration
            end_date: End date of collaboration
            milestones: List of milestone definitions
            resources: Dictionary of allocated resources
            metadata: Optional additional metadata
            
        Returns:
            Collaboration: The created collaboration object
            
        Raises:
            CollaborationValidationError: If validation fails
            CollaborationError: If creation fails
        """
        try:
            # Validate collaboration
            await self._validate_collaboration(
                initiator_id,
                participants,
                rules,
                start_date,
                end_date
            )
            
            # Create collaboration object
            collaboration = Collaboration(
                id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                initiator_id=initiator_id,
                participants=participants,
                collab_type=collab_type,
                status=CollaborationStatus.INITIATED,
                title=title,
                description=description,
                rules=rules,
                start_date=start_date,
                end_date=end_date,
                milestones=milestones,
                resources=resources,
                metadata=metadata,
                last_updated=datetime.utcnow()
            )
            
            # Save collaboration to database
            await self._save_collaboration(collaboration)
            
            # Log collaboration creation
            await self._log_activity(
                collaboration.id,
                initiator_id,
                "create",
                {"status": "initiated"}
            )
            
            logger.info(f"Created new collaboration: {collaboration.id}")
            return collaboration
            
        except Exception as e:
            logger.error(f"Failed to create collaboration: {str(e)}")
            raise CollaborationError(f"Collaboration creation failed: {str(e)}")

    async def _validate_collaboration(
        self,
        initiator_id: str,
        participants: List[str],
        rules: CollaborationRules,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        Validate collaboration parameters.
        
        Args:
            initiator_id: ID of the initiator
            participants: List of participant IDs
            rules: Collaboration rules
            start_date: Start date
            end_date: End date
            
        Raises:
            CollaborationValidationError: If validation fails
        """
        # Check participant limit
        if len(participants) > self.config.max_participants_per_collab:
            raise CollaborationValidationError(
                f"Maximum participants exceeded. Limit is "
                f"{self.config.max_participants_per_collab}"
            )
        
        # Check dates
        if start_date >= end_date:
            raise CollaborationValidationError(
                "Start date must be before end date"
            )
        
        if (end_date - start_date) > rules.max_duration:
            raise CollaborationValidationError(
                f"Duration exceeds maximum allowed: {rules.max_duration}"
            )
        
        # Check active collaborations limit
        active_count = await self._get_active_collaboration_count(initiator_id)
        if active_count >= self.config.max_active_collaborations:
            raise CollaborationValidationError(
                f"Maximum active collaborations ({self.config.max_active_collaborations}) "
                f"reached for initiator"
            )
    # 1. Collaboration Management Methods
    async def add_participant(
        self,
        collaboration_id: str,
        participant_id: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Collaboration:
        """Add a new participant to the collaboration"""
        collab = await self.get_collaboration(collaboration_id)
        
        if participant_id in collab.participants:
            raise CollaborationValidationError(
                f"Participant {participant_id} already in collaboration"
            )
            
        if len(collab.participants) >= collab.rules.max_participants:
            raise CollaborationValidationError(
                f"Maximum participants ({collab.rules.max_participants}) reached"
            )
            
        collab.participants.append(participant_id)
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            participant_id,
            "join",
            {"role": role, "metadata": metadata}
        )
        
        return collab

    async def remove_participant(
        self,
        collaboration_id: str,
        participant_id: str,
        reason: str
    ) -> Collaboration:
        """Remove a participant from the collaboration"""
        collab = await self.get_collaboration(collaboration_id)
        
        if participant_id not in collab.participants:
            raise CollaborationValidationError(
                f"Participant {participant_id} not in collaboration"
            )
            
        collab.participants.remove(participant_id)
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            participant_id,
            "leave",
            {"reason": reason}
        )
        
        return collab

    async def update_status(
        self,
        collaboration_id: str,
        new_status: CollaborationStatus,
        actor_id: str,
        reason: str
    ) -> Collaboration:
        """Update collaboration status"""
        collab = await self.get_collaboration(collaboration_id)
        
        if not self._is_valid_status_transition(collab.status, new_status):
            raise CollaborationValidationError(
                f"Invalid status transition from {collab.status} to {new_status}"
            )
            
        collab.status = new_status
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            actor_id,
            "status_change",
            {"new_status": new_status.value, "reason": reason}
        )
        
        return collab

    # 2. Resource Management Methods
    async def allocate_resource(
        self,
        collaboration_id: str,
        resource_type: str,
        resource_id: str,
        allocation_details: Dict[str, Any],
        actor_id: str
    ) -> Collaboration:
        """Allocate a resource to the collaboration"""
        collab = await self.get_collaboration(collaboration_id)
        
        if resource_id in collab.resources:
            raise CollaborationValidationError(
                f"Resource {resource_id} already allocated"
            )
            
        collab.resources[resource_id] = {
            "type": resource_type,
            "allocated_at": datetime.utcnow().isoformat(),
            "allocated_by": actor_id,
            **allocation_details
        }
        
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            actor_id,
            "resource_allocation",
            {"resource_id": resource_id, "details": allocation_details}
        )
        
        return collab

    async def release_resource(
        self,
        collaboration_id: str,
        resource_id: str,
        actor_id: str,
        reason: str
    ) -> Collaboration:
        """Release an allocated resource"""
        collab = await self.get_collaboration(collaboration_id)
        
        if resource_id not in collab.resources:
            raise CollaborationValidationError(
                f"Resource {resource_id} not found in collaboration"
            )
            
        released_resource = collab.resources.pop(resource_id)
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            actor_id,
            "resource_release",
            {"resource_id": resource_id, "reason": reason}
        )
        
        return collab

    # 3. Progress Tracking Methods
    async def update_milestone(
        self,
        collaboration_id: str,
        milestone_id: str,
        status: str,
        progress: float,
        actor_id: str,
        notes: Optional[str] = None
    ) -> Collaboration:
        """Update milestone status and progress"""
        collab = await self.get_collaboration(collaboration_id)
        
        milestone_found = False
        for milestone in collab.milestones:
            if milestone["id"] == milestone_id:
                milestone["status"] = status
                milestone["progress"] = progress
                milestone["last_updated"] = datetime.utcnow().isoformat()
                milestone["last_updated_by"] = actor_id
                if notes:
                    milestone["notes"] = notes
                milestone_found = True
                break
                
        if not milestone_found:
            raise CollaborationValidationError(
                f"Milestone {milestone_id} not found"
            )
            
        collab.last_updated = datetime.utcnow()
        
        await self._save_collaboration(collab)
        await self._log_activity(
            collaboration_id,
            actor_id,
            "milestone_update",
            {
                "milestone_id": milestone_id,
                "status": status,
                "progress": progress,
                "notes": notes
            }
        )
        
        return collab

    # 4. Analytics and Reporting Methods
    async def generate_collaboration_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        collab_type: Optional[CollaborationType] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive collaboration analytics report"""
        if not self.config.enable_analytics:
            raise CollaborationError("Analytics are disabled in configuration")
            
        query = "SELECT * FROM collaborations WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())
        if collab_type:
            query += " AND collab_type = ?"
            params.append(collab_type.value)
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        df = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        return {
            "total_collaborations": len(df),
            "active_collaborations": len(df[df["status"] == CollaborationStatus.ACTIVE.value]),
            "completed_collaborations": len(df[df["status"] == CollaborationStatus.COMPLETED.value]),
            "average_duration": (
                pd.to_datetime(df["end_date"]) - pd.to_datetime(df["start_date"])
            ).mean().total_seconds() / 86400,  # Convert to days
            "status_distribution": df["status"].value_counts().to_dict(),
            "type_distribution": df["collab_type"].value_counts().to_dict(),
            "success_rate": (
                len(df[df["status"] == CollaborationStatus.COMPLETED.value]) / len(df)
                if len(df) > 0 else 0
            ),
            "total_participants": df["participants"].apply(json.loads).apply(len).sum(),
            "avg_participants": df["participants"].apply(json.loads).apply(len).mean(),
            "generated_at": datetime.utcnow().isoformat()
        }

    # 5. Backup and Maintenance Methods
    async def archive_inactive_collaborations(self) -> int:
        """Archive collaborations that have been inactive"""
        cutoff_date = datetime.utcnow() - self.config.auto_archive_after
        
        query = """
            UPDATE collaborations 
            SET status = ?, last_updated = ?
            WHERE status IN (?, ?)
            AND last_updated < ?
            AND id IN (
                SELECT id FROM collaborations 
                WHERE status IN (?, ?) 
                AND last_updated < ?
            )
        """
        
        params = (
            CollaborationStatus.ARCHIVED.value,
            datetime.utcnow().isoformat(),
            CollaborationStatus.COMPLETED.value,
            CollaborationStatus.TERMINATED.value,
            cutoff_date.isoformat(),
            CollaborationStatus.COMPLETED.value,
            CollaborationStatus.TERMINATED.value,
            cutoff_date.isoformat()
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
                
        archived_count = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            _execute
        )
        
        logger.info(f"Archived {archived_count} inactive collaborations")
        return archived_count

    # Helper Methods
    async def get_collaboration(self, collaboration_id: str) -> Collaboration:
        """Retrieve a specific collaboration by ID"""
        query = "SELECT * FROM collaborations WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (collaboration_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise CollaborationNotFoundError(f"Collaboration {collaboration_id} not found")
            
        return self._row_to_collaboration(row)

    async def _save_collaboration(self, collab: Collaboration) -> None:
        """Save collaboration to database"""
        query = """
            INSERT OR REPLACE INTO collaborations (
                id, created_at, initiator_id, participants, collab_type,
                status, title, description, rules, start_date, end_date,
                milestones, resources, metadata, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            collab.id,
            collab.created_at.isoformat(),
            collab.initiator_id,
            json.dumps(collab.participants),
            collab.collab_type.value,
            collab.status.value,
            collab.title,
            collab.description,
            json.dumps(dataclasses.asdict(collab.rules)),
            collab.start_date.isoformat(),
            collab.end_date.isoformat(),
            json.dumps(collab.milestones),
            json.dumps(collab.resources),
            json.dumps(collab.metadata) if collab.metadata else None,
            collab.last_updated.isoformat() if collab.last_updated else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _log_activity(
        self,
        collaboration_id: str,
        actor_id: str,
        action_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log collaboration activity"""
        query = """
            INSERT INTO collaboration_activity (
                id, collaboration_id, timestamp, actor_id,
                action_type, details, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            str(uuid.uuid4()),
            collaboration_id,
            datetime.utcnow().isoformat(),
            actor_id,
            action_type,
            json.dumps(details),
            None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _get_active_collaboration_count(self, user_id: str) -> int:
        """Get count of active collaborations for a user"""
        query = """
            SELECT COUNT(*) FROM collaborations 
            WHERE (initiator_id = ? OR participants LIKE ?)
            AND status IN (?, ?)
        """
        
        params = (
            user_id,
            f'%"{user_id}"%',
            CollaborationStatus.INITIATED.value,
            CollaborationStatus.ACTIVE.value
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                return conn.execute(query, params).fetchone()[0]
                
        return await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    def _row_to_collaboration(self, row: sqlite3.Row) -> Collaboration:
        """Convert database row to Collaboration object"""
        return Collaboration(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            initiator_id=row["initiator_id"],
            participants=json.loads(row["participants"]),
            collab_type=CollaborationType(row["collab_type"]),
            status=CollaborationStatus(row["status"]),
            title=row["title"],
            description=row["description"],
            rules=CollaborationRules(**json.loads(row["rules"])),
            start_date=datetime.fromisoformat(row["start_date"]),
            end_date=datetime.fromisoformat(row["end_date"]),
            milestones=json.loads(row["milestones"]),
            resources=json.loads(row["resources"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None
        )

    def _is_valid_status_transition(
        self,
        current_status: CollaborationStatus,
        new_status: CollaborationStatus
    ) -> bool:
        """
        Validate if a status transition is allowed
        
        Valid transitions:
        - INITIATED -> PENDING, TERMINATED
        - PENDING -> ACTIVE, TERMINATED
        - ACTIVE -> PAUSED, COMPLETED, DISPUTED, TERMINATED
        - PAUSED -> ACTIVE, TERMINATED
        - DISPUTED -> ACTIVE, TERMINATED
        - COMPLETED -> ARCHIVED
        - TERMINATED -> ARCHIVED
        """
        valid_transitions = {
            CollaborationStatus.INITIATED: {
                CollaborationStatus.PENDING,
                CollaborationStatus.TERMINATED
            },
            CollaborationStatus.PENDING: {
                CollaborationStatus.ACTIVE,
                CollaborationStatus.TERMINATED
            },
            CollaborationStatus.ACTIVE: {
                CollaborationStatus.PAUSED,
                CollaborationStatus.COMPLETED,
                CollaborationStatus.DISPUTED,
                CollaborationStatus.TERMINATED
            },
            CollaborationStatus.PAUSED: {
                CollaborationStatus.ACTIVE,
                CollaborationStatus.TERMINATED
            },
            CollaborationStatus.DISPUTED: {
                CollaborationStatus.ACTIVE,
                CollaborationStatus.TERMINATED
            },
            CollaborationStatus.COMPLETED: {
                CollaborationStatus.ARCHIVED
            },
            CollaborationStatus.TERMINATED: {
                CollaborationStatus.ARCHIVED
            },
            CollaborationStatus.ARCHIVED: set()  # Terminal state
        }
        
        return new_status in valid_transitions.get(current_status, set())