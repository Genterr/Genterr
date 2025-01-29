# src/marketplace/matching/match_manager.py

from typing import Dict, Any, Optional, List, Union, Tuple, Set, Generator
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
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class MatchingError(Exception):
    """Base exception for matching-related errors"""
    pass

class MatchNotFoundError(MatchingError):
    """Raised when match cannot be found"""
    pass

class MatchValidationError(MatchingError):
    """Raised when match validation fails"""
    pass

class MatchStatus(Enum):
    """Status of a match in the system"""
    PROPOSED = "proposed"      # Initial match proposal
    PENDING = "pending"       # Waiting for acceptance
    ACCEPTED = "accepted"     # Both parties accepted
    REJECTED = "rejected"     # One party rejected
    EXPIRED = "expired"       # Match proposal expired
    COMPLETED = "completed"   # Match successfully completed
    CANCELLED = "cancelled"   # Match cancelled after acceptance
    FAILED = "failed"        # Match failed after acceptance

class MatchType(Enum):
    """Types of matches supported by the system"""
    DIRECT = "direct"                # Direct task-bidder match
    TEAM = "team"                    # Team-based match
    SKILL_BASED = "skill_based"      # Based on skill requirements
    REPUTATION = "reputation"        # Based on reputation scores
    HYBRID = "hybrid"               # Combination of multiple factors
    AI_SUGGESTED = "ai_suggested"    # AI-powered matching

@dataclass
class MatchCriteria:
    """Criteria for match validation and scoring"""
    min_reputation: float
    required_skills: Set[str]
    preferred_skills: Set[str]
    experience_level: int
    min_success_rate: float
    max_active_tasks: int
    location_preferences: Optional[List[str]] = None
    time_zone_preferences: Optional[List[str]] = None
    budget_range: Optional[Tuple[float, float]] = None
    specialized_requirements: Optional[Dict[str, Any]] = None

@dataclass
class MatchScore:
    """Scoring details for a match"""
    overall_score: float
    skill_match_score: float
    reputation_score: float
    success_rate_score: float
    availability_score: float
    budget_fit_score: float
    location_score: Optional[float] = None
    time_zone_score: Optional[float] = None
    specialized_scores: Optional[Dict[str, float]] = None
    confidence_score: float = 1.0

@dataclass
class Match:
    """Individual match entry"""
    id: str
    created_at: datetime
    task_id: str
    bidder_id: str
    match_type: MatchType
    status: MatchStatus
    criteria: MatchCriteria
    score: MatchScore
    expiration: datetime
    task_details: Dict[str, Any]
    bidder_details: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

@dataclass
class MatchConfig:
    """Configuration for match management"""
    database_path: Path
    backup_path: Path
    model_path: Path
    max_matches_per_task: int = 10
    max_matches_per_bidder: int = 20
    match_expiration: timedelta = timedelta(days=3)
    min_match_score: float = 0.6
    enable_ai_matching: bool = True
    max_workers: int = 4
    update_interval: timedelta = timedelta(hours=1)

class MatchManager:
    """
    Manages matching operations in the marketplace.
    
    This class handles:
    - Match creation and validation
    - Score calculation and ranking
    - Match status management
    - AI-powered matching suggestions
    - Analytics and reporting
    - Match history and tracking
    """

    def __init__(self, config: MatchConfig):
        """Initialize MatchManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._vectorizer = TfidfVectorizer(stop_words='english')
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize match scoring models
        if self.config.enable_ai_matching:
            self._init_ai_models()

    def _setup_logging(self) -> None:
        """Configure logging for match management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'match_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create matches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    bidder_id TEXT NOT NULL,
                    match_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    criteria TEXT NOT NULL,
                    score TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    task_details TEXT NOT NULL,
                    bidder_details TEXT NOT NULL,
                    metadata TEXT,
                    last_updated TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create match history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    previous_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_task ON matches(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_bidder ON matches(bidder_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_status ON matches(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_match_type ON matches(match_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_match ON match_history(match_id)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_match_timestamp 
                AFTER UPDATE ON matches
                BEGIN
                    UPDATE matches SET updated_timestamp = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)

    def _init_ai_models(self) -> None:
        """Initialize AI models for match scoring"""
        # Here we would initialize more sophisticated AI models
        # For now, we'll use simple TF-IDF and cosine similarity
        self._skill_vectorizer = TfidfVectorizer(stop_words='english')
        self._description_vectorizer = TfidfVectorizer(stop_words='english')

    async def create_match(
        self,
        task_id: str,
        bidder_id: str,
        match_type: MatchType,
        criteria: MatchCriteria,
        task_details: Dict[str, Any],
        bidder_details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Match:
        """
        Create a new match in the system.
        
        Args:
            task_id: ID of the task to be matched
            bidder_id: ID of the bidder to be matched
            match_type: Type of match
            criteria: Match criteria and requirements
            task_details: Task-related details
            bidder_details: Bidder-related details
            metadata: Optional additional metadata
            
        Returns:
            Match: The created match object
            
        Raises:
            MatchValidationError: If validation fails
            MatchingError: If creation fails
        """
        try:
            # Calculate match score
            score = await self._calculate_match_score(
                criteria,
                task_details,
                bidder_details
            )
            
            # Validate match
            await self._validate_match(
                task_id,
                bidder_id,
                criteria,
                score
            )
            
            # Create match object
            match = Match(
                id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                task_id=task_id,
                bidder_id=bidder_id,
                match_type=match_type,
                status=MatchStatus.PROPOSED,
                criteria=criteria,
                score=score,
                expiration=datetime.utcnow() + self.config.match_expiration,
                task_details=task_details,
                bidder_details=bidder_details,
                metadata=metadata,
                last_updated=datetime.utcnow()
            )
            
            # Save match to database
            await self._save_match(match)
            
            # Log match creation
            await self._log_match_history(
                match.id,
                "system",
                "create",
                None,
                MatchStatus.PROPOSED,
                {"score": dataclasses.asdict(score)}
            )
            
            logger.info(f"Created new match: {match.id} between task {task_id} and bidder {bidder_id}")
            return match
            
        except Exception as e:
            logger.error(f"Failed to create match: {str(e)}")
            raise MatchingError(f"Match creation failed: {str(e)}")
    # 1. Match Scoring and Ranking Methods
    async def _calculate_match_score(
        self,
        criteria: MatchCriteria,
        task_details: Dict[str, Any],
        bidder_details: Dict[str, Any]
    ) -> MatchScore:
        """Calculate comprehensive match score"""
        # Calculate individual scores
        skill_score = self._calculate_skill_match_score(
            criteria.required_skills,
            criteria.preferred_skills,
            bidder_details.get("skills", set())
        )
        
        reputation_score = self._calculate_reputation_score(
            criteria.min_reputation,
            bidder_details.get("reputation", 0.0)
        )
        
        success_rate_score = self._calculate_success_rate_score(
            criteria.min_success_rate,
            bidder_details.get("success_rate", 0.0)
        )
        
        availability_score = self._calculate_availability_score(
            criteria.max_active_tasks,
            bidder_details.get("active_tasks", 0)
        )
        
        budget_score = self._calculate_budget_fit_score(
            criteria.budget_range,
            bidder_details.get("rate_range", (0.0, float("inf")))
        )
        
        # Calculate optional scores
        location_score = None
        if criteria.location_preferences:
            location_score = self._calculate_location_score(
                criteria.location_preferences,
                bidder_details.get("location")
            )
            
        timezone_score = None
        if criteria.time_zone_preferences:
            timezone_score = self._calculate_timezone_score(
                criteria.time_zone_preferences,
                bidder_details.get("timezone")
            )
            
        # Calculate specialized scores if any
        specialized_scores = None
        if criteria.specialized_requirements:
            specialized_scores = self._calculate_specialized_scores(
                criteria.specialized_requirements,
                bidder_details
            )
            
        # Calculate overall score with weighted components
        weights = {
            "skill": 0.35,
            "reputation": 0.20,
            "success_rate": 0.15,
            "availability": 0.15,
            "budget": 0.15
        }
        
        overall_score = (
            weights["skill"] * skill_score +
            weights["reputation"] * reputation_score +
            weights["success_rate"] * success_rate_score +
            weights["availability"] * availability_score +
            weights["budget"] * budget_score
        )
        
        # Apply AI-based confidence score if enabled
        confidence_score = 1.0
        if self.config.enable_ai_matching:
            confidence_score = await self._calculate_ai_confidence_score(
                task_details,
                bidder_details
            )
            
        return MatchScore(
            overall_score=overall_score,
            skill_match_score=skill_score,
            reputation_score=reputation_score,
            success_rate_score=success_rate_score,
            availability_score=availability_score,
            budget_fit_score=budget_score,
            location_score=location_score,
            time_zone_score=timezone_score,
            specialized_scores=specialized_scores,
            confidence_score=confidence_score
        )

    def _calculate_skill_match_score(
        self,
        required_skills: Set[str],
        preferred_skills: Set[str],
        bidder_skills: Set[str]
    ) -> float:
        """Calculate skill match score"""
        if not required_skills:
            return 1.0
            
        # Check required skills
        required_match = len(required_skills & bidder_skills) / len(required_skills)
        if required_match < 1.0:
            return required_match * 0.6  # Penalty for missing required skills
            
        # Calculate preferred skills bonus
        preferred_match = (
            len(preferred_skills & bidder_skills) / len(preferred_skills)
            if preferred_skills else 0.0
        )
        
        return min(1.0, required_match + (preferred_match * 0.4))

    # 2. Match Status Management Methods
    async def update_match_status(
        self,
        match_id: str,
        new_status: MatchStatus,
        actor_id: str,
        reason: str
    ) -> Match:
        """Update match status"""
        match = await self.get_match(match_id)
        
        if not self._is_valid_status_transition(match.status, new_status):
            raise MatchValidationError(
                f"Invalid status transition from {match.status} to {new_status}"
            )
            
        previous_status = match.status
        match.status = new_status
        match.last_updated = datetime.utcnow()
        
        await self._save_match(match)
        await self._log_match_history(
            match_id,
            actor_id,
            "status_change",
            previous_status,
            new_status,
            {"reason": reason}
        )
        
        return match

    # 3. AI-powered Matching Methods
    async def generate_ai_matches(
        self,
        task_id: str,
        task_details: Dict[str, Any],
        criteria: MatchCriteria,
        max_suggestions: int = 5
    ) -> List[Tuple[str, float]]:
        """Generate AI-powered match suggestions"""
        if not self.config.enable_ai_matching:
            raise MatchingError("AI matching is disabled in configuration")
            
        # Get potential bidders
        potential_bidders = await self._get_potential_bidders(criteria)
        
        # Calculate similarity scores
        similarity_scores = []
        for bidder in potential_bidders:
            confidence_score = await self._calculate_ai_confidence_score(
                task_details,
                bidder
            )
            similarity_scores.append((bidder["id"], confidence_score))
            
        # Sort by score and return top matches
        return sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )[:max_suggestions]

    async def _calculate_ai_confidence_score(
        self,
        task_details: Dict[str, Any],
        bidder_details: Dict[str, Any]
    ) -> float:
        """Calculate AI confidence score for a match"""
        # Convert details to feature vectors
        task_vector = self._vectorizer.fit_transform([
            task_details.get("description", "") +
            " " +
            " ".join(task_details.get("requirements", []))
        ])
        
        bidder_vector = self._vectorizer.transform([
            bidder_details.get("bio", "") +
            " " +
            " ".join(bidder_details.get("skills", []))
        ])
        
        # Calculate similarity
        similarity = cosine_similarity(task_vector, bidder_vector)[0][0]
        
        return float(similarity)

    # 4. Analytics and Reporting Methods
    async def generate_matching_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        match_type: Optional[MatchType] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive matching analytics report"""
        query = "SELECT * FROM matches WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())
        if match_type:
            query += " AND match_type = ?"
            params.append(match_type.value)
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        df = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        return {
            "total_matches": len(df),
            "successful_matches": len(df[df["status"] == MatchStatus.COMPLETED.value]),
            "failed_matches": len(df[df["status"] == MatchStatus.FAILED.value]),
            "average_score": df["score"].apply(json.loads).apply(
                lambda x: x["overall_score"]
            ).mean(),
            "status_distribution": df["status"].value_counts().to_dict(),
            "type_distribution": df["match_type"].value_counts().to_dict(),
            "success_rate": (
                len(df[df["status"] == MatchStatus.COMPLETED.value]) / len(df)
                if len(df) > 0 else 0
            ),
            "average_time_to_complete": self._calculate_avg_completion_time(df),
            "generated_at": datetime.utcnow().isoformat()
        }

    # 5. Match History and Tracking Methods
    async def get_match_history(
        self,
        match_id: str
    ) -> List[Dict[str, Any]]:
        """Get complete history for a match"""
        query = """
            SELECT * FROM match_history 
            WHERE match_id = ? 
            ORDER BY timestamp DESC
        """
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (match_id,))
                return cursor.fetchall()
                
        rows = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        return [dict(row) for row in rows]

    # Helper Methods
    async def get_match(self, match_id: str) -> Match:
        """Retrieve a specific match by ID"""
        query = "SELECT * FROM matches WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (match_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise MatchNotFoundError(f"Match {match_id} not found")
            
        return self._row_to_match(row)

    async def _save_match(self, match: Match) -> None:
        """Save match to database"""
        query = """
            INSERT OR REPLACE INTO matches (
                id, created_at, task_id, bidder_id, match_type,
                status, criteria, score, expiration, task_details,
                bidder_details, metadata, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            match.id,
            match.created_at.isoformat(),
            match.task_id,
            match.bidder_id,
            match.match_type.value,
            match.status.value,
            json.dumps(dataclasses.asdict(match.criteria)),
            json.dumps(dataclasses.asdict(match.score)),
            match.expiration.isoformat(),
            json.dumps(match.task_details),
            json.dumps(match.bidder_details),
            json.dumps(match.metadata) if match.metadata else None,
            match.last_updated.isoformat() if match.last_updated else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _log_match_history(
        self,
        match_id: str,
        actor_id: str,
        action_type: str,
        previous_status: Optional[MatchStatus],
        new_status: MatchStatus,
        details: Dict[str, Any]
    ) -> None:
        """Log match history entry"""
        query = """
            INSERT INTO match_history (
                id, match_id, timestamp, actor_id,
                action_type, previous_status, new_status, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            str(uuid.uuid4()),
            match_id,
            datetime.utcnow().isoformat(),
            actor_id,
            action_type,
            previous_status.value if previous_status else None,
            new_status.value,
            json.dumps(details)
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    def _row_to_match(self, row: sqlite3.Row) -> Match:
        """Convert database row to Match object"""
        return Match(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            task_id=row["task_id"],
            bidder_id=row["bidder_id"],
            match_type=MatchType(row["match_type"]),
            status=MatchStatus(row["status"]),
            criteria=MatchCriteria(**json.loads(row["criteria"])),
            score=MatchScore(**json.loads(row["score"])),
            expiration=datetime.fromisoformat(row["expiration"]),
            task_details=json.loads(row["task_details"]),
            bidder_details=json.loads(row["bidder_details"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None
        )

    def _calculate_avg_completion_time(self, df: pd.DataFrame) -> float:
        """Calculate average time to complete matches"""
        completed_matches = df[df["status"] == MatchStatus.COMPLETED.value]
        if len(completed_matches) == 0:
            return 0.0
            
        completion_times = []
        for _, match in completed_matches.iterrows():
            created = datetime.fromisoformat(match["created_at"])
            completed = datetime.fromisoformat(match["last_updated"])
            completion_times.append((completed - created).total_seconds())
            
        return sum(completion_times) / len(completion_times) / 86400  # Convert to days

    def _is_valid_status_transition(
        self,
        current_status: MatchStatus,
        new_status: MatchStatus
    ) -> bool:
        """
        Validate if a status transition is allowed
        
        Valid transitions:
        - PROPOSED -> PENDING, REJECTED, EXPIRED
        - PENDING -> ACCEPTED, REJECTED, EXPIRED
        - ACCEPTED -> COMPLETED, CANCELLED, FAILED
        - Others are terminal states
        """
        valid_transitions = {
            MatchStatus.PROPOSED: {
                MatchStatus.PENDING,
                MatchStatus.REJECTED,
                MatchStatus.EXPIRED
            },
            MatchStatus.PENDING: {
                MatchStatus.ACCEPTED,
                MatchStatus.REJECTED,
                MatchStatus.EXPIRED
            },
            MatchStatus.ACCEPTED: {
                MatchStatus.COMPLETED,
                MatchStatus.CANCELLED,
                MatchStatus.FAILED
            },
            MatchStatus.REJECTED: set(),    # Terminal state
            MatchStatus.EXPIRED: set(),     # Terminal state
            MatchStatus.COMPLETED: set(),   # Terminal state
            MatchStatus.CANCELLED: set(),   # Terminal state
            MatchStatus.FAILED: set()       # Terminal state
        }
        
        return new_status in valid_transitions.get(current_status, set())