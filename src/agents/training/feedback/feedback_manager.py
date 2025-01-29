from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging
from datetime import datetime
import uuid
import asyncio
import aiofiles
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pandas as pd
from scipy import stats

class FeedbackError(Exception):
    """Base exception for feedback-related errors"""
    pass

class FeedbackNotFoundError(FeedbackError):
    """Raised when feedback entry cannot be found"""
    pass

class FeedbackValidationError(FeedbackError):
    """Raised when feedback validation fails"""
    pass

class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    SPEED = "speed"
    SAFETY = "safety"
    CLARITY = "clarity"
    CREATIVITY = "creativity"
    HELPFULNESS = "helpfulness"
    ETHICAL = "ethical"
    CUSTOM = "custom"

class FeedbackSource(Enum):
    """Sources of feedback"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    EXPERT = "expert"
    AUTOMATED = "automated"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    id: str
    timestamp: datetime
    agent_id: str
    model_id: str
    feedback_type: FeedbackType
    source: FeedbackSource
    score: float
    details: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FeedbackConfig:
    """Configuration for feedback management"""
    database_path: Path
    backup_path: Path
    min_score: float = 0.0
    max_score: float = 1.0
    enable_analytics: bool = True
    retention_days: int = 365
    max_workers: int = 4
    batch_size: int = 100

class FeedbackManager:
    """
    Manages feedback collection, storage, and analysis for AI agents.
    Provides mechanisms for tracking performance and improving agent behavior.
    """

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.logger = logging.getLogger("feedback_manager")
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    score REAL NOT NULL,
                    details TEXT NOT NULL,
                    context TEXT,
                    metadata TEXT,
                    CONSTRAINT score_range CHECK (score >= 0 AND score <= 1)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_agent 
                ON feedback(agent_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_model 
                ON feedback(model_id)
            """)

    async def add_feedback(
        self,
        agent_id: str,
        model_id: str,
        feedback_type: FeedbackType,
        source: FeedbackSource,
        score: float,
        details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackEntry:
        """Add new feedback entry"""
        try:
            # Validate score
            if not self.config.min_score <= score <= self.config.max_score:
                raise FeedbackValidationError(
                    f"Score must be between {self.config.min_score} and {self.config.max_score}"
                )
            
            # Create feedback entry
            entry = FeedbackEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                agent_id=agent_id,
                model_id=model_id,
                feedback_type=feedback_type,
                source=source,
                score=score,
                details=details,
                context=context,
                metadata=metadata
            )
            
            # Save to database
            await self._save_feedback(entry)
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to add feedback: {str(e)}")
            raise FeedbackError(f"Failed to add feedback: {str(e)}")

    async def _save_feedback(self, entry: FeedbackEntry) -> None:
        """Save feedback entry to database"""
        query = """
            INSERT INTO feedback 
            (id, timestamp, agent_id, model_id, feedback_type, source, score, details, context, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            entry.id,
            entry.timestamp.isoformat(),
            entry.agent_id,
            entry.model_id,
            entry.feedback_type.value,
            entry.source.value,
            entry.score,
            json.dumps(entry.details),
            json.dumps(entry.context) if entry.context else None,
            json.dumps(entry.metadata) if entry.metadata else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def get_feedback(
        self,
        feedback_id: str
    ) -> FeedbackEntry:
        """Retrieve specific feedback entry"""
        query = "SELECT * FROM feedback WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (feedback_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        if not row:
            raise FeedbackNotFoundError(f"Feedback with id {feedback_id} not found")
            
        return FeedbackEntry(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            agent_id=row['agent_id'],
            model_id=row['model_id'],
            feedback_type=FeedbackType(row['feedback_type']),
            source=FeedbackSource(row['source']),
            score=row['score'],
            details=json.loads(row['details']),
            context=json.loads(row['context']) if row['context'] else None,
            metadata=json.loads(row['metadata']) if row['metadata'] else None
        )

    async def get_agent_feedback(
        self,
        agent_id: str,
        feedback_type: Optional[FeedbackType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[FeedbackEntry]:
        """Get all feedback for specific agent"""
        query = "SELECT * FROM feedback WHERE agent_id = ?"
        params = [agent_id]
        
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return cursor.fetchall()
                
        rows = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        return [
            FeedbackEntry(
                id=row['id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                agent_id=row['agent_id'],
                model_id=row['model_id'],
                feedback_type=FeedbackType(row['feedback_type']),
                source=FeedbackSource(row['source']),
                score=row['score'],
                details=json.loads(row['details']),
                context=json.loads(row['context']) if row['context'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else None
            )
            for row in rows
        ]

    async def analyze_feedback(
        self,
        agent_id: Optional[str] = None,
        model_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze feedback data and generate statistics"""
        if not self.config.enable_analytics:
            raise FeedbackError("Analytics are disabled in configuration")
            
        # Build query
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
            
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
            
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        df = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        if df.empty:
            return {
                "error": "No feedback data found for the specified criteria"
            }
            
        analysis = {
            "total_entries": len(df),
            "unique_agents": df['agent_id'].nunique(),
            "unique_models": df['model_id'].nunique(),
            "feedback_types": df['feedback_type'].value_counts().to_dict(),
            "sources": df['source'].value_counts().to_dict(),
            "score_stats": {
                "mean": df['score'].mean(),
                "median": df['score'].median(),
                "std": df['score'].std(),
                "min": df['score'].min(),
                "max": df['score'].max(),
                "quartiles": df['score'].quantile([0.25, 0.5, 0.75]).to_dict()
            },
            "temporal_analysis": {
                "daily_scores": df.groupby(pd.to_datetime(df['timestamp']).dt.date)['score'].mean().to_dict(),
                "trend": stats.linregress(
                    range(len(df)),
                    df['score'].values
                )._asdict()
            }
        }
        
        return analysis

    async def export_feedback(
        self,
        path: Path,
        format: str = "json",
        compression: bool = True
    ) -> bool:
        """Export feedback data to file"""
        try:
            def _execute():
                with sqlite3.connect(self.config.database_path) as conn:
                    df = pd.read_sql_query("SELECT * FROM feedback", conn)
                    
                if format.lower() == "json":
                    if compression:
                        df.to_json(path.with_suffix('.json.gz'), orient='records', compression='gzip')
                    else:
                        df.to_json(path.with_suffix('.json'), orient='records')
                elif format.lower() == "csv":
                    if compression:
                        df.to_csv(path.with_suffix('.csv.gz'), index=False, compression='gzip')
                    else:
                        df.to_csv(path.with_suffix('.csv'), index=False)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                    
            await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export feedback: {str(e)}")
            return False

    async def backup_database(self) -> bool:
        """Create backup of feedback database"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.config.backup_path / f"feedback_backup_{timestamp}.db"
            
            def _execute():
                with sqlite3.connect(self.config.database_path) as source:
                    backup = sqlite3.connect(backup_file)
                    source.backup(backup)
                    backup.close()
                    
            await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            return False

    async def cleanup_old_feedback(self) -> int:
        """Remove feedback entries older than retention period"""
        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=self.config.retention_days)
            
            def _execute():
                with sqlite3.connect(self.config.database_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM feedback WHERE timestamp < ?",
                        (cutoff_date.isoformat(),)
                    )
                    return cursor.rowcount
                    
            deleted_count = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old feedback: {str(e)}")
            return 0

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return f"FeedbackManager(database={self.config.database_path})"