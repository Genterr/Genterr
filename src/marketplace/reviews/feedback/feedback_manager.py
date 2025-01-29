# src/marketplace/reviews/feedback/feedback_manager.py

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
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from textblob import TextBlob
from profanity_check import predict_prob

logger = logging.getLogger(__name__)

class FeedbackError(Exception):
    """Base exception for feedback-related errors"""
    pass

class FeedbackNotFoundError(FeedbackError):
    """Raised when feedback cannot be found"""
    pass

class FeedbackValidationError(FeedbackError):
    """Raised when feedback validation fails"""
    pass

class FeedbackStatus(Enum):
    """Status of feedback in the system"""
    PENDING = "pending"       # Awaiting moderation
    PUBLISHED = "published"   # Visible to public
    HIDDEN = "hidden"        # Temporarily hidden
    FLAGGED = "flagged"      # Marked for review
    ARCHIVED = "archived"    # No longer visible
    DELETED = "deleted"      # Soft deleted

class FeedbackType(Enum):
    """Types of feedback supported"""
    TASK_REVIEW = "task_review"          # Review for completed task
    BIDDER_REVIEW = "bidder_review"      # Review for bidder
    COLLABORATION_REVIEW = "collab_review"  # Review for collaboration
    DISPUTE_FEEDBACK = "dispute_feedback"   # Feedback during dispute
    SYSTEM_FEEDBACK = "system_feedback"     # Feedback about platform
    SUPPORT_FEEDBACK = "support_feedback"   # Feedback about support

class SentimentCategory(Enum):
    """Categories for sentiment analysis"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

@dataclass
class RatingMetrics:
    """Metrics for numerical ratings"""
    overall_rating: float
    quality_rating: float
    communication_rating: float
    timeliness_rating: float
    professionalism_rating: float
    cost_rating: Optional[float] = None
    specialized_ratings: Optional[Dict[str, float]] = None

@dataclass
class SentimentAnalysis:
    """Results of sentiment analysis"""
    category: SentimentCategory
    sentiment_score: float
    subjectivity_score: float
    key_phrases: List[str]
    toxicity_probability: float
    confidence_score: float

@dataclass
class Feedback:
    """Individual feedback entry"""
    id: str
    created_at: datetime
    author_id: str
    recipient_id: str
    feedback_type: FeedbackType
    status: FeedbackStatus
    ratings: RatingMetrics
    sentiment: SentimentAnalysis
    content: str
    reference_id: str  # ID of task/collaboration/dispute
    reference_type: str
    attachments: Optional[List[Dict[str, Any]]] = None
    responses: Optional[List[Dict[str, Any]]] = None
    flags: Optional[List[Dict[str, Any]]] = None
    moderation_notes: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

@dataclass
class FeedbackConfig:
    """Configuration for feedback management"""
    database_path: Path
    backup_path: Path
    attachments_path: Path
    min_feedback_length: int = 20
    max_feedback_length: int = 2000
    required_ratings: List[str] = dataclasses.field(
        default_factory=lambda: [
            "overall_rating",
            "quality_rating",
            "communication_rating"
        ]
    )
    enable_sentiment_analysis: bool = True
    toxicity_threshold: float = 0.8
    auto_moderation: bool = True
    max_workers: int = 4
    backup_interval: timedelta = timedelta(hours=24)

class FeedbackManager:
    """
    Manages feedback operations in the marketplace.
    
    This class handles:
    - Feedback creation and validation
    - Rating calculations and aggregation
    - Sentiment analysis and content moderation
    - Response management
    - Analytics and reporting
    - Moderation and flagging
    """

    def __init__(self, config: FeedbackConfig):
        """Initialize FeedbackManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        self.config.attachments_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _setup_logging(self) -> None:
        """Configure logging for feedback management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'feedback_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    ratings TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reference_id TEXT NOT NULL,
                    reference_type TEXT NOT NULL,
                    attachments TEXT,
                    responses TEXT,
                    flags TEXT,
                    moderation_notes TEXT,
                    metadata TEXT,
                    last_updated TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create feedback_history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_history (
                    id TEXT PRIMARY KEY,
                    feedback_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    details TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (feedback_id) REFERENCES feedback(id)
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_author ON feedback(author_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_recipient ON feedback(recipient_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_reference ON feedback(reference_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_feedback_timestamp 
                AFTER UPDATE ON feedback
                BEGIN
                    UPDATE feedback SET updated_timestamp = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)

    async def create_feedback(
        self,
        author_id: str,
        recipient_id: str,
        feedback_type: FeedbackType,
        ratings: RatingMetrics,
        content: str,
        reference_id: str,
        reference_type: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """
        Create new feedback in the system.
        
        Args:
            author_id: ID of feedback author
            recipient_id: ID of feedback recipient
            feedback_type: Type of feedback
            ratings: Numerical ratings
            content: Feedback text content
            reference_id: ID of referenced item
            reference_type: Type of referenced item
            attachments: Optional list of attachments
            metadata: Optional additional metadata
            
        Returns:
            Feedback: The created feedback object
            
        Raises:
            FeedbackValidationError: If validation fails
            FeedbackError: If creation fails
        """
        try:
            # Validate feedback content and ratings
            await self._validate_feedback(
                content,
                ratings,
                feedback_type
            )
            
            # Perform sentiment analysis
            sentiment = await self._analyze_sentiment(content)
            
            # Create feedback object
            feedback = Feedback(
                id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                author_id=author_id,
                recipient_id=recipient_id,
                feedback_type=feedback_type,
                status=FeedbackStatus.PENDING,
                ratings=ratings,
                sentiment=sentiment,
                content=content,
                reference_id=reference_id,
                reference_type=reference_type,
                attachments=attachments,
                responses=[],
                flags=[],
                moderation_notes=[],
                metadata=metadata,
                last_updated=datetime.utcnow()
            )
            
            # Auto-moderate if enabled
            if self.config.auto_moderation:
                await self._auto_moderate_feedback(feedback)
            
            # Save feedback to database
            await self._save_feedback(feedback)
            
            # Log feedback creation
            await self._log_feedback_history(
                feedback.id,
                author_id,
                "create",
                None,
                feedback.status,
                {"sentiment_score": sentiment.sentiment_score}
            )
            
            logger.info(f"Created new feedback: {feedback.id} from {author_id} to {recipient_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to create feedback: {str(e)}")
            raise FeedbackError(f"Feedback creation failed: {str(e)}")
    # 1. Feedback Validation and Auto-Moderation Methods
    async def _validate_feedback(
        self,
        content: str,
        ratings: RatingMetrics,
        feedback_type: FeedbackType
    ) -> None:
        """Validate feedback content and ratings"""
        # Validate content length
        if len(content) < self.config.min_feedback_length:
            raise FeedbackValidationError(
                f"Feedback content too short. Minimum length is "
                f"{self.config.min_feedback_length} characters"
            )
            
        if len(content) > self.config.max_feedback_length:
            raise FeedbackValidationError(
                f"Feedback content too long. Maximum length is "
                f"{self.config.max_feedback_length} characters"
            )
            
        # Validate required ratings
        for required_rating in self.config.required_ratings:
            if not hasattr(ratings, required_rating):
                raise FeedbackValidationError(
                    f"Missing required rating: {required_rating}"
                )
            
            rating_value = getattr(ratings, required_rating)
            if not (0.0 <= rating_value <= 5.0):
                raise FeedbackValidationError(
                    f"Invalid rating value for {required_rating}. "
                    f"Must be between 0.0 and 5.0"
                )

    async def _auto_moderate_feedback(self, feedback: Feedback) -> None:
        """Perform automatic moderation of feedback"""
        # Check toxicity
        if feedback.sentiment.toxicity_probability > self.config.toxicity_threshold:
            feedback.status = FeedbackStatus.FLAGGED
            feedback.moderation_notes.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": "auto_moderation",
                "reason": "high_toxicity",
                "details": {
                    "toxicity_score": feedback.sentiment.toxicity_probability
                }
            })
            return
            
        # Check sentiment
        if feedback.sentiment.category in [
            SentimentCategory.VERY_NEGATIVE,
            SentimentCategory.NEGATIVE
        ]:
            if feedback.sentiment.subjectivity_score > 0.8:
                feedback.status = FeedbackStatus.FLAGGED
                feedback.moderation_notes.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "auto_moderation",
                    "reason": "high_negativity_subjectivity",
                    "details": {
                        "sentiment_score": feedback.sentiment.sentiment_score,
                        "subjectivity_score": feedback.sentiment.subjectivity_score
                    }
                })
                return
                
        # If passed all checks, publish
        feedback.status = FeedbackStatus.PUBLISHED

    # 2. Sentiment Analysis Methods
    async def _analyze_sentiment(self, content: str) -> SentimentAnalysis:
        """Analyze sentiment of feedback content"""
        # Perform TextBlob analysis
        blob = TextBlob(content)
        sentiment_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity
        
        # Determine sentiment category
        category = self._get_sentiment_category(sentiment_score)
        
        # Extract key phrases
        key_phrases = [
            phrase.string.strip()
            for phrase in blob.noun_phrases
        ]
        
        # Check toxicity
        toxicity_prob = float(predict_prob([content])[0])
        
        # Calculate confidence
        confidence = 1.0 - (subjectivity_score * 0.5)
        
        return SentimentAnalysis(
            category=category,
            sentiment_score=sentiment_score,
            subjectivity_score=subjectivity_score,
            key_phrases=key_phrases,
            toxicity_probability=toxicity_prob,
            confidence_score=confidence
        )

    def _get_sentiment_category(self, score: float) -> SentimentCategory:
        """Convert sentiment score to category"""
        if score <= -0.6:
            return SentimentCategory.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentCategory.NEGATIVE
        elif score <= 0.2:
            return SentimentCategory.NEUTRAL
        elif score <= 0.6:
            return SentimentCategory.POSITIVE
        else:
            return SentimentCategory.VERY_POSITIVE

    # 3. Response Management Methods
    async def add_response(
        self,
        feedback_id: str,
        author_id: str,
        content: str,
        response_type: str = "reply",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Add a response to feedback"""
        feedback = await self.get_feedback(feedback_id)
        
        response = {
            "id": str(uuid.uuid4()),
            "author_id": author_id,
            "content": content,
            "type": response_type,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata
        }
        
        if not feedback.responses:
            feedback.responses = []
            
        feedback.responses.append(response)
        feedback.last_updated = datetime.utcnow()
        
        await self._save_feedback(feedback)
        await self._log_feedback_history(
            feedback_id,
            author_id,
            "add_response",
            feedback.status,
            feedback.status,
            {"response_id": response["id"]}
        )
        
        return feedback

    # 4. Analytics and Reporting Methods
    async def generate_feedback_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feedback_type: Optional[FeedbackType] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive feedback analytics report"""
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date.isoformat())
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        df = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        return {
            "total_feedback": len(df),
            "average_ratings": self._calculate_average_ratings(df),
            "sentiment_distribution": self._calculate_sentiment_distribution(df),
            "feedback_type_distribution": df["feedback_type"].value_counts().to_dict(),
            "status_distribution": df["status"].value_counts().to_dict(),
            "response_rate": self._calculate_response_rate(df),
            "average_response_time": self._calculate_avg_response_time(df),
            "flagged_feedback_rate": len(df[df["status"] == FeedbackStatus.FLAGGED.value]) / len(df) if len(df) > 0 else 0,
            "generated_at": datetime.utcnow().isoformat()
        }

    # 5. Moderation and Flagging Methods
    async def flag_feedback(
        self,
        feedback_id: str,
        flagger_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """Flag feedback for moderation review"""
        feedback = await self.get_feedback(feedback_id)
        
        flag = {
            "id": str(uuid.uuid4()),
            "flagger_id": flagger_id,
            "reason": reason,
            "details": details,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat()
        }
        
        if not feedback.flags:
            feedback.flags = []
            
        feedback.flags.append(flag)
        feedback.status = FeedbackStatus.FLAGGED
        feedback.last_updated = datetime.utcnow()
        
        await self._save_feedback(feedback)
        await self._log_feedback_history(
            feedback_id,
            flagger_id,
            "flag",
            FeedbackStatus.PUBLISHED,
            FeedbackStatus.FLAGGED,
            {"flag_id": flag["id"], "reason": reason}
        )
        
        return feedback

    # Helper Methods
    async def get_feedback(self, feedback_id: str) -> Feedback:
        """Retrieve specific feedback by ID"""
        query = "SELECT * FROM feedback WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (feedback_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise FeedbackNotFoundError(f"Feedback {feedback_id} not found")
            
        return self._row_to_feedback(row)

    async def _save_feedback(self, feedback: Feedback) -> None:
        """Save feedback to database"""
        query = """
            INSERT OR REPLACE INTO feedback (
                id, created_at, author_id, recipient_id, feedback_type,
                status, ratings, sentiment, content, reference_id,
                reference_type, attachments, responses, flags,
                moderation_notes, metadata, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            feedback.id,
            feedback.created_at.isoformat(),
            feedback.author_id,
            feedback.recipient_id,
            feedback.feedback_type.value,
            feedback.status.value,
            json.dumps(dataclasses.asdict(feedback.ratings)),
            json.dumps(dataclasses.asdict(feedback.sentiment)),
            feedback.content,
            feedback.reference_id,
            feedback.reference_type,
            json.dumps(feedback.attachments) if feedback.attachments else None,
            json.dumps(feedback.responses) if feedback.responses else None,
            json.dumps(feedback.flags) if feedback.flags else None,
            json.dumps(feedback.moderation_notes) if feedback.moderation_notes else None,
            json.dumps(feedback.metadata) if feedback.metadata else None,
            feedback.last_updated.isoformat() if feedback.last_updated else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _log_feedback_history(
        self,
        feedback_id: str,
        actor_id: str,
        action_type: str,
        previous_status: Optional[FeedbackStatus],
        new_status: FeedbackStatus,
        details: Dict[str, Any]
    ) -> None:
        """Log feedback history entry"""
        query = """
            INSERT INTO feedback_history (
                id, feedback_id, timestamp, actor_id,
                action_type, previous_status, new_status, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            str(uuid.uuid4()),
            feedback_id,
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

    def _row_to_feedback(self, row: sqlite3.Row) -> Feedback:
        """Convert database row to Feedback object"""
        return Feedback(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            author_id=row["author_id"],
            recipient_id=row["recipient_id"],
            feedback_type=FeedbackType(row["feedback_type"]),
            status=FeedbackStatus(row["status"]),
            ratings=RatingMetrics(**json.loads(row["ratings"])),
            sentiment=SentimentAnalysis(**json.loads(row["sentiment"])),
            content=row["content"],
            reference_id=row["reference_id"],
            reference_type=row["reference_type"],
            attachments=json.loads(row["attachments"]) if row["attachments"] else None,
            responses=json.loads(row["responses"]) if row["responses"] else None,
            flags=json.loads(row["flags"]) if row["flags"] else None,
            moderation_notes=json.loads(row["moderation_notes"]) if row["moderation_notes"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None
        )

    def _calculate_average_ratings(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate average ratings from DataFrame"""
        all_ratings = df["ratings"].apply(json.loads)
        
        result = {}
        for rating_type in self.config.required_ratings:
            values = [r.get(rating_type, 0.0) for r in all_ratings]
            result[rating_type] = sum(values) / len(values) if values else 0.0
            
        return result

    def _calculate_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calculate sentiment distribution from DataFrame"""
        sentiments = df["sentiment"].apply(json.loads)
        categories = [s["category"] for s in sentiments]
        return pd.Series(categories).value_counts().to_dict()

    def _calculate_response_rate(self, df: pd.DataFrame) -> float:
        """Calculate response rate from DataFrame"""
        responses = df["responses"].apply(json.loads)
        has_responses = [bool(r) for r in responses]
        return sum(has_responses) / len(has_responses) if has_responses else 0.0

    def _calculate_avg_response_time(self, df: pd.DataFrame) -> float:
        """Calculate average response time in hours"""
        response_times = []
        
        for _, row in df.iterrows():
            created = datetime.fromisoformat(row["created_at"])
            responses = json.loads(row["responses"]) if row["responses"] else []
            
            if responses:
                first_response = datetime.fromisoformat(responses[0]["created_at"])
                response_times.append(
                    (first_response - created).total_seconds() / 3600
                )
                
        return sum(response_times) / len(response_times) if response_times else 0.0