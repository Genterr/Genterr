# src/marketplace/reviews/ranking/review_manager.py

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ReviewError(Exception):
    """Base exception for review-related errors"""
    pass

class ReviewNotFoundError(ReviewError):
    """Raised when review cannot be found"""
    pass

class ReviewValidationError(ReviewError):
    """Raised when review validation fails"""
    pass

class ReviewStatus(Enum):
    """Status of reviews in the system"""
    DRAFT = "draft"           # Initial draft
    PENDING = "pending"       # Awaiting moderation
    PUBLISHED = "published"   # Visible to public
    HIDDEN = "hidden"         # Temporarily hidden
    FLAGGED = "flagged"      # Marked for review
    ARCHIVED = "archived"     # No longer visible
    DELETED = "deleted"       # Soft deleted

class ReviewType(Enum):
    """Types of reviews supported"""
    PRODUCT = "product"       # Product review
    SERVICE = "service"       # Service review
    SELLER = "seller"        # Seller review
    BUYER = "buyer"          # Buyer review
    PLATFORM = "platform"    # Platform review
    SUPPORT = "support"      # Support review

class RankingAlgorithm(Enum):
    """Supported ranking algorithms"""
    WEIGHTED_AVERAGE = "weighted_average"
    BAYESIAN_AVERAGE = "bayesian_average"
    TIME_DECAY = "time_decay"
    WILSON_SCORE = "wilson_score"
    CUSTOM = "custom"

@dataclass
class ReviewRating:
    """Rating components of a review"""
    overall: float
    quality: float
    communication: float
    reliability: float
    value: float
    accuracy: Optional[float] = None
    specialized: Optional[Dict[str, float]] = None

@dataclass
class ReviewMetrics:
    """Metrics for review ranking"""
    helpfulness_score: float
    relevance_score: float
    quality_score: float
    authenticity_score: float
    recency_score: float
    weighted_score: float
    confidence: float

@dataclass
class Review:
    """Individual review entry"""
    id: str
    created_at: datetime
    author_id: str
    subject_id: str
    review_type: ReviewType
    status: ReviewStatus
    rating: ReviewRating
    content: str
    title: Optional[str] = None
    metrics: Optional[ReviewMetrics] = None
    helpful_votes: int = 0
    unhelpful_votes: int = 0
    responses: Optional[List[Dict[str, Any]]] = None
    flags: Optional[List[Dict[str, Any]]] = None
    moderation_notes: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

@dataclass
class ReviewConfig:
    """Configuration for review management"""
    database_path: Path
    backup_path: Path
    min_content_length: int = 20
    max_content_length: int = 5000
    min_rating: float = 1.0
    max_rating: float = 5.0
    ranking_algorithm: RankingAlgorithm = RankingAlgorithm.WEIGHTED_AVERAGE
    ranking_weights: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "helpfulness": 0.3,
            "relevance": 0.2,
            "quality": 0.2,
            "authenticity": 0.2,
            "recency": 0.1
        }
    )
    enable_auto_moderation: bool = True
    toxicity_threshold: float = 0.8
    max_workers: int = 4
    backup_interval: timedelta = timedelta(hours=24)

class ReviewManager:
    """
    Manages review operations in the marketplace.
    
    This class handles:
    - Review creation and validation
    - Review ranking and scoring
    - Content moderation
    - Vote management
    - Analytics and reporting
    - Ranking algorithm management
    """

    def __init__(self, config: ReviewConfig):
        """Initialize ReviewManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._vectorizer = TfidfVectorizer()
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _setup_logging(self) -> None:
        """Configure logging for review management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'review_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create reviews table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    review_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    content TEXT NOT NULL,
                    title TEXT,
                    metrics TEXT,
                    helpful_votes INTEGER DEFAULT 0,
                    unhelpful_votes INTEGER DEFAULT 0,
                    responses TEXT,
                    flags TEXT,
                    moderation_notes TEXT,
                    metadata TEXT,
                    last_updated TEXT,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create review_history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_history (
                    id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    previous_status TEXT,
                    new_status TEXT,
                    details TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (review_id) REFERENCES reviews(id)
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_author ON reviews(author_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_subject ON reviews(subject_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_type ON reviews(review_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON reviews(status)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_review_timestamp 
                AFTER UPDATE ON reviews
                BEGIN
                    UPDATE reviews SET updated_timestamp = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)
    # 1. Review Creation and Validation Methods
    async def create_review(
        self,
        author_id: str,
        subject_id: str,
        review_type: ReviewType,
        rating: ReviewRating,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Review:
        """Create a new review"""
        try:
            # Validate review content
            await self._validate_review(
                content,
                rating,
                author_id,
                subject_id
            )
            
            # Calculate initial metrics
            metrics = await self._calculate_review_metrics(
                content,
                rating,
                author_id
            )
            
            # Create review
            review = Review(
                id=str(uuid.uuid4()),
                created_at=datetime.utcnow(),
                author_id=author_id,
                subject_id=subject_id,
                review_type=review_type,
                status=ReviewStatus.PENDING,
                rating=rating,
                content=content,
                title=title,
                metrics=metrics,
                metadata=metadata,
                last_updated=datetime.utcnow()
            )
            
            # Auto-moderate if enabled
            if self.config.enable_auto_moderation:
                await self._auto_moderate_review(review)
            
            # Save review
            await self._save_review(review)
            
            # Log review creation
            await self._log_review_history(
                review.id,
                author_id,
                "create",
                None,
                review.status,
                {"metrics": dataclasses.asdict(metrics)}
            )
            
            logger.info(f"Created new review: {review.id} from {author_id}")
            return review
            
        except Exception as e:
            logger.error(f"Failed to create review: {str(e)}")
            raise ReviewError(f"Review creation failed: {str(e)}")

    # 2. Review Ranking and Scoring Methods
    async def rank_reviews(
        self,
        subject_id: str,
        algorithm: Optional[RankingAlgorithm] = None,
        weights: Optional[Dict[str, float]] = None,
        limit: int = 10
    ) -> List[Review]:
        """Rank reviews for a subject using specified algorithm"""
        try:
            # Get all published reviews for subject
            reviews = await self._get_subject_reviews(subject_id)
            
            if not reviews:
                return []
                
            # Use configured algorithm if none specified
            algorithm = algorithm or self.config.ranking_algorithm
            weights = weights or self.config.ranking_weights
            
            # Calculate ranking scores
            ranked_reviews = await self._rank_reviews_with_algorithm(
                reviews,
                algorithm,
                weights
            )
            
            # Return top N reviews
            return ranked_reviews[:limit]
            
        except Exception as e:
            logger.error(f"Failed to rank reviews: {str(e)}")
            raise ReviewError(f"Review ranking failed: {str(e)}")

    async def _rank_reviews_with_algorithm(
        self,
        reviews: List[Review],
        algorithm: RankingAlgorithm,
        weights: Dict[str, float]
    ) -> List[Review]:
        """Apply ranking algorithm to reviews"""
        if algorithm == RankingAlgorithm.WEIGHTED_AVERAGE:
            return await self._rank_by_weighted_average(reviews, weights)
        elif algorithm == RankingAlgorithm.BAYESIAN_AVERAGE:
            return await self._rank_by_bayesian_average(reviews)
        elif algorithm == RankingAlgorithm.TIME_DECAY:
            return await self._rank_by_time_decay(reviews)
        elif algorithm == RankingAlgorithm.WILSON_SCORE:
            return await self._rank_by_wilson_score(reviews)
        elif algorithm == RankingAlgorithm.CUSTOM:
            return await self._rank_by_custom_algorithm(reviews, weights)
        else:
            raise ReviewError(f"Unsupported ranking algorithm: {algorithm}")

    # 3. Vote Management Methods
    async def add_vote(
        self,
        review_id: str,
        user_id: str,
        is_helpful: bool
    ) -> Review:
        """Add helpful/unhelpful vote to review"""
        review = await self.get_review(review_id)
        
        # Update vote counts
        if is_helpful:
            review.helpful_votes += 1
        else:
            review.unhelpful_votes += 1
            
        # Recalculate metrics
        review.metrics = await self._calculate_review_metrics(
            review.content,
            review.rating,
            review.author_id
        )
        
        # Save updated review
        await self._save_review(review)
        
        # Log vote
        await self._log_review_history(
            review_id,
            user_id,
            "vote",
            review.status,
            review.status,
            {"vote_type": "helpful" if is_helpful else "unhelpful"}
        )
        
        return review

    # 4. Content Moderation Methods
    async def _auto_moderate_review(self, review: Review) -> None:
        """Perform automatic moderation of review"""
        # Check content length
        if len(review.content) < self.config.min_content_length:
            review.status = ReviewStatus.FLAGGED
            review.moderation_notes = [{
                "timestamp": datetime.utcnow().isoformat(),
                "type": "auto_moderation",
                "reason": "content_too_short",
                "details": {
                    "length": len(review.content),
                    "min_required": self.config.min_content_length
                }
            }]
            return
            
        # Check for toxicity
        blob = TextBlob(review.content)
        if blob.sentiment.polarity < -0.8:
            review.status = ReviewStatus.FLAGGED
            review.moderation_notes = [{
                "timestamp": datetime.utcnow().isoformat(),
                "type": "auto_moderation",
                "reason": "high_toxicity",
                "details": {
                    "sentiment_polarity": blob.sentiment.polarity,
                    "sentiment_subjectivity": blob.sentiment.subjectivity
                }
            }]
            return
            
        # If passed all checks, set to published
        review.status = ReviewStatus.PUBLISHED

    # 5. Analytics and Reporting Methods
    async def generate_review_analytics(
        self,
        subject_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate analytics report for reviews"""
        reviews = await self._get_subject_reviews(
            subject_id,
            start_time,
            end_time
        )
        
        if not reviews:
            return {
                "total_reviews": 0,
                "average_rating": 0.0,
                "rating_distribution": {},
                "sentiment_distribution": {},
                "review_volume_trend": []
            }
            
        return {
            "total_reviews": len(reviews),
            "average_rating": self._calculate_average_rating(reviews),
            "rating_distribution": self._calculate_rating_distribution(reviews),
            "sentiment_distribution": self._calculate_sentiment_distribution(reviews),
            "review_volume_trend": await self._calculate_volume_trend(reviews),
            "helpful_ratio": self._calculate_helpful_ratio(reviews),
            "top_keywords": await self._extract_top_keywords(reviews),
            "quality_metrics": self._calculate_quality_metrics(reviews),
            "generated_at": datetime.utcnow().isoformat()
        }

    # Helper Methods
    async def get_review(self, review_id: str) -> Review:
        """Get review by ID"""
        query = "SELECT * FROM reviews WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (review_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise ReviewNotFoundError(f"Review {review_id} not found")
            
        return self._row_to_review(row)

    async def _save_review(self, review: Review) -> None:
        """Save review to database"""
        query = """
            INSERT OR REPLACE INTO reviews (
                id, created_at, author_id, subject_id,
                review_type, status, rating, content,
                title, metrics, helpful_votes, unhelpful_votes,
                responses, flags, moderation_notes, metadata,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            review.id,
            review.created_at.isoformat(),
            review.author_id,
            review.subject_id,
            review.review_type.value,
            review.status.value,
            json.dumps(dataclasses.asdict(review.rating)),
            review.content,
            review.title,
            json.dumps(dataclasses.asdict(review.metrics)) if review.metrics else None,
            review.helpful_votes,
            review.unhelpful_votes,
            json.dumps(review.responses) if review.responses else None,
            json.dumps(review.flags) if review.flags else None,
            json.dumps(review.moderation_notes) if review.moderation_notes else None,
            json.dumps(review.metadata) if review.metadata else None,
            review.last_updated.isoformat() if review.last_updated else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    def _row_to_review(self, row: sqlite3.Row) -> Review:
        """Convert database row to Review object"""
        return Review(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            author_id=row["author_id"],
            subject_id=row["subject_id"],
            review_type=ReviewType(row["review_type"]),
            status=ReviewStatus(row["status"]),
            rating=ReviewRating(**json.loads(row["rating"])),
            content=row["content"],
            title=row["title"],
            metrics=ReviewMetrics(**json.loads(row["metrics"])) if row["metrics"] else None,
            helpful_votes=row["helpful_votes"],
            unhelpful_votes=row["unhelpful_votes"],
            responses=json.loads(row["responses"]) if row["responses"] else None,
            flags=json.loads(row["flags"]) if row["flags"] else None,
            moderation_notes=json.loads(row["moderation_notes"]) if row["moderation_notes"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None
        )