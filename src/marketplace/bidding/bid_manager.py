# src/marketplace/bidding/bid_manager.py

from typing import Dict, Any, Optional, List, Union, Tuple
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

class BiddingError(Exception):
    """Base exception for bidding-related errors"""
    pass

class BidNotFoundError(BiddingError):
    """Raised when bid cannot be found"""
    pass

class BidValidationError(BiddingError):
    """Raised when bid validation fails"""
    pass

class BidStatus(Enum):
    """Status of a bid in the system"""
    PENDING = "pending"
    ACTIVE = "active"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    COMPLETED = "completed"

class BidType(Enum):
    """Types of bids supported by the system"""
    FIXED_PRICE = "fixed_price"
    AUCTION = "auction"
    REVERSE_AUCTION = "reverse_auction"
    TIME_BASED = "time_based"
    PERFORMANCE_BASED = "performance_based"
    HYBRID = "hybrid"

@dataclass
class BidConstraints:
    """Constraints for bid validation"""
    min_amount: float
    max_amount: float
    min_duration: timedelta
    max_duration: timedelta
    allowed_types: List[BidType]
    required_reputation: float
    min_success_rate: float
    max_concurrent_bids: int

@dataclass
class Bid:
    """Individual bid entry"""
    id: str
    timestamp: datetime
    bidder_id: str
    task_id: str
    bid_type: BidType
    amount: float
    proposed_duration: timedelta
    status: BidStatus
    details: Dict[str, Any]
    expiration: datetime
    requirements: Dict[str, Any]
    constraints: BidConstraints
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BidConfig:
    """Configuration for bid management"""
    database_path: Path
    backup_path: Path
    default_expiration: timedelta = timedelta(days=7)
    max_active_bids: int = 100
    min_bid_interval: timedelta = timedelta(minutes=5)
    enable_analytics: bool = True
    max_workers: int = 4

class BidManager:
    """
    Manages bidding operations in the marketplace.
    
    This class handles:
    - Bid creation and validation
    - Bid status management
    - Bid querying and filtering
    - Analytics and reporting
    - Data persistence and backup
    """

    def __init__(self, config: BidConfig):
        """Initialize BidManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()

    def _setup_logging(self) -> None:
        """Configure logging for bid management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'bid_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create bids table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bids (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    bidder_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    bid_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    proposed_duration INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    expiration TEXT NOT NULL,
                    requirements TEXT NOT NULL,
                    constraints TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bids_bidder ON bids(bidder_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bids_task ON bids(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bids_status ON bids(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bids_expiration ON bids(expiration)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_bid_timestamp 
                AFTER UPDATE ON bids
                BEGIN
                    UPDATE bids SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)

    async def create_bid(
        self,
        bidder_id: str,
        task_id: str,
        bid_type: BidType,
        amount: float,
        proposed_duration: timedelta,
        details: Dict[str, Any],
        requirements: Dict[str, Any],
        constraints: BidConstraints,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Bid:
        """
        Create a new bid in the system.
        
        Args:
            bidder_id: Unique identifier of the bidder
            task_id: Unique identifier of the task
            bid_type: Type of bid being placed
            amount: Bid amount
            proposed_duration: Proposed duration for task completion
            details: Additional bid details
            requirements: Task requirements
            constraints: Bid constraints
            expiration: Optional bid expiration time
            metadata: Optional metadata
            
        Returns:
            Bid: The created bid object
            
        Raises:
            BidValidationError: If bid validation fails
            BiddingError: If bid creation fails
        """
        try:
            # Validate bid
            await self._validate_bid(
                bidder_id,
                amount,
                proposed_duration,
                constraints
            )
            
            # Create bid object
            bid = Bid(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                bidder_id=bidder_id,
                task_id=task_id,
                bid_type=bid_type,
                amount=amount,
                proposed_duration=proposed_duration,
                status=BidStatus.PENDING,
                details=details,
                expiration=expiration or (datetime.utcnow() + self.config.default_expiration),
                requirements=requirements,
                constraints=constraints,
                metadata=metadata
            )
            
            # Save bid to database
            await self._save_bid(bid)
            
            logger.info(f"Created new bid: {bid.id} for task: {task_id}")
            return bid
            
        except Exception as e:
            logger.error(f"Failed to create bid: {str(e)}")
            raise BiddingError(f"Bid creation failed: {str(e)}")

    async def _validate_bid(
        self,
        bidder_id: str,
        amount: float,
        duration: timedelta,
        constraints: BidConstraints
    ) -> None:
        """
        Validate bid against constraints.
        
        Args:
            bidder_id: ID of the bidder
            amount: Bid amount
            duration: Proposed duration
            constraints: Bid constraints
            
        Raises:
            BidValidationError: If validation fails
        """
        # Check amount constraints
        if not constraints.min_amount <= amount <= constraints.max_amount:
            raise BidValidationError(
                f"Bid amount {amount} must be between "
                f"{constraints.min_amount} and {constraints.max_amount}"
            )
        
        # Check duration constraints
        if not constraints.min_duration <= duration <= constraints.max_duration:
            raise BidValidationError(
                f"Proposed duration must be between "
                f"{constraints.min_duration} and {constraints.max_duration}"
            )
        
        # Check concurrent bids limit
        active_bids = await self.get_active_bids(bidder_id)
        if len(active_bids) >= constraints.max_concurrent_bids:
            raise BidValidationError(
                f"Bidder has reached maximum concurrent bids limit of "
                f"{constraints.max_concurrent_bids}"
            )
        
        # Check bid interval
        last_bid = await self._get_last_bid(bidder_id)
        if last_bid and (datetime.utcnow() - last_bid.timestamp) < self.config.min_bid_interval:
            raise BidValidationError(
                f"Must wait {self.config.min_bid_interval} between bids"
            )

    # 1. Bid retrieval and querying methods
    async def get_bid(self, bid_id: str) -> Bid:
        """Retrieve a specific bid by ID"""
        query = "SELECT * FROM bids WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (bid_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise BidNotFoundError(f"Bid {bid_id} not found")
            
        return self._row_to_bid(row)

    async def get_active_bids(self, bidder_id: str) -> List[Bid]:
        """Get all active bids for a bidder"""
        query = """
            SELECT * FROM bids 
            WHERE bidder_id = ? 
            AND status IN (?, ?)
            ORDER BY timestamp DESC
        """
        params = (
            bidder_id,
            BidStatus.PENDING.value,
            BidStatus.ACTIVE.value
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return cursor.fetchall()
                
        rows = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        return [self._row_to_bid(row) for row in rows]

    # 2. Bid status management
    async def update_bid_status(
        self,
        bid_id: str,
        new_status: BidStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Bid:
        """Update the status of a bid"""
        bid = await self.get_bid(bid_id)
        
        query = """
            UPDATE bids 
            SET status = ?, metadata = ?
            WHERE id = ?
        """
        params = (
            new_status.value,
            json.dumps(metadata) if metadata else None,
            bid_id
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        return await self.get_bid(bid_id)

    # 3. Analytics and reporting
    async def generate_bidding_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive bidding report"""
        if not self.config.enable_analytics:
            raise BiddingError("Analytics are disabled in configuration")
            
        query = "SELECT * FROM bids WHERE 1=1"
        params = []
        
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
        
        return {
            "total_bids": len(df),
            "total_value": df["amount"].sum(),
            "avg_bid_amount": df["amount"].mean(),
            "status_distribution": df["status"].value_counts().to_dict(),
            "bid_types": df["bid_type"].value_counts().to_dict(),
            "success_rate": (
                df[df["status"] == BidStatus.COMPLETED.value].shape[0] / len(df)
                if len(df) > 0 else 0
            ),
            "avg_duration": df["proposed_duration"].mean(),
            "timestamp": datetime.utcnow().isoformat()
        }

    # 4. Data export and backup
    async def export_bids(
        self,
        format: str = "json",
        path: Optional[Path] = None,
        compress: bool = True
    ) -> Path:
        """Export bid data to file"""
        if not path:
            path = self.config.backup_path / f"bids_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                df = pd.read_sql_query("SELECT * FROM bids", conn)
                
            if format.lower() == "json":
                if compress:
                    df.to_json(f"{path}.json.gz", orient="records", compression="gzip")
                    return Path(f"{path}.json.gz")
                df.to_json(f"{path}.json", orient="records")
                return Path(f"{path}.json")
            elif format.lower() == "csv":
                if compress:
                    df.to_csv(f"{path}.csv.gz", index=False, compression="gzip")
                    return Path(f"{path}.csv.gz")
                df.to_csv(f"{path}.csv", index=False)
                return Path(f"{path}.csv")
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        return await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    # 5. Cleanup and maintenance operations
    async def cleanup_expired_bids(self) -> int:
        """Update status of expired bids"""
        query = """
            UPDATE bids 
            SET status = ?
            WHERE status IN (?, ?)
            AND expiration < ?
            AND id IN (
                SELECT id FROM bids 
                WHERE status IN (?, ?) 
                AND expiration < ?
            )
        """
        
        now = datetime.utcnow().isoformat()
        params = (
            BidStatus.EXPIRED.value,
            BidStatus.PENDING.value,
            BidStatus.ACTIVE.value,
            now,
            BidStatus.PENDING.value,
            BidStatus.ACTIVE.value,
            now
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
                
        updated_count = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            _execute
        )
        
        logger.info(f"Cleaned up {updated_count} expired bids")
        return updated_count

    # Helper methods
    def _row_to_bid(self, row: sqlite3.Row) -> Bid:
        """Convert database row to Bid object"""
        return Bid(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bidder_id=row["bidder_id"],
            task_id=row["task_id"],
            bid_type=BidType(row["bid_type"]),
            amount=float(row["amount"]),
            proposed_duration=timedelta(seconds=int(row["proposed_duration"])),
            status=BidStatus(row["status"]),
            details=json.loads(row["details"]),
            expiration=datetime.fromisoformat(row["expiration"]),
            requirements=json.loads(row["requirements"]),
            constraints=BidConstraints(**json.loads(row["constraints"])),
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )

    async def _save_bid(self, bid: Bid) -> None:
        """Save bid to database"""
        query = """
            INSERT INTO bids (
                id, timestamp, bidder_id, task_id, bid_type, amount,
                proposed_duration, status, details, expiration,
                requirements, constraints, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            bid.id,
            bid.timestamp.isoformat(),
            bid.bidder_id,
            bid.task_id,
            bid.bid_type.value,
            bid.amount,
            int(bid.proposed_duration.total_seconds()),
            bid.status.value,
            json.dumps(bid.details),
            bid.expiration.isoformat(),
            json.dumps(bid.requirements),
            json.dumps(dataclasses.asdict(bid.constraints)),
            json.dumps(bid.metadata) if bid.metadata else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _get_last_bid(self, bidder_id: str) -> Optional[Bid]:
        """Get the last bid made by a bidder"""
        query = """
            SELECT * FROM bids 
            WHERE bidder_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (bidder_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        return self._row_to_bid(row) if row else None

    def _is_valid_status_transition(self, current_status: BidStatus, new_status: BidStatus) -> bool:
        """
        Validate if a status transition is allowed
        
        Valid transitions:
        - PENDING -> ACTIVE, REJECTED, CANCELLED
        - ACTIVE -> ACCEPTED, REJECTED, CANCELLED
        - Any -> EXPIRED (handled by cleanup)
        - ACCEPTED -> COMPLETED
        """
        valid_transitions = {
            BidStatus.PENDING: {BidStatus.ACTIVE, BidStatus.REJECTED, BidStatus.CANCELLED, BidStatus.EXPIRED},
            BidStatus.ACTIVE: {BidStatus.ACCEPTED, BidStatus.REJECTED, BidStatus.CANCELLED, BidStatus.EXPIRED},
            BidStatus.ACCEPTED: {BidStatus.COMPLETED, BidStatus.CANCELLED},
            BidStatus.REJECTED: set(),  # Terminal state
            BidStatus.CANCELLED: set(),  # Terminal state
            BidStatus.EXPIRED: set(),    # Terminal state
            BidStatus.COMPLETED: set()   # Terminal state
        }
        
        return new_status in valid_transitions.get(current_status, set())