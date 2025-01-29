# src/marketplace/reviews/metrics/metrics_manager.py

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
from statistics import mean, median, stdev
from scipy import stats

logger = logging.getLogger(__name__)

class MetricsError(Exception):
    """Base exception for metrics-related errors"""
    pass

class MetricsNotFoundError(MetricsError):
    """Raised when metrics cannot be found"""
    pass

class MetricsValidationError(MetricsError):
    """Raised when metrics validation fails"""
    pass

class MetricType(Enum):
    """Types of metrics supported by the system"""
    PERFORMANCE = "performance"      # Task completion, quality, etc.
    REPUTATION = "reputation"        # User reputation scores
    ENGAGEMENT = "engagement"        # Platform engagement metrics
    SATISFACTION = "satisfaction"    # User satisfaction metrics
    FINANCIAL = "financial"         # Financial performance metrics
    OPERATIONAL = "operational"     # System operational metrics

class MetricCategory(Enum):
    """Categories for metric classification"""
    USER = "user"              # User-specific metrics
    TASK = "task"             # Task-related metrics
    PLATFORM = "platform"      # Platform-wide metrics
    FINANCIAL = "financial"    # Financial metrics
    TECHNICAL = "technical"    # Technical performance metrics
    CUSTOM = "custom"         # Custom defined metrics

class MetricPeriod(Enum):
    """Time periods for metric calculation"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"

@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    value: float
    timestamp: datetime
    confidence: float
    source: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    id: str
    name: str
    description: str
    type: MetricType
    category: MetricCategory
    unit: str
    period: MetricPeriod
    aggregation_method: str
    calculation_formula: str
    thresholds: Dict[str, float]
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricSeries:
    """Time series of metric values"""
    metric_id: str
    values: List[MetricValue]
    start_time: datetime
    end_time: datetime
    period: MetricPeriod
    statistics: Dict[str, float]
    trends: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MetricsConfig:
    """Configuration for metrics management"""
    database_path: Path
    backup_path: Path
    export_path: Path
    calculation_interval: Dict[MetricPeriod, timedelta]
    retention_period: Dict[MetricPeriod, timedelta]
    enable_anomaly_detection: bool = True
    confidence_threshold: float = 0.8
    max_workers: int = 4
    backup_interval: timedelta = timedelta(hours=24)

class MetricsManager:
    """
    Manages metrics operations in the marketplace.
    
    This class handles:
    - Metric definition and validation
    - Metric calculation and aggregation
    - Time series management
    - Statistical analysis
    - Anomaly detection
    - Reporting and visualization
    """

    def __init__(self, config: MetricsConfig):
        """Initialize MetricsManager with configuration"""
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Setup logging
        self._setup_logging()
        
        # Create necessary directories
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backup_path.mkdir(parents=True, exist_ok=True)
        self.config.export_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize metric definitions cache
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._load_metric_definitions()

    def _setup_logging(self) -> None:
        """Configure logging for metrics management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.config.backup_path / 'metrics_manager.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.config.database_path) as conn:
            # Create metric_definitions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    period TEXT NOT NULL,
                    aggregation_method TEXT NOT NULL,
                    calculation_formula TEXT NOT NULL,
                    thresholds TEXT NOT NULL,
                    dependencies TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create metric_values table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_values (
                    id TEXT PRIMARY KEY,
                    metric_id TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (metric_id) REFERENCES metric_definitions(id)
                )
            """)
            
            # Create metric_series table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_series (
                    id TEXT PRIMARY KEY,
                    metric_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    period TEXT NOT NULL,
                    statistics TEXT NOT NULL,
                    trends TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (metric_id) REFERENCES metric_definitions(id)
                )
            """)
            
            # Create indices for frequent queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_type ON metric_definitions(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_category ON metric_definitions(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_value_metric ON metric_values(metric_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_value_timestamp ON metric_values(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_series_metric ON metric_series(metric_id)")
            
            # Create trigger for updating timestamps
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS update_metric_definition_timestamp 
                AFTER UPDATE ON metric_definitions
                BEGIN
                    UPDATE metric_definitions SET updated_at = CURRENT_TIMESTAMP 
                    WHERE id = NEW.id;
                END;
            """)

    def _load_metric_definitions(self) -> None:
        """Load metric definitions into cache"""
        with sqlite3.connect(self.config.database_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM metric_definitions")
            for row in cursor:
                metric_def = MetricDefinition(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"],
                    type=MetricType(row["type"]),
                    category=MetricCategory(row["category"]),
                    unit=row["unit"],
                    period=MetricPeriod(row["period"]),
                    aggregation_method=row["aggregation_method"],
                    calculation_formula=row["calculation_formula"],
                    thresholds=json.loads(row["thresholds"]),
                    dependencies=json.loads(row["dependencies"]) if row["dependencies"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                )
                self._metric_definitions[metric_def.id] = metric_def
    # 1. Metric Definition and Management Methods
    async def create_metric_definition(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        category: MetricCategory,
        unit: str,
        period: MetricPeriod,
        aggregation_method: str,
        calculation_formula: str,
        thresholds: Dict[str, float],
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MetricDefinition:
        """Create a new metric definition"""
        try:
            # Validate metric definition
            await self._validate_metric_definition(
                name,
                calculation_formula,
                dependencies
            )
            
            # Create metric definition
            metric_def = MetricDefinition(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                type=metric_type,
                category=category,
                unit=unit,
                period=period,
                aggregation_method=aggregation_method,
                calculation_formula=calculation_formula,
                thresholds=thresholds,
                dependencies=dependencies,
                metadata=metadata
            )
            
            # Save to database
            await self._save_metric_definition(metric_def)
            
            # Update cache
            self._metric_definitions[metric_def.id] = metric_def
            
            logger.info(f"Created new metric definition: {metric_def.id} - {name}")
            return metric_def
            
        except Exception as e:
            logger.error(f"Failed to create metric definition: {str(e)}")
            raise MetricsError(f"Metric definition creation failed: {str(e)}")

    # 2. Metric Calculation and Aggregation Methods
    async def calculate_metric(
        self,
        metric_id: str,
        parameters: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> MetricValue:
        """Calculate a metric value"""
        metric_def = await self.get_metric_definition(metric_id)
        
        try:
            # Get dependent values if any
            dependent_values = {}
            if metric_def.dependencies:
                for dep_id in metric_def.dependencies:
                    dep_value = await self.get_latest_metric_value(dep_id)
                    dependent_values[dep_id] = dep_value.value
            
            # Calculate metric value
            value = await self._execute_calculation(
                metric_def.calculation_formula,
                parameters,
                dependent_values
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                value,
                metric_def.thresholds,
                parameters
            )
            
            # Create metric value
            metric_value = MetricValue(
                value=value,
                timestamp=timestamp or datetime.utcnow(),
                confidence=confidence,
                source="calculation",
                metadata={"parameters": parameters}
            )
            
            # Save metric value
            await self._save_metric_value(metric_id, metric_value)
            
            return metric_value
            
        except Exception as e:
            logger.error(f"Failed to calculate metric {metric_id}: {str(e)}")
            raise MetricsError(f"Metric calculation failed: {str(e)}")

    # 3. Time Series Management Methods
    async def create_metric_series(
        self,
        metric_id: str,
        start_time: datetime,
        end_time: datetime,
        period: MetricPeriod
    ) -> MetricSeries:
        """Create a time series for a metric"""
        # Get metric values for the period
        values = await self._get_metric_values(
            metric_id,
            start_time,
            end_time
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(values)
        
        # Calculate trends
        trends = self._calculate_trends(values)
        
        # Create series
        series = MetricSeries(
            metric_id=metric_id,
            values=values,
            start_time=start_time,
            end_time=end_time,
            period=period,
            statistics=statistics,
            trends=trends,
            metadata={
                "sample_size": len(values),
                "generated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Save series
        await self._save_metric_series(series)
        
        return series

    # 4. Statistical Analysis Methods
    def _calculate_statistics(
        self,
        values: List[MetricValue]
    ) -> Dict[str, float]:
        """Calculate statistical measures for metric values"""
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0
            }
            
        raw_values = [v.value for v in values]
        
        return {
            "count": len(raw_values),
            "mean": float(mean(raw_values)),
            "median": float(median(raw_values)),
            "std_dev": float(stdev(raw_values)) if len(raw_values) > 1 else 0.0,
            "min": float(min(raw_values)),
            "max": float(max(raw_values))
        }

    def _calculate_trends(
        self,
        values: List[MetricValue]
    ) -> Dict[str, float]:
        """Calculate trend indicators for metric values"""
        if len(values) < 2:
            return {
                "slope": 0.0,
                "r_squared": 0.0,
                "trend_strength": 0.0
            }
            
        times = [(v.timestamp - values[0].timestamp).total_seconds() 
                for v in values]
        values_array = [v.value for v in values]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            times,
            values_array
        )
        
        return {
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "trend_strength": float(abs(r_value))
        }

    # 5. Anomaly Detection Methods
    async def detect_anomalies(
        self,
        metric_id: str,
        window_size: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values"""
        if not self.config.enable_anomaly_detection:
            return []
            
        # Get recent values
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=window_size)
        values = await self._get_metric_values(metric_id, start_time, end_time)
        
        if len(values) < 3:  # Need minimum points for detection
            return []
            
        # Calculate z-scores
        raw_values = [v.value for v in values]
        z_scores = stats.zscore(raw_values)
        
        # Detect anomalies (z-score > 3)
        anomalies = []
        for i, (value, z_score) in enumerate(zip(values, z_scores)):
            if abs(z_score) > 3:
                anomalies.append({
                    "timestamp": value.timestamp.isoformat(),
                    "value": value.value,
                    "z_score": float(z_score),
                    "confidence": min(1.0, abs(z_score) / 5.0),
                    "metadata": value.metadata
                })
                
        return anomalies

    # 6. Reporting and Export Methods
    async def generate_metrics_report(
        self,
        metric_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "metrics": {}
        }
        
        for metric_id in metric_ids:
            metric_def = await self.get_metric_definition(metric_id)
            series = await self.create_metric_series(
                metric_id,
                start_time,
                end_time,
                MetricPeriod.DAILY
            )
            
            anomalies = []
            if self.config.enable_anomaly_detection:
                anomalies = await self.detect_anomalies(metric_id)
            
            report["metrics"][metric_id] = {
                "definition": dataclasses.asdict(metric_def),
                "statistics": series.statistics,
                "trends": series.trends,
                "anomalies": anomalies
            }
        
        if format == "json":
            return report
        elif format == "csv":
            return self._convert_report_to_csv(report)
        else:
            raise MetricsError(f"Unsupported report format: {format}")

    # Helper Methods
    async def get_metric_definition(
        self,
        metric_id: str
    ) -> MetricDefinition:
        """Get metric definition by ID"""
        if metric_id in self._metric_definitions:
            return self._metric_definitions[metric_id]
            
        query = "SELECT * FROM metric_definitions WHERE id = ?"
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (metric_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise MetricsNotFoundError(f"Metric definition {metric_id} not found")
            
        metric_def = MetricDefinition(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            type=MetricType(row["type"]),
            category=MetricCategory(row["category"]),
            unit=row["unit"],
            period=MetricPeriod(row["period"]),
            aggregation_method=row["aggregation_method"],
            calculation_formula=row["calculation_formula"],
            thresholds=json.loads(row["thresholds"]),
            dependencies=json.loads(row["dependencies"]) if row["dependencies"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )
        
        self._metric_definitions[metric_id] = metric_def
        return metric_def

    async def get_latest_metric_value(
        self,
        metric_id: str
    ) -> MetricValue:
        """Get latest value for a metric"""
        query = """
            SELECT * FROM metric_values 
            WHERE metric_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (metric_id,))
                return cursor.fetchone()
                
        row = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        if not row:
            raise MetricsNotFoundError(f"No values found for metric {metric_id}")
            
        return MetricValue(
            value=row["value"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            confidence=row["confidence"],
            source=row["source"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )

    async def _save_metric_definition(
        self,
        metric_def: MetricDefinition
    ) -> None:
        """Save metric definition to database"""
        query = """
            INSERT OR REPLACE INTO metric_definitions (
                id, name, description, type, category,
                unit, period, aggregation_method,
                calculation_formula, thresholds,
                dependencies, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            metric_def.id,
            metric_def.name,
            metric_def.description,
            metric_def.type.value,
            metric_def.category.value,
            metric_def.unit,
            metric_def.period.value,
            metric_def.aggregation_method,
            metric_def.calculation_formula,
            json.dumps(metric_def.thresholds),
            json.dumps(metric_def.dependencies) if metric_def.dependencies else None,
            json.dumps(metric_def.metadata) if metric_def.metadata else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _save_metric_value(
        self,
        metric_id: str,
        value: MetricValue
    ) -> None:
        """Save metric value to database"""
        query = """
            INSERT INTO metric_values (
                id, metric_id, value, timestamp,
                confidence, source, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            str(uuid.uuid4()),
            metric_id,
            value.value,
            value.timestamp.isoformat(),
            value.confidence,
            value.source,
            json.dumps(value.metadata) if value.metadata else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)

    async def _save_metric_series(
        self,
        series: MetricSeries
    ) -> None:
        """Save metric series to database"""
        query = """
            INSERT INTO metric_series (
                id, metric_id, start_time, end_time,
                period, statistics, trends, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            str(uuid.uuid4()),
            series.metric_id,
            series.start_time.isoformat(),
            series.end_time.isoformat(),
            series.period.value,
            json.dumps(series.statistics),
            json.dumps(series.trends),
            json.dumps(series.metadata) if series.metadata else None
        )
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.execute(query, params)
                conn.commit()
                
        await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
    async def _validate_metric_definition(
        self,
        name: str,
        calculation_formula: str,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """Validate metric definition parameters"""
        # Check name uniqueness
        for metric_def in self._metric_definitions.values():
            if metric_def.name == name:
                raise MetricsValidationError(f"Metric name '{name}' already exists")
        
        # Validate calculation formula
        if not calculation_formula or len(calculation_formula.strip()) == 0:
            raise MetricsValidationError("Calculation formula cannot be empty")
            
        # Validate dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self._metric_definitions:
                    raise MetricsValidationError(
                        f"Dependent metric {dep_id} does not exist"
                    )

    async def _execute_calculation(
        self,
        formula: str,
        parameters: Dict[str, Any],
        dependent_values: Dict[str, float]
    ) -> float:
        """Execute metric calculation formula"""
        try:
            # Create local namespace with parameters and dependent values
            namespace = {
                **parameters,
                **dependent_values,
                "np": np,
                "math": __import__("math")
            }
            
            # Execute formula in isolated namespace
            return float(eval(formula, {"__builtins__": {}}, namespace))
            
        except Exception as e:
            raise MetricsError(f"Calculation failed: {str(e)}")

    def _calculate_confidence(
        self,
        value: float,
        thresholds: Dict[str, float],
        parameters: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for metric value"""
        # Basic range check
        min_val = thresholds.get("min", float("-inf"))
        max_val = thresholds.get("max", float("inf"))
        
        if value < min_val or value > max_val:
            return 0.0
            
        # Calculate relative position within expected range
        expected_range = thresholds.get("expected_range", 0.0)
        if expected_range > 0:
            deviation = abs(value - thresholds.get("expected_value", value))
            confidence = max(0.0, 1.0 - (deviation / expected_range))
            return confidence
            
        return 1.0

    async def _get_metric_values(
        self,
        metric_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MetricValue]:
        """Get metric values for a time period"""
        query = """
            SELECT * FROM metric_values 
            WHERE metric_id = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        
        def _execute():
            with sqlite3.connect(self.config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    query,
                    (
                        metric_id,
                        start_time.isoformat(),
                        end_time.isoformat()
                    )
                )
                return cursor.fetchall()
                
        rows = await asyncio.get_event_loop().run_in_executor(self._executor, _execute)
        
        return [
            MetricValue(
                value=row["value"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                confidence=row["confidence"],
                source=row["source"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None
            )
            for row in rows
        ]

    def _convert_report_to_csv(self, report: Dict[str, Any]) -> str:
        """Convert metrics report to CSV format"""
        # Create DataFrame from report
        rows = []
        for metric_id, metric_data in report["metrics"].items():
            row = {
                "metric_id": metric_id,
                "name": metric_data["definition"]["name"],
                "type": metric_data["definition"]["type"],
                "period_start": report["period"]["start"],
                "period_end": report["period"]["end"],
                **metric_data["statistics"],
                **metric_data["trends"]
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Convert to CSV string
        return df.to_csv(index=False)