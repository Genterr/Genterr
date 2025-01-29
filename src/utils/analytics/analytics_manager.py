# src/utils/analytics/analytics_manager.py
# Created: 2025-01-29 20:06:40
# Author: Genterr

from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logger = logging.getLogger(__name__)

class AnalyticsError(Exception):
    """Base exception for analytics-related errors"""
    pass

class DataProcessingError(AnalyticsError):
    """Raised when data processing fails"""
    pass

class VisualizationError(AnalyticsError):
    """Raised when visualization creation fails"""
    pass

class AnalyticsMetricType(Enum):
    """Types of analytics metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class TimeFrame(Enum):
    """Time frames for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

@dataclass
class AnalyticsConfig:
    """Configuration for analytics settings"""
    storage_path: Path = Path("analytics_data")
    max_data_age: int = 365  # days
    batch_size: int = 1000
    enable_auto_cleanup: bool = True
    cleanup_interval: int = 86400  # seconds
    visualization_dpi: int = 300
    default_plot_style: str = "seaborn"
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds

@dataclass
class MetricData:
    """Container for metric data"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    type: AnalyticsMetricType
    labels: Dict[str, str]
    source: str

class PlatformAnalytics:
    """
    Manages platform analytics and metrics processing.
    
    This class handles:
    - Data collection and processing
    - Statistical analysis
    - Trend detection
    - Visualization generation
    - Report creation
    - Data storage and cleanup
    """

    def __init__(self, config: AnalyticsConfig):
        """Initialize PlatformAnalytics with configuration"""
        self.config = config
        self._metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Setup logging and storage
        self._setup_logging()
        self._setup_storage()
        
        # Set plot style
        plt.style.use(self.config.default_plot_style)
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_data())

    def _setup_logging(self) -> None:
        """Configure analytics-related logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('analytics.log'),
                logging.StreamHandler()
            ]
        )

    def _setup_storage(self) -> None:
        """Initialize storage for analytics data"""
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        (self.config.storage_path / "plots").mkdir(exist_ok=True)
        (self.config.storage_path / "reports").mkdir(exist_ok=True)

    async def record_metric(self, metric: MetricData) -> None:
        """
        Record a new metric
        
        Args:
            metric: Metric data to record
        """
        self._metrics[metric.name].append(metric)
        
        if len(self._metrics[metric.name]) >= self.config.batch_size:
            await self._persist_metrics(metric.name)

    async def _persist_metrics(self, metric_name: str) -> None:
        """Persist metrics to storage"""
        metrics = self._metrics[metric_name]
        if not metrics:
            return
            
        file_path = self.config.storage_path / f"{metric_name}_{datetime.utcnow().strftime('%Y%m')}.json"
        
        data = [
            {
                "name": m.name,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "type": m.type.value,
                "labels": m.labels,
                "source": m.source
            }
            for m in metrics
        ]
        
        try:
            async with aiofiles.open(file_path, 'a') as f:
                for item in data:
                    await f.write(json.dumps(item) + "\n")
            
            self._metrics[metric_name].clear()
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {str(e)}")
            raise DataProcessingError(f"Failed to persist metrics: {str(e)}")

    async def get_metric_stats(
        self,
        metric_name: str,
        timeframe: TimeFrame,
        aggregation: str = "mean"
    ) -> Dict[str, Any]:
        """
        Get statistical information about a metric
        
        Args:
            metric_name: Name of the metric
            timeframe: Time frame to analyze
            aggregation: Aggregation method (mean, median, sum, etc.)
            
        Returns:
            Dict containing statistical information
        """
        cache_key = f"{metric_name}_{timeframe.value}_{aggregation}"
        
        # Check cache
        if self.config.enable_caching:
            cached = self._cache.get(cache_key)
            if cached and (datetime.utcnow() - cached[1]).total_seconds() < self.config.cache_ttl:
                return cached[0]
                
        # Load metric data
        data = await self._load_metric_data(metric_name, timeframe)
        if not data:
            return {}
            
        # Calculate statistics
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        stats_data = {
            "count": len(df),
            "mean": df['value'].mean(),
            "median": df['value'].median(),
            "std": df['value'].std(),
            "min": df['value'].min(),
            "max": df['value'].max(),
            "quartiles": df['value'].quantile([0.25, 0.5, 0.75]).to_dict(),
        }
        
        # Update cache
        if self.config.enable_caching:
            self._cache[cache_key] = (stats_data, datetime.utcnow())
            
        return stats_data

    async def create_visualization(
        self,
        metric_name: str,
        timeframe: TimeFrame,
        plot_type: str = "line",
        **kwargs
    ) -> Path:
        """
        Create visualization for metric data
        
        Args:
            metric_name: Name of the metric
            timeframe: Time frame to visualize
            plot_type: Type of plot to create
            **kwargs: Additional plotting parameters
            
        Returns:
            Path to the created visualization file
        """
        try:
            data = await self._load_metric_data(metric_name, timeframe)
            if not data:
                raise VisualizationError("No data available for visualization")
                
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            
            if plot_type == "line":
                sns.lineplot(data=df, x='timestamp', y='value', **kwargs)
            elif plot_type == "scatter":
                sns.scatterplot(data=df, x='timestamp', y='value', **kwargs)
            elif plot_type == "histogram":
                sns.histplot(data=df, x='value', **kwargs)
            else:
                raise VisualizationError(f"Unsupported plot type: {plot_type}")
                
            plt.title(f"{metric_name} - {timeframe.value}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_path = self.config.storage_path / "plots" / f"{metric_name}_{timeframe.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=self.config.visualization_dpi)
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            raise VisualizationError(f"Failed to create visualization: {str(e)}")

    async def detect_anomalies(
        self,
        metric_name: str,
        timeframe: TimeFrame,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metric data using statistical methods
        
        Args:
            metric_name: Name of the metric
            timeframe: Time frame to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        data = await self._load_metric_data(metric_name, timeframe)
        if not data:
            return []
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df['value']))
        
        # Identify anomalies
        anomalies = []
        for idx, z_score in enumerate(z_scores):
            if z_score > threshold:
                anomalies.append({
                    "timestamp": df['timestamp'].iloc[idx].isoformat(),
                    "value": df['value'].iloc[idx],
                    "z_score": z_score,
                    "labels": data[idx]['labels']
                })
                
        return anomalies

    async def generate_report(
        self,
        metric_names: List[str],
        timeframe: TimeFrame,
        include_plots: bool = True
    ) -> Path:
        """
        Generate a comprehensive analytics report
        
        Args:
            metric_names: List of metrics to include
            timeframe: Time frame to analyze
            include_plots: Whether to include visualizations
            
        Returns:
            Path to the generated report
        """
        report_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "timeframe": timeframe.value,
            "metrics": {}
        }
        
        for metric_name in metric_names:
            # Get statistics
            stats = await self.get_metric_stats(metric_name, timeframe)
            
            # Get anomalies
            anomalies = await self.detect_anomalies(metric_name, timeframe)
            
            report_data["metrics"][metric_name] = {
                "statistics": stats,
                "anomalies": anomalies,
                "trend": await self._analyze_trend(metric_name, timeframe)
            }
            
        return report_data

    async def _analyze_trend(
        self,
        metric_name: str,
        timeframe: TimeFrame
    ) -> Dict[str, Any]:
        """Analyze trend for a metric"""
        if not self.config.enable_trend_analysis:
            return {}
            
        try:
            data = await self._get_metric_data(metric_name, timeframe)
            if not data or len(data) < self.config.min_data_points:
                return {}
                
            values = [d["value"] for d in data]
            timestamps = range(len(values))
            
            # Calculate trend line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps,
                values
            )
            
            return {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "direction": "up" if slope > 0 else "down"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {}

    async def _get_metric_data(
        self,
        metric_name: str,
        timeframe: TimeFrame
    ) -> List[Dict[str, Any]]:
        """Get metric data for specified timeframe"""
        try:
            now = datetime.utcnow()
            
            if timeframe == TimeFrame.HOUR:
                cutoff = now - timedelta(hours=1)
            elif timeframe == TimeFrame.DAY:
                cutoff = now - timedelta(days=1)
            elif timeframe == TimeFrame.WEEK:
                cutoff = now - timedelta(weeks=1)
            elif timeframe == TimeFrame.MONTH:
                cutoff = now - timedelta(days=30)
            else:  # YEAR
                cutoff = now - timedelta(days=365)
                
            return [
                data for data in self._metric_cache.get(metric_name, [])
                if data["timestamp"] >= cutoff
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving metric data: {str(e)}")
            return []

    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data periodically"""
        while True:
            try:
                cutoff = datetime.utcnow() - timedelta(days=self.config.data_retention_days)
                
                for metric_name in self._metric_cache:
                    self._metric_cache[metric_name] = [
                        data for data in self._metric_cache[metric_name]
                        if data["timestamp"] >= cutoff
                    ]
                    
            except Exception as e:
                logger.error(f"Error cleaning up old data: {str(e)}")
                
            await asyncio.sleep(3600)  # Clean up every hour