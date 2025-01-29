# src/utils/analytics/analytics_manager.py
# Created: 2025-01-29 20:25:41
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
import aiofiles

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
    min_data_points: int = 30
    enable_trend_analysis: bool = True
    data_retention_days: int = 90

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
        if self.config.enable_auto_cleanup:
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
        try:
            self._metrics[metric.name].append(metric)
            
            if len(self._metrics[metric.name]) >= self.config.batch_size:
                await self._persist_metrics(metric.name)
        except Exception as e:
            logger.error(f"Failed to record metric: {str(e)}")
            raise DataProcessingError(f"Failed to record metric: {str(e)}")

    async def _persist_metrics(self, metric_name: str) -> None:
        """Persist metrics to storage"""
        metrics = self._metrics[metric_name]
        if not metrics:
            return
            
        file_path = self.config.storage_path / f"{metric_name}_{datetime.utcnow().strftime('%Y%m')}.json"
        
        try:
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
        
        try:
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
                "mean": float(df['value'].mean()),
                "median": float(df['value'].median()),
                "std": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
                "quartiles": {
                    str(q): float(v) 
                    for q, v in df['value'].quantile([0.25, 0.5, 0.75]).items()
                }
            }
            
            # Update cache
            if self.config.enable_caching:
                self._cache[cache_key] = (stats_data, datetime.utcnow())
                
            return stats_data
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {str(e)}")
            raise DataProcessingError(f"Failed to calculate statistics: {str(e)}")

    async def _load_metric_data(
        self,
        metric_name: str,
        timeframe: TimeFrame
    ) -> List[Dict[str, Any]]:
        """Load metric data from storage"""
        try:
            now = datetime.utcnow()
            if timeframe == TimeFrame.HOUR:
                start_time = now - timedelta(hours=1)
            elif timeframe == TimeFrame.DAY:
                start_time = now - timedelta(days=1)
            elif timeframe == TimeFrame.WEEK:
                start_time = now - timedelta(weeks=1)
            elif timeframe == TimeFrame.MONTH:
                start_time = now - timedelta(days=30)
            elif timeframe == TimeFrame.QUARTER:
                start_time = now - timedelta(days=90)
            else:  # YEAR
                start_time = now - timedelta(days=365)

            # Get relevant files
            pattern = f"{metric_name}_*.json"
            data = []
            
            for file_path in self.config.storage_path.glob(pattern):
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                        for line in content.splitlines():
                            item = json.loads(line)
                            timestamp = datetime.fromisoformat(item['timestamp'])
                            if timestamp >= start_time:
                                data.append(item)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    continue

            return sorted(data, key=lambda x: x['timestamp'])

        except Exception as e:
            logger.error(f"Error loading metric data: {str(e)}")
            raise DataProcessingError(f"Failed to load metric data: {str(e)}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data periodically"""
        while True:
            try:
                logger.info("Starting data cleanup...")
                cutoff = datetime.utcnow() - timedelta(days=self.config.max_data_age)
                
                # Cleanup metric cache
                for metric_name in list(self._metrics.keys()):
                    self._metrics[metric_name] = [
                        m for m in self._metrics[metric_name]
                        if m.timestamp >= cutoff
                    ]
                
                # Cleanup old files
                for pattern in ["*.json", "plots/*.png", "reports/*.json"]:
                    for file_path in self.config.storage_path.glob(pattern):
                        try:
                            # Extract timestamp from filename
                            timestamp_str = file_path.stem.split('_')[-1]
                            file_time = datetime.strptime(
                                timestamp_str,
                                "%Y%m%d" if len(timestamp_str) == 8 else "%Y%m%d_%H%M%S"
                            )
                            
                            if file_time < cutoff:
                                file_path.unlink()
                                logger.info(f"Deleted old file: {file_path}")
                                
                        except (ValueError, OSError) as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")
                            continue
                
                # Cleanup cache
                if self.config.enable_caching:
                    current_time = datetime.utcnow()
                    expired_keys = [
                        key for key, (_, timestamp) in self._cache.items()
                        if (current_time - timestamp).total_seconds() > self.config.cache_ttl
                    ]
                    for key in expired_keys:
                        self._cache.pop(key, None)
                
                logger.info("Data cleanup completed")
                
            except Exception as e:
                logger.error(f"Error during data cleanup: {str(e)}")
                
            await asyncio.sleep(self.config.cleanup_interval)

    def _validate_metric(self, metric: MetricData) -> bool:
        """
        Validate a metric before recording
        
        Args:
            metric: Metric to validate
            
        Returns:
            bool: True if metric is valid, False otherwise
        """
        try:
            if not isinstance(metric.value, (int, float)):
                logger.error(f"Invalid metric value type: {type(metric.value)}")
                return False
                
            if not isinstance(metric.timestamp, datetime):
                logger.error(f"Invalid timestamp type: {type(metric.timestamp)}")
                return False
                
            if not isinstance(metric.labels, dict):
                logger.error(f"Invalid labels type: {type(metric.labels)}")
                return False
                
            if not metric.name or not isinstance(metric.name, str):
                logger.error("Invalid or missing metric name")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating metric: {str(e)}")
            return False

    async def export_data(
        self,
        metric_names: List[str],
        timeframe: TimeFrame,
        format: str = "json"
    ) -> Path:
        """
        Export metric data to a file
        
        Args:
            metric_names: List of metrics to export
            timeframe: Time frame to export
            format: Export format (json or csv)
            
        Returns:
            Path to the exported file
        """
        try:
            all_data = []
            for metric_name in metric_names:
                data = await self._load_metric_data(metric_name, timeframe)
                all_data.extend(data)
                
            if not all_data:
                raise DataProcessingError("No data to export")
                
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                export_path = self.config.storage_path / f"export_{timestamp}.json"
                async with aiofiles.open(export_path, 'w') as f:
                    await f.write(json.dumps(all_data, indent=2))
                    
            elif format.lower() == "csv":
                export_path = self.config.storage_path / f"export_{timestamp}.csv"
                df = pd.DataFrame(all_data)
                df.to_csv(export_path, index=False)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise DataProcessingError(f"Failed to export data: {str(e)}")
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
            plot_type: Type of plot (line, scatter, histogram, box)
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
            elif plot_type == "box":
                sns.boxplot(data=df, y='value', **kwargs)
            else:
                raise VisualizationError(f"Unsupported plot type: {plot_type}")
                
            plt.title(f"{metric_name} - {timeframe.value}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            plot_path = self.config.storage_path / "plots" / f"{metric_name}_{timeframe.value}_{timestamp}.png"
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
        Detect anomalies in metric data
        
        Args:
            metric_name: Name of the metric
            timeframe: Time frame to analyze
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        try:
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
                        "value": float(df['value'].iloc[idx]),
                        "z_score": float(z_score),
                        "labels": data[idx]['labels']
                    })
                    
            return anomalies
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {str(e)}")
            raise AnalyticsError(f"Failed to detect anomalies: {str(e)}")

    async def generate_report(
        self,
        metric_names: List[str],
        timeframe: TimeFrame,
        include_plots: bool = True
    ) -> Path:
        """
        Generate comprehensive analytics report
        
        Args:
            metric_names: List of metrics to include
            timeframe: Time frame to analyze
            include_plots: Whether to include visualizations
            
        Returns:
            Path to the generated report file
        """
        try:
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
                
                # Create visualization if requested
                plot_path = None
                if include_plots:
                    try:
                        plot_path = await self.create_visualization(metric_name, timeframe)
                    except Exception as e:
                        logger.error(f"Failed to create visualization for {metric_name}: {str(e)}")
                
                report_data["metrics"][metric_name] = {
                    "statistics": stats,
                    "anomalies": anomalies,
                    "plot_path": str(plot_path) if plot_path else None
                }
            
            # Save report
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_path = self.config.storage_path / "reports" / f"report_{timestamp}.json"
            
            async with aiofiles.open(report_path, 'w') as f:
                await f.write(json.dumps(report_data, indent=2))
            
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            raise AnalyticsError(f"Failed to generate report: {str(e)}")

    async def get_trend_analysis(
        self,
        metric_name: str,
        timeframe: TimeFrame
    ) -> Dict[str, Any]:
        """
        Analyze trend for a metric
        
        Args:
            metric_name: Name of the metric
            timeframe: Time frame to analyze
            
        Returns:
            Dict containing trend analysis results
        """
        try:
            data = await self._load_metric_data(metric_name, timeframe)
            if not data or len(data) < self.config.min_data_points:
                return {}
                
            df = pd.DataFrame(data)
            values = df['value'].values
            timestamps = np.arange(len(values))
            
            # Calculate trend line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                timestamps,
                values
            )
            
            return {
                "slope": float(slope),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_error": float(std_err),
                "direction": "increasing" if slope > 0 else "decreasing",
                "significance": p_value < 0.05
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trend: {str(e)}")
            raise AnalyticsError(f"Failed to analyze trend: {str(e)}")