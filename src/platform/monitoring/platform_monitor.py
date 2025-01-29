# src/platform/monitoring/platform_monitor.py
# Created: 2025-01-29 19:27:38
# Author: Genterr

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
from pathlib import Path
import json
import psutil
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MonitoringError(Exception):
    """Base exception for monitoring-related errors"""
    pass

class MetricCollectionError(MonitoringError):
    """Raised when metric collection fails"""
    pass

class AlertingError(MonitoringError):
    """Raised when alert processing fails"""
    pass

class MetricType(Enum):
    """Types of metrics that can be collected"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    APPLICATION = "application"
    CUSTOM = "custom"

class AlertSeverity(Enum):
    """Severity levels for monitoring alerts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class MetricConfig:
    """Configuration for metric collection"""
    collection_interval: int = 60  # seconds
    retention_period: int = 7      # days
    storage_path: Path = Path("metrics")
    enable_disk_logging: bool = True
    enable_remote_export: bool = False
    max_batch_size: int = 1000
    compression_enabled: bool = True

@dataclass
class AlertConfig:
    """Configuration for alert processing"""
    notification_endpoint: Optional[str] = None
    alert_history_path: Path = Path("alerts")
    max_alerts_per_hour: int = 100
    cooldown_period: int = 300     # seconds
    enable_aggregation: bool = True
    default_severity: AlertSeverity = AlertSeverity.MEDIUM

@dataclass
class MetricData:
    """Container for collected metric data"""
    metric_type: MetricType
    timestamp: datetime
    value: Union[float, int, str, Dict]
    labels: Dict[str, str]
    source: str
    unit: Optional[str] = None

class PlatformMonitor:
    """
    Manages platform monitoring and metric collection.
    
    This class handles:
    - System metric collection (CPU, Memory, Disk, Network)
    - Custom metric collection
    - Alert generation and processing
    - Metric storage and retention
    - Remote metric export
    - Alert notification
    """

    def __init__(
        self,
        metric_config: MetricConfig,
        alert_config: AlertConfig
    ):
        """Initialize PlatformMonitor with configurations"""
        self.metric_config = metric_config
        self.alert_config = alert_config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._session = None
        self._metric_buffer: List[MetricData] = []
        self._alert_history: Dict[str, List[datetime]] = {}
        self._running = False
        
        # Setup logging and storage
        self._setup_logging()
        self._setup_storage()

    def _setup_logging(self) -> None:
        """Configure logging for monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.metric_config.storage_path / 'monitor.log',
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )

    def _setup_storage(self) -> None:
        """Initialize storage directories"""
        self.metric_config.storage_path.mkdir(parents=True, exist_ok=True)
        self.alert_config.alert_history_path.mkdir(parents=True, exist_ok=True)

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is created"""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def start(self) -> None:
        """Start monitoring"""
        if self._running:
            logger.warning("Monitoring is already running")
            return
            
        self._running = True
        logger.info("Starting platform monitoring")
        
        try:
            await asyncio.gather(
                self._collect_system_metrics(),
                self._process_metric_buffer(),
                self._cleanup_old_metrics()
            )
        except Exception as e:
            logger.error(f"Error in monitoring: {str(e)}")
            self._running = False
            raise MonitoringError(f"Monitoring failed: {str(e)}")

    async def stop(self) -> None:
        """Stop monitoring"""
        self._running = False
        if self._session:
            await self._session.close()
        self._executor.shutdown(wait=True)
        logger.info("Stopped platform monitoring")

    async def _collect_system_metrics(self) -> None:
        """Collect system metrics periodically"""
        while self._running:
            try:
                # Collect CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self._add_metric(MetricData(
                    metric_type=MetricType.CPU,
                    timestamp=datetime.utcnow(),
                    value=cpu_percent,
                    labels={"type": "utilization"},
                    source="system",
                    unit="percent"
                ))
                
                # Collect memory metrics
                memory = psutil.virtual_memory()
                self._add_metric(MetricData(
                    metric_type=MetricType.MEMORY,
                    timestamp=datetime.utcnow(),
                    value=memory.percent,
                    labels={"type": "utilization"},
                    source="system",
                    unit="percent"
                ))
                
                # Collect disk metrics
                disk = psutil.disk_usage('/')
                self._add_metric(MetricData(
                    metric_type=MetricType.DISK,
                    timestamp=datetime.utcnow(),
                    value=disk.percent,
                    labels={"type": "utilization"},
                    source="system",
                    unit="percent"
                ))
                
                # Collect network metrics
                network = psutil.net_io_counters()
                self._add_metric(MetricData(
                    metric_type=MetricType.NETWORK,
                    timestamp=datetime.utcnow(),
                    value={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    },
                    labels={"type": "throughput"},
                    source="system",
                    unit="bytes"
                ))
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
                await self._process_alert(
                    "System metric collection failed",
                    AlertSeverity.HIGH,
                    {"error": str(e)}
                )
                
            await asyncio.sleep(self.metric_config.collection_interval)

    def _add_metric(self, metric: MetricData) -> None:
        """Add metric to buffer"""
        self._metric_buffer.append(metric)
        if len(self._metric_buffer) >= self.metric_config.max_batch_size:
            asyncio.create_task(self._process_metric_buffer())

    async def _process_metric_buffer(self) -> None:
        """Process and store metrics from buffer"""
        while self._running:
            if not self._metric_buffer:
                await asyncio.sleep(1)
                continue
                
            metrics = self._metric_buffer.copy()
            self._metric_buffer.clear()
            
            try:
                # Store metrics to disk if enabled
                if self.metric_config.enable_disk_logging:
                    await self._store_metrics(metrics)
                
                # Export metrics if enabled
                if self.metric_config.enable_remote_export:
                    await self._export_metrics(metrics)
                    
            except Exception as e:
                logger.error(f"Error processing metrics: {str(e)}")
                await self._process_alert(
                    "Metric processing failed",
                    AlertSeverity.MEDIUM,
                    {"error": str(e)}
                )
                
            await asyncio.sleep(1)

    async def _store_metrics(self, metrics: List[MetricData]) -> None:
        """Store metrics to disk"""
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filepath = self.metric_config.storage_path / f"metrics-{timestamp}.json"
        
        metric_data = [
            {
                "type": m.metric_type.value,
                "timestamp": m.timestamp.isoformat(),
                "value": m.value,
                "labels": m.labels,
                "source": m.source,
                "unit": m.unit
            }
            for m in metrics
        ]
        
        try:
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(metric_data))
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
            raise MetricCollectionError(f"Metric storage failed: {str(e)}")

    async def _export_metrics(self, metrics: List[MetricData]) -> None:
        """Export metrics to remote endpoint if configured"""
        if not self.metric_config.enable_remote_export:
            return
            
        await self._ensure_session()
        # Implementation for remote metric export would go here
        pass

    async def _process_alert(
        self,
        message: str,
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Process and send monitoring alerts"""
        alert_key = f"{severity.value}:{message}"
        
        # Check alert cooldown
        now = datetime.utcnow()
        if alert_key in self._alert_history:
            last_alert = self._alert_history[alert_key][-1]
            if (now - last_alert).total_seconds() < self.alert_config.cooldown_period:
                return
                
        # Update alert history
        if alert_key not in self._alert_history:
            self._alert_history[alert_key] = []
        self._alert_history[alert_key].append(now)
        
        # Trim old alerts
        self._alert_history[alert_key] = [
            ts for ts in self._alert_history[alert_key]
            if (now - ts).total_seconds() < 3600
        ]
        
        # Check rate limiting
        if len(self._alert_history[alert_key]) > self.alert_config.max_alerts_per_hour:
            logger.warning(f"Alert rate limit exceeded for: {alert_key}")
            return
            
        # Send alert
        if self.alert_config.notification_endpoint:
            await self._send_alert(message, severity, context)

    async def _send_alert(
        self,
        message: str,
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Send alert to configured notification endpoint"""
        if not self.alert_config.notification_endpoint:
            return
            
        await self._ensure_session()
        
        try:
            async with self._session.post(
                self.alert_config.notification_endpoint,
                json={
                    "message": message,
                    "severity": severity.value,
                    "context": context,
                    "timestamp": datetime.utcnow().isoformat()
                }
            ) as response:
                if response.status >= 400:
                    raise AlertingError(
                        f"Alert sending failed with status {response.status}"
                    )
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
            raise AlertingError(f"Alert sending failed: {str(e)}")

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metric files"""
        while self._running:
            try:
                cutoff = datetime.utcnow() - timedelta(
                    days=self.metric_config.retention_period
                )
                
                for file in self.metric_config.storage_path.glob("metrics-*.json"):
                    try:
                        # Extract timestamp from filename
                        timestamp_str = file.stem.split('-', 1)[1]
                        file_time = datetime.strptime(
                            timestamp_str,
                            "%Y%m%d-%H%M%S"
                        )
                        
                        if file_time < cutoff:
                            file.unlink()
                            logger.info(f"Cleaned up old metric file: {file.name}")
                    except (ValueError, OSError) as e:
                        logger.error(f"Error processing file {file}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in metric cleanup: {str(e)}")
                
            await asyncio.sleep(3600)  # Check once per hour