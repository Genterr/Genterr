# src/utils/analytics/__init__.py
# Created: 2025-01-29 20:40:38
# Author: Genterr

"""
Analytics utility package for data collection, analysis, and reporting.
"""

from .analytics_manager import (
    PlatformAnalytics,
    AnalyticsConfig,
    AnalyticsMetricType,
    TimeFrame,
    MetricData,
    AnalyticsError,
    DataProcessingError,
    VisualizationError
)

__all__ = [
    'PlatformAnalytics',
    'AnalyticsConfig',
    'AnalyticsMetricType',
    'TimeFrame',
    'MetricData',
    'AnalyticsError',
    'DataProcessingError',
    'VisualizationError'
]