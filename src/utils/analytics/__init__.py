# src/utils/analytics/__init__.py
# Created: 2025-01-29 20:01:12
# Author: Genterr

"""
Analytics utility package for data collection, analysis, and reporting.
"""

from .analytics_manager import (
    AnalyticsManager,
    AnalyticsConfig,
    MetricType,
    TimeFrame,
    AggregationType,
    AnalyticsError,
    DataCollectionError,
    AnalysisError
)

__all__ = [
    'AnalyticsManager',
    'AnalyticsConfig',
    'MetricType',
    'TimeFrame',
    'AggregationType',
    'AnalyticsError',
    'DataCollectionError',
    'AnalysisError'
]