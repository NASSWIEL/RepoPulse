"""
Preprocessing module for GitHub repository activity data.

This module provides tools for:
- Data inspection and schema analysis
- Quarterly aggregation of time series data
- Activity labeling with threshold optimization
"""

__version__ = "1.0.0"

from pathlib import Path

# Module metadata
__author__ = "GitHub Forecasting Team"
__all__ = ['inspect_data', 'aggregate_quarters', 'label_activity']
