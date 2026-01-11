"""
Models module for repository activity forecasting and classification.

This module provides:
- Baseline forecasters (Naive, Moving Average)
- Neural forecasters (LSTM, GRU)
- Binary classifiers for activity prediction
- Training and evaluation utilities
"""

__version__ = "1.0.0"

from pathlib import Path

# Module metadata
__author__ = "GitHub Forecasting Team"
__all__ = ['forecaster', 'classifier', 'train_forecaster', 'train_classifier']
