#!/usr/bin/env python3
"""
Binary Classifier for Repository Activity

Wraps sklearn classifiers for active/inactive prediction.
"""

import logging
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityClassifier:
    """Binary classifier for repository activity."""
    
    def __init__(
        self,
        model_type: str = 'logistic',
        random_seed: int = 42,
        **model_kwargs
    ):
        """
        Args:
            model_type: Type of classifier ('logistic', 'rf', 'gbm')
            random_seed: Random seed
            **model_kwargs: Additional arguments for the classifier
        """
        self.model_type = model_type
        self.random_seed = random_seed
        self.model_kwargs = model_kwargs
        
        self.scaler = StandardScaler()
        self.model = self._create_model()
        self.is_fitted = False
    
    def _create_model(self):
        """Create the underlying sklearn model."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000,
                **self.model_kwargs
            )
        elif self.model_type == 'rf':
            return RandomForestClassifier(
                random_state=self.random_seed,
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                **{k: v for k, v in self.model_kwargs.items() if k != 'n_estimators'}
            )
        elif self.model_type == 'gbm':
            return GradientBoostingClassifier(
                random_state=self.random_seed,
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                **{k: v for k, v in self.model_kwargs.items() if k != 'n_estimators'}
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        logger.info(f"Training {self.model_type} classifier...")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        logger.info("Training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier performance."""
        predictions = self.predict(X)
        probas = self.predict_proba(X)[:, 1]
        
        metrics = {
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probas) if len(np.unique(y)) > 1 else 0.0,
            'pr_auc': average_precision_score(y, probas) if len(np.unique(y)) > 1 else 0.0,
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to: {path}")
    
    @staticmethod
    def load(path: str) -> 'ActivityClassifier':
        """Load model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from: {path}")
        return model
