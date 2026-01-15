#!/usr/bin/env python3
"""
FastAPI Inference Service for Repository Activity Prediction

This service loads the best-performing model checkpoint and provides
an interactive API for predicting repository activity status from time-series data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Model Architecture (must match training)
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model architecture matching the trained checkpoint."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


class GRUModel(nn.Module):
    """GRU model architecture matching the trained checkpoint."""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Use last timestep output
        predictions = self.fc(gru_out[:, -1, :])
        return predictions


# ============================================================================
# Activity Scoring & Classification
# ============================================================================

def compute_activity_score(metrics: np.ndarray) -> float:
    """
    Compute activity score from repository metrics.
    
    Args:
        metrics: Array of shape (8,) with features:
            [commit_count, contributor_count, issue_count, pr_count,
             star_count, watch_count, release_count, fork_count]
    
    Returns:
        Activity score (higher = more active)
    """
    weights = np.array([1.0, 2.0, 0.5, 0.8, 0.2, 0.3, 1.5, 0.3])
    return float(np.dot(metrics, weights))


def classify_activity(score: float, threshold: float = 1319.5) -> str:
    """
    Classify repository as active or inactive based on score.
    
    Args:
        score: Activity score
        threshold: Classification threshold (75th percentile from training)
    
    Returns:
        'active' or 'inactive'
    """
    return 'active' if score >= threshold else 'inactive'


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_stats = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature names
        self.feature_names = [
            'commit_count',
            'contributor_count',
            'issue_count',
            'pr_count',
            'star_count',
            'watch_count',
            'release_count',
            'fork_count'
        ]
    
    def load_feature_stats(self, stats_path: str = 'data/processed/timeseries/feature_stats.json'):
        """Load normalization statistics from training."""
        path = Path(stats_path)
        if not path.exists():
            logger.warning(f"Feature stats not found at {stats_path}")
            return
        
        with open(path, 'r') as f:
            self.feature_stats = json.load(f)
        
        logger.info(f"Loaded feature statistics from {stats_path}")
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using training statistics."""
        if self.feature_stats is None:
            logger.warning("Feature stats not loaded, skipping normalization")
            return data
        
        mean = np.array(self.feature_stats['mean'], dtype=np.float32)
        std = np.array(self.feature_stats['std'], dtype=np.float32)
        std[std == 0] = 1.0  # Avoid division by zero
        
        return (data - mean) / std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization."""
        if self.feature_stats is None:
            return data
        
        mean = np.array(self.feature_stats['mean'], dtype=np.float32)
        std = np.array(self.feature_stats['std'], dtype=np.float32)
        std[std == 0] = 1.0
        
        return data * std + mean
    
    def load_best_model(self, model_type: str = 'gru'):
        """
        Load the best model checkpoint.
        
        Args:
            model_type: 'lstm' or 'gru'
        """
        checkpoint_path = Path(f'models/checkpoints/{model_type}_best.pt')
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading {model_type.upper()} model from {checkpoint_path}")
        
        # Create model architecture
        if model_type == 'lstm':
            self.model = LSTMModel(input_size=8, hidden_size=64, num_layers=2, dropout=0.2)
        elif model_type == 'gru':
            self.model = GRUModel(input_size=8, hidden_size=64, num_layers=2, dropout=0.2)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.model_type = model_type
        
        logger.info(f"✓ {model_type.upper()} model loaded successfully")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction from time-series sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, n_features)
                     Expected: (4, 8) - 4 quarters, 8 metrics
        
        Returns:
            Dictionary with predictions and activity classification
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Validate shape
        if sequence.shape != (4, 8):
            raise ValueError(f"Expected shape (4, 8), got {sequence.shape}")
        
        # Normalize input
        sequence_normalized = self.normalize(sequence)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions_normalized = self.model(sequence_tensor)
            predictions = predictions_normalized.squeeze(0).cpu().numpy()
        
        # Denormalize predictions
        predictions_denormalized = self.denormalize(predictions)
        
        # Clip to non-negative values
        predictions_denormalized = np.clip(predictions_denormalized, 0, None)
        
        # Compute activity score and classification
        activity_score = compute_activity_score(predictions_denormalized)
        activity_status = classify_activity(activity_score)
        
        # Build result
        result = {
            'predicted_metrics': {
                name: float(value) 
                for name, value in zip(self.feature_names, predictions_denormalized)
            },
            'activity_score': activity_score,
            'activity_status': activity_status,
            'confidence': {
                'score_threshold': 1319.5,
                'score_distance': abs(activity_score - 1319.5),
                'classification': 'high' if abs(activity_score - 1319.5) > 500 else 'medium'
            },
            'model_info': {
                'model_type': self.model_type,
                'device': str(self.device)
            }
        }
        
        return result


# Global model manager
model_manager = ModelManager()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Repository Activity Prediction API",
    description="""
    **Predict repository activity status from time-series data**
    
    This API uses trained LSTM/GRU models to forecast repository metrics for the next quarter
    and classify the repository as **active** or **inactive**.
    
    ### Input Format
    Time-series sequence of **4 quarters** × **8 metrics**:
    - `commit_count`: Number of commits
    - `contributor_count`: Number of unique contributors
    - `issue_count`: Number of issues opened
    - `pr_count`: Number of pull requests
    - `star_count`: Repository stars received
    - `watch_count`: Repository watchers
    - `release_count`: Number of releases
    - `fork_count`: Number of forks
    
    ### Output
    - Predicted metrics for next quarter
    - Activity score (weighted combination of metrics)
    - Activity status: **active** (score ≥ 1319.5) or **inactive** (score < 1319.5)
    - Confidence indicators
    
    ### Example
    Try the interactive Swagger UI below to submit sample data and view predictions!
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# Request/Response Schemas
# ============================================================================

class PredictionRequest(BaseModel):
    """Request schema for activity prediction."""
    
    sequence: List[List[float]] = Field(
        ...,
        description="Time-series sequence of shape (4, 8): 4 quarters × 8 metrics",
        example=[
            [120.0, 55.0, 80.0, 65.0, 85.0, 42.0, 5.0, 0.0],
            [140.0, 60.0, 90.0, 70.0, 95.0, 45.0, 6.0, 0.0],
            [130.0, 58.0, 85.0, 68.0, 90.0, 44.0, 5.0, 0.0],
            [150.0, 62.0, 95.0, 75.0, 100.0, 48.0, 7.0, 0.0]
        ]
    )
    
    @field_validator('sequence')
    @classmethod
    def validate_sequence_shape(cls, v):
        """Validate sequence dimensions."""
        if len(v) != 4:
            raise ValueError(f"Expected 4 quarters, got {len(v)}")
        
        for i, quarter in enumerate(v):
            if len(quarter) != 8:
                raise ValueError(f"Quarter {i} has {len(quarter)} metrics, expected 8")
        
        return v


class PredictionResponse(BaseModel):
    """Response schema for activity prediction."""
    
    predicted_metrics: Dict[str, float] = Field(
        ...,
        description="Predicted values for next quarter (8 metrics)"
    )
    activity_score: float = Field(
        ...,
        description="Computed activity score (weighted sum of metrics)"
    )
    activity_status: str = Field(
        ...,
        description="Classification: 'active' or 'inactive'"
    )
    confidence: Dict[str, Any] = Field(
        ...,
        description="Confidence indicators for the prediction"
    )
    model_info: Dict[str, str] = Field(
        ...,
        description="Information about the model used"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_type: str
    device: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("=" * 60)
    logger.info("Starting Repository Activity Prediction API")
    logger.info("=" * 60)
    
    try:
        # Load feature statistics
        model_manager.load_feature_stats()
        
        # Load best model (GRU performed best based on evaluation)
        model_manager.load_best_model(model_type='gru')
        
        logger.info("✓ API ready for inference")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Repository Activity Prediction API",
        "version": "1.0.0",
        "description": "Predict repository activity status from time-series metrics",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check API health and model status.
    
    Returns service status and loaded model information.
    """
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "degraded",
        model_loaded=model_manager.model is not None,
        model_type=model_manager.model_type or "none",
        device=str(model_manager.device)
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_activity(request: PredictionRequest):
    """
    **Predict repository activity status from time-series data**
    
    Submit a sequence of 4 quarters with 8 metrics each. The model will:
    1. Forecast metrics for the next quarter
    2. Compute an activity score
    3. Classify the repository as **active** or **inactive**
    
    ### Input Metrics (in order)
    1. `commit_count` - Number of commits
    2. `contributor_count` - Number of unique contributors  
    3. `issue_count` - Number of issues opened
    4. `pr_count` - Number of pull requests
    5. `star_count` - Stars received
    6. `watch_count` - Watchers
    7. `release_count` - Number of releases
    8. `fork_count` - Number of forks
    
    ### Example Input
    ```json
    {
      "sequence": [
        [120, 55, 80, 65, 85, 42, 5, 0],  // Quarter 1
        [140, 60, 90, 70, 95, 45, 6, 0],  // Quarter 2
        [130, 58, 85, 68, 90, 44, 5, 0],  // Quarter 3
        [150, 62, 95, 75, 100, 48, 7, 0]  // Quarter 4 (most recent)
      ]
    }
    ```
    
    ### Returns
    - Predicted metrics for next quarter
    - Activity score and classification
    - Confidence indicators
    """
    try:
        # Convert to numpy array
        sequence = np.array(request.sequence, dtype=np.float32)
        
        # Make prediction
        result = model_manager.predict(sequence)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model type, architecture details, and feature names.
    """
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_manager.model_type,
        "architecture": {
            "type": "LSTM" if model_manager.model_type == "lstm" else "GRU",
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "parameters": sum(p.numel() for p in model_manager.model.parameters())
        },
        "input_shape": "(4, 8)",
        "output_shape": "(8,)",
        "feature_names": model_manager.feature_names,
        "device": str(model_manager.device),
        "activity_threshold": 1319.5
    }


# ============================================================================
# Run with: uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
