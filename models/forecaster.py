#!/usr/bin/env python3
"""
Forecasting Models for GitHub Repository Activity

This module implements various autoregressive forecasting models:
- Baseline models (naive, seasonal, moving average)
- LSTM/GRU sequence models
- Temporal Transformer (optional)

All models predict numeric metrics for the next quarter given previous quarters.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series forecasting with variable-length support."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        sequence_lengths: Optional[np.ndarray] = None
    ):
        """
        Args:
            sequences: Historical sequences (n_samples, sequence_length, n_features)
            targets: Target values (n_samples, n_features)
            sequence_length: Maximum length of input sequences (for padding)
            sequence_lengths: Actual lengths of sequences before padding (optional)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
        if sequence_lengths is not None:
            self.sequence_lengths = torch.LongTensor(sequence_lengths)
        else:
            # All sequences have full length
            self.sequence_lengths = torch.LongTensor([sequence_length] * len(sequences))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx], self.sequence_lengths[idx]


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    def __init__(self, random_seed: int = 42):
        """
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.is_fitted = False
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        """
        Train the model.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target values (n_samples, n_features)
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions (n_samples, n_features)
        """
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input sequences
            y: True target values
            
        Returns:
            Dictionary of metric name to value
        """
        predictions = self.predict(X)
        
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(mse)
        
        # Per-feature metrics
        mse_per_feature = np.mean((predictions - y) ** 2, axis=0)
        mae_per_feature = np.mean(np.abs(predictions - y), axis=0)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mse_per_feature': mse_per_feature.tolist(),
            'mae_per_feature': mae_per_feature.tolist()
        }


class NaiveForecaster(BaseForecaster):
    """Naive baseline: predicts last observed value."""
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveForecaster':
        """Fit (does nothing for naive model)."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using last value in sequence."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Return last timestep of each sequence
        return X[:, -1, :]


class MovingAverageForecaster(BaseForecaster):
    """Moving average baseline."""
    
    def __init__(self, window_size: int = 3, random_seed: int = 42):
        """
        Args:
            window_size: Number of timesteps to average
            random_seed: Random seed
        """
        super().__init__(random_seed)
        self.window_size = window_size
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MovingAverageForecaster':
        """Fit (does nothing for moving average)."""
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using moving average of last window_size values."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Average over last window_size timesteps
        window = min(self.window_size, X.shape[1])
        return np.mean(X[:, -window:, :], axis=1)


class LSTMForecaster(BaseForecaster):
    """LSTM-based autoregressive forecaster."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            random_seed: Random seed
            device: Device to use ('cuda' or 'cpu')
        """
        super().__init__(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self._build_model()
    
    def _build_model(self):
        """Build LSTM model architecture."""
        self.model = nn.Sequential(
            nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True
            ),
        ).to(self.device)
        
        # Add output layer separately since LSTM returns tuple
        self.output_layer = nn.Linear(self.hidden_size, self.input_size).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sequence_lengths: Optional[np.ndarray] = None) -> 'LSTMForecaster':
        """
        Train the LSTM model.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            y: Target values (n_samples, n_features)
            sequence_lengths: Actual lengths of sequences (for padded input)
            
        Returns:
            Self
        """
        logger.info(f"Training LSTM on device: {self.device}")
        
        # Normalize data
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(X_scaled, y_scaled, seq_len, sequence_lengths)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.output_layer.parameters()),
            lr=self.learning_rate
        )
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch_X, batch_y, batch_lengths in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.cpu()
                
                # Pack padded sequence for efficient processing
                # Sort by length (descending) for pack_padded_sequence
                sorted_lengths, sort_idx = batch_lengths.sort(descending=True)
                batch_X_sorted = batch_X[sort_idx]
                batch_y_sorted = batch_y[sort_idx]
                
                # Pack sequences
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    batch_X_sorted, sorted_lengths.tolist(), batch_first=True
                )
                
                # Forward pass
                packed_output, (hidden, cell) = self.model[0](packed_input)
                
                # Use last hidden state (which corresponds to actual last timestep)
                # For LSTM with batch_first, hidden shape is [num_layers, batch, hidden_size]
                last_hidden = hidden[-1]  # Take last layer
                predictions = self.output_layer(last_hidden)
                
                # Unsort predictions to match original batch order
                unsort_idx = sort_idx.argsort()
                predictions = predictions[unsort_idx]
                
                # Compute loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        logger.info("LSTM training complete")
        return self
    
    def predict(self, X: np.ndarray, sequence_lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            sequence_lengths: Actual lengths of sequences (for padded input)
            
        Returns:
            Predictions (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Normalize input
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        
        # Default to full length if not provided
        if sequence_lengths is None:
            sequence_lengths = np.array([seq_len] * n_samples)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            lengths_tensor = torch.LongTensor(sequence_lengths)
            
            # Sort by length for pack_padded_sequence
            sorted_lengths, sort_idx = lengths_tensor.sort(descending=True)
            X_sorted = X_tensor[sort_idx]
            
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(
                X_sorted, sorted_lengths.tolist(), batch_first=True
            )
            
            # Forward pass
            _, (hidden, cell) = self.model[0](packed_input)
            predictions_scaled = self.output_layer(hidden[-1])
            
            # Unsort predictions
            unsort_idx = sort_idx.argsort()
            predictions_scaled = predictions_scaled[unsort_idx]
            predictions_scaled = predictions_scaled.cpu().numpy()
        
        # Denormalize
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions


class GRUForecaster(BaseForecaster):
    """GRU-based autoregressive forecaster (lighter than LSTM)."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            random_seed: Random seed
            device: Device ('cuda' or 'cpu')
        """
        super().__init__(random_seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self._build_model()
    
    def _build_model(self):
        """Build GRU model architecture."""
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        ).to(self.device)
        
        self.output_layer = nn.Linear(self.hidden_size, self.input_size).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sequence_lengths: Optional[np.ndarray] = None) -> 'GRUForecaster':
        """Train the GRU model."""
        logger.info(f"Training GRU on device: {self.device}")
        
        # Normalize
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Dataset and dataloader
        dataset = TimeSeriesDataset(X_scaled, y_scaled, seq_len, sequence_lengths)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            list(self.gru.parameters()) + list(self.output_layer.parameters()),
            lr=self.learning_rate
        )
        
        # Training
        self.gru.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch_X, batch_y, batch_lengths in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_lengths = batch_lengths.cpu()
                
                # Sort by length for pack_padded_sequence
                sorted_lengths, sort_idx = batch_lengths.sort(descending=True)
                batch_X_sorted = batch_X[sort_idx]
                batch_y_sorted = batch_y[sort_idx]
                
                # Pack sequences
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    batch_X_sorted, sorted_lengths.tolist(), batch_first=True
                )
                
                # Forward
                _, hidden = self.gru(packed_input)
                predictions = self.output_layer(hidden[-1])
                
                # Unsort predictions
                unsort_idx = sort_idx.argsort()
                predictions = predictions[unsort_idx]
                
                # Loss
                loss = criterion(predictions, batch_y)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        logger.info("GRU training complete")
        return self
    
    def predict(self, X: np.ndarray, sequence_lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Normalize
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        
        # Default to full length if not provided
        if sequence_lengths is None:
            sequence_lengths = np.array([seq_len] * n_samples)
        
        # Predict
        self.gru.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            lengths_tensor = torch.LongTensor(sequence_lengths)
            
            # Sort by length
            sorted_lengths, sort_idx = lengths_tensor.sort(descending=True)
            X_sorted = X_tensor[sort_idx]
            
            # Pack sequences
            packed_input = nn.utils.rnn.pack_padded_sequence(
                X_sorted, sorted_lengths.tolist(), batch_first=True
            )
            
            # Forward pass
            _, hidden = self.gru(packed_input)
            predictions_scaled = self.output_layer(hidden[-1])
            
            # Unsort predictions
            unsort_idx = sort_idx.argsort()
            predictions_scaled = predictions_scaled[unsort_idx]
            predictions_scaled = predictions_scaled.cpu().numpy()
        
        # Denormalize
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions


def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    target_columns: List[str],
    repo_id_col: str = 'repo_id'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create sequences for time series forecasting.
    
    Args:
        data: DataFrame with time series data
        sequence_length: Length of input sequences
        target_columns: List of column names to predict
        repo_id_col: Name of repository ID column
        
    Returns:
        Tuple of (X, y, repo_ids) where:
        - X: Input sequences (n_samples, sequence_length, n_features)
        - y: Target values (n_samples, n_features)
        - repo_ids: List of repository IDs for each sample
    """
    sequences = []
    targets = []
    repo_ids = []
    
    # Group by repository
    for repo_id, group in data.groupby(repo_id_col):
        group = group.sort_values('quarter_start').reset_index(drop=True)
        
        # Extract features
        features = group[target_columns].values
        
        # Create sequences
        for i in range(len(group) - sequence_length):
            seq = features[i:i+sequence_length]
            target = features[i+sequence_length]
            
            sequences.append(seq)
            targets.append(target)
            repo_ids.append(repo_id)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    return X, y, repo_ids


def save_model(model: BaseForecaster, path: str) -> None:
    """
    Save model to disk.
    
    Args:
        model: Forecaster model
        path: Path to save model
    """
    import pickle
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to: {path}")


def load_model(path: str) -> BaseForecaster:
    """
    Load model from disk.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model
    """
    import pickle
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from: {path}")
    return model


if __name__ == '__main__':
    # Simple test
    logger.info("Testing forecaster models...")
    
    # Generate synthetic data
    n_samples = 1000
    seq_len = 4
    n_features = 5
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples, n_features)
    
    # Test naive model
    naive = NaiveForecaster()
    naive.fit(X, y)
    pred = naive.predict(X[:10])
    logger.info(f"Naive predictions shape: {pred.shape}")
    
    # Test moving average
    ma = MovingAverageForecaster(window_size=3)
    ma.fit(X, y)
    pred = ma.predict(X[:10])
    logger.info(f"Moving average predictions shape: {pred.shape}")
    
    # Test LSTM (small)
    lstm = LSTMForecaster(
        input_size=n_features,
        hidden_size=16,
        num_layers=1,
        epochs=5,
        batch_size=64
    )
    lstm.fit(X[:100], y[:100])
    pred = lstm.predict(X[:10])
    logger.info(f"LSTM predictions shape: {pred.shape}")
    
    logger.info("All tests passed!")
