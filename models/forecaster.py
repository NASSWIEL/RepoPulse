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


# ============================================================================
# TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================

class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolutions and residual connections.
    
    Key features:
    - Causal padding (no future information leakage)
    - Dilation for exponentially growing receptive field
    - Residual connections for gradient flow
    - Weight normalization for training stability
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Causal padding: (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        
        # First convolutional layer
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=self.padding, dilation=dilation
            )
        )
        
        # Second convolutional layer
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=self.padding, dilation=dilation
            )
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with proper scaling."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Forward pass with causal convolution and residual connection.
        
        Args:
            x: (batch, channels, sequence_length)
        
        Returns:
            out: (batch, channels, sequence_length)
        """
        # First conv block
        out = self.conv1(x)
        # Remove future padding (causal)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)


class TCNForecaster:
    """
    Temporal Convolutional Network for time-series forecasting.
    
    TCN advantages over RNNs:
    - Parallelizable training (no sequential dependency)
    - Stable gradients (no vanishing/exploding gradient issues)
    - Flexible receptive field (controlled by dilation and depth)
    - Causal convolutions (no information leakage from future)
    
    Architecture:
    - Multiple temporal blocks with increasing dilation (1, 2, 4, 8, ...)
    - Residual connections between blocks
    - Final fully-connected layer for prediction
    
    The receptive field grows exponentially: 2^(num_levels) * kernel_size
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [32, 32, 32],
        kernel_size: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        device: str = None
    ):
        """
        Args:
            input_size: Number of input features
            num_channels: List of channel sizes for each TCN level (determines depth)
            kernel_size: Kernel size for temporal convolutions (typically 2-4)
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            epochs: Maximum training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            device: Device to run on ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build TCN model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Scaler for normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"TCN initialized: {self._count_parameters():,} parameters")
        logger.info(f"  Levels: {len(num_channels)}, Channels: {num_channels}")
        logger.info(f"  Receptive field: ~{2 ** len(num_channels) * kernel_size} timesteps")
    
    def _build_model(self) -> nn.Module:
        """Build TCN architecture with temporal blocks."""
        layers = []
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=self.dropout
                )
            )
        
        # TCN layers
        tcn = nn.Sequential(*layers)
        
        # Final projection layer
        fc = nn.Linear(self.num_channels[-1], self.input_size)
        
        # Complete model
        class TCNModel(nn.Module):
            def __init__(self, tcn, fc):
                super().__init__()
                self.tcn = tcn
                self.fc = fc
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                # TCN expects (batch, features, seq_len)
                x = x.transpose(1, 2)
                
                # Apply TCN
                y = self.tcn(x)
                
                # Take last timestep (causal: only past information)
                y = y[:, :, -1]
                
                # Project to output
                return self.fc(y)
        
        return TCNModel(tcn, fc)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_lengths: Optional[np.ndarray] = None
    ) -> 'TCNForecaster':
        """
        Train the TCN model.
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: Training targets (n_samples, horizon, n_features)
            sequence_lengths: Actual sequence lengths (for padded input) - not used by TCN
        
        Returns:
            self: Trained model
        """
        # Flatten targets if necessary
        if y.ndim == 3:
            y = y.reshape(y.shape[0], -1)
        
        # Normalize data
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train/validation split
        val_split = int(0.9 * len(X_scaled))
        X_train, X_val = X_scaled[:val_split], X_scaled[val_split:]
        y_train, y_val = y_scaled[:val_split], y_scaled[val_split:]
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
        val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y, _ in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y, _ in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    val_losses.append(loss.item())
            
            # Record history
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(self.best_state)
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        sequence_lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            sequence_lengths: Actual sequence lengths (for padded input) - not used by TCN
        
        Returns:
            predictions: Predicted values (n_samples, n_features)
        """
        # Normalize
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_reshaped).reshape(n_samples, seq_len, n_features)
        
        # Create dataset and loader
        dataset = TimeSeriesDataset(X_scaled, np.zeros((len(X_scaled), n_features)), seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_pred = self.model(batch_X)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Denormalize
        predictions = self.scaler_y.inverse_transform(predictions)
        
        return predictions
    
    def forecast_autoregressive(
        self,
        initial_sequence: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Autoregressive forecasting (predict multiple steps ahead).
        
        Args:
            initial_sequence: Initial sequence (seq_len, n_features)
            steps: Number of steps to forecast
        
        Returns:
            forecasts: Predicted sequence (steps, n_features)
        """
        forecasts = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(steps):
            # Predict next step
            pred = self.predict(current_sequence[np.newaxis, :, :])[0]
            forecasts.append(pred)
            
            # Update sequence (append prediction, remove oldest)
            current_sequence = np.vstack([current_sequence[1:], pred])
        
        return np.array(forecasts)
