#!/usr/bin/env python3
"""
Baseline and advanced autoregressive forecasting models.

Implements:
1. Baseline: LastQuarter, MovingAverage
2. Advanced: LSTM, GRU
3. Training with teacher forcing
4. Rolling autoregressive prediction
5. Activity label generation from forecasts

Usage:
    python train_forecasters.py --model lstm --epochs 50
    python train_forecasters.py --model baseline --baseline_type moving_average
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Tuple, List
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

# MLflow for experiment tracking
import mlflow
import mlflow.pytorch
from mlflow_utils import (
    setup_mlflow_experiment,
    log_params_from_config,
    log_training_metrics,
    log_pytorch_model,
    log_dataset_info
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time-series sequences."""
    
    def __init__(self, data_path: Path, normalize: bool = True, stats_path: Path = None):
        """Load and optionally normalize data."""
        data = np.load(data_path)
        
        self.lookback_features = data['lookback_features']
        self.target_metrics = data['target_metrics']
        self.target_labels = data['target_labels']
        self.target_scores = data['target_scores']
        
        # Normalization
        if normalize and stats_path and stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            
            mean = np.array(stats['mean'])
            std = np.array(stats['std'])
            
            # Log normalization info
            logger.info(f"Applying normalization with stats from {stats_path}")
            if 'zero_variance_features' in stats and stats['zero_variance_features']:
                logger.warning(f"Zero-variance features (scaling disabled): {stats['zero_variance_features']}")
            
            # Additional guard against zero std (should already be handled in prep, but double-check)
            std = np.where(std < 1e-8, 1.0, std)
            
            # Check for NaN/Inf before normalization
            if np.isnan(self.lookback_features).any():
                raise ValueError(f"Lookback features contain {np.isnan(self.lookback_features).sum()} NaN values BEFORE normalization!")
            if np.isnan(self.target_metrics).any():
                raise ValueError(f"Target metrics contain {np.isnan(self.target_metrics).sum()} NaN values BEFORE normalization!")
            
            # Normalize lookback
            self.lookback_features = (self.lookback_features - mean) / std
            # Normalize targets
            self.target_metrics = (self.target_metrics - mean) / std
            
            # Sanity check for NaN/Inf after normalization
            if np.isnan(self.lookback_features).any():
                n_nans = np.isnan(self.lookback_features).sum()
                logger.error(f"❌ Lookback features contain {n_nans} NaN values after normalization!")
                logger.error(f"Mean: {mean}")
                logger.error(f"Std: {std}")
                raise ValueError(f"Lookback features contain {n_nans} NaN values after normalization!")
            
            if np.isnan(self.target_metrics).any():
                n_nans = np.isnan(self.target_metrics).sum()
                logger.error(f"❌ Target metrics contain {n_nans} NaN values after normalization!")
                raise ValueError(f"Target metrics contain {n_nans} NaN values after normalization!")
            
            if np.isinf(self.lookback_features).any():
                n_infs = np.isinf(self.lookback_features).sum()
                logger.error(f"❌ Lookback features contain {n_infs} Inf values after normalization!")
                raise ValueError(f"Lookback features contain {n_infs} Inf values after normalization!")
            
            if np.isinf(self.target_metrics).any():
                n_infs = np.isinf(self.target_metrics).sum()
                logger.error(f"❌ Target metrics contain {n_infs} Inf values after normalization!")
                raise ValueError(f"Target metrics contain {n_infs} Inf values after normalization!")
            
            logger.info("✓ Data normalized successfully (no NaN/Inf)")
            
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None
        
    def __len__(self):
        return len(self.lookback_features)
    
    def __getitem__(self, idx):
        return {
            'lookback': torch.FloatTensor(self.lookback_features[idx]),
            'target': torch.FloatTensor(self.target_metrics[idx]),
            'label': torch.LongTensor([self.target_labels[idx]]),
            'score': torch.FloatTensor([self.target_scores[idx]])
        }


class LSTMForecaster(nn.Module):
    """LSTM-based autoregressive forecaster."""
    
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, n_features)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: [batch, lookback, n_features]
        
        Returns:
            [batch, 1, n_features] - next quarter prediction
        """
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.unsqueeze(1)


class GRUForecaster(nn.Module):
    """GRU-based autoregressive forecaster."""
    
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, n_features)
        
    def forward(self, x):
        """Forward pass."""
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.unsqueeze(1)


class BaselineForecaster:
    """Baseline forecasting methods."""
    
    def __init__(self, method: str = 'last'):
        """
        Initialize baseline forecaster.
        
        Args:
            method: 'last' (last quarter carry-forward) or 'moving_average'
        """
        self.method = method
        
    def predict(self, lookback: np.ndarray) -> np.ndarray:
        """
        Predict next quarter.
        
        Args:
            lookback: [batch, lookback, n_features] or [lookback, n_features]
        
        Returns:
            [batch, 1, n_features] or [1, n_features]
        """
        if lookback.ndim == 2:
            lookback = lookback[np.newaxis, :]
        
        if self.method == 'last':
            # Use last quarter
            prediction = lookback[:, -1:, :]
        elif self.method == 'moving_average':
            # Average of all lookback quarters
            prediction = lookback.mean(axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return prediction


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    checkpoint_path: Path = None,
    log_to_mlflow: bool = True
) -> Dict:
    """
    Train neural forecasting model with MLflow tracking.
    
    MLflow Integration:
    - Logs training and validation loss at each epoch
    - Tracks best model performance
    - Enables comparison across different runs
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_dev_loss = float('inf')
    history = {'train_loss': [], 'dev_loss': []}
    patience = 10
    patience_counter = 0
    
    logger.info(f"\nTraining on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            lookback = batch['lookback'].to(device)
            target = batch['target'].to(device)
            
            # Check for NaN in batch data
            if torch.isnan(lookback).any():
                logger.error(f"❌ NaN detected in lookback at batch {batch_idx}")
                raise ValueError(f"NaN in training batch {batch_idx}")
            if torch.isnan(target).any():
                logger.error(f"❌ NaN detected in target at batch {batch_idx}")
                raise ValueError(f"NaN in training target {batch_idx}")
            
            optimizer.zero_grad()
            prediction = model(lookback)
            
            # Check for NaN in prediction
            if torch.isnan(prediction).any():
                logger.error(f"❌ NaN in model prediction at epoch {epoch}, batch {batch_idx}")
                logger.error(f"Lookback stats: min={lookback.min():.4f}, max={lookback.max():.4f}, mean={lookback.mean():.4f}")
                raise ValueError(f"Model produced NaN predictions")
            
            loss = criterion(prediction, target)
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.error(f"❌ NaN loss at epoch {epoch}, batch {batch_idx}")
                logger.error(f"Prediction stats: min={prediction.min():.4f}, max={prediction.max():.4f}")
                logger.error(f"Target stats: min={target.min():.4f}, max={target.max():.4f}")
                raise ValueError(f"NaN loss detected")
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        dev_losses = []
        
        with torch.no_grad():
            for batch in dev_loader:
                lookback = batch['lookback'].to(device)
                target = batch['target'].to(device)
                
                prediction = model(lookback)
                loss = criterion(prediction, target)
                dev_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        dev_loss = np.mean(dev_losses)
        
        history['train_loss'].append(train_loss)
        history['dev_loss'].append(dev_loss)
        
        # Log metrics to MLflow
        if log_to_mlflow:
            log_training_metrics({
                'train_loss': train_loss,
                'dev_loss': dev_loss,
                'best_dev_loss': best_dev_loss
            }, step=epoch)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Dev Loss: {dev_loss:.6f}")
        
        # Early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_loss': dev_loss,
                }, checkpoint_path)
                logger.info(f"  → Saved checkpoint (dev_loss improved to {dev_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return history


def main():
    """
    Main training script with comprehensive MLflow integration.
    
    MLflow captures:
    1. All hyperparameters (model architecture, training config)
    2. Training metrics at each epoch (train/dev loss)
    3. Dataset information and statistics
    4. Trained model artifacts with signature
    5. Training plots and checkpoints
    """
    parser = argparse.ArgumentParser(description="Train autoregressive forecasting models")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'gru', 'baseline'])
    parser.add_argument('--baseline_type', type=str, default='last', choices=['last', 'moving_average'])
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='data/processed/timeseries')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--no_mlflow', action='store_true', help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info(f"TRAINING {args.model.upper()} FORECASTER")
    logger.info("="*70)
    
    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)
    
    n_features = metadata['n_features']
    logger.info(f"Features: {n_features} ({', '.join(metadata['feature_names'])})")
    logger.info(f"Lookback: {metadata['lookback']} quarters")
    
    if args.model == 'baseline':
        logger.info(f"\nBaseline method: {args.baseline_type}")
        logger.info("No training required for baseline models")
        return
    
    # MLflow Experiment Setup
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        run_name = f"{args.model}-h{args.hidden_size}-l{args.num_layers}-e{args.epochs}"
        
        with setup_mlflow_experiment(
            config=config,
            experiment_type='forecasting',
            run_name=run_name,
            tags={'model_type': args.model, 'architecture': 'RNN'}
        ) as run:
            logger.info(f"\n📊 MLflow Run ID: {run.info.run_id}")
            logger.info(f"📊 MLflow Experiment: {run.info.experiment_id}")
            
            # Log all hyperparameters
            mlflow.log_params({
                'model_type': args.model,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'device': device,
                'n_features': n_features,
                'lookback': metadata['lookback'],
                'random_seed': SEED
            })
            
            # Create datasets
            logger.info("\nLoading data...")
            train_dataset = TimeSeriesDataset(
                data_dir / 'train.npz',
                normalize=True,
                stats_path=data_dir / 'feature_stats.json'
            )
            dev_dataset = TimeSeriesDataset(
                data_dir / 'dev.npz',
                normalize=True,
                stats_path=data_dir / 'feature_stats.json'
            )
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Log dataset information
            log_dataset_info(
                dataset_path=str(data_dir / 'train.npz'),
                dataset_type='train',
                n_samples=len(train_dataset),
                n_features=n_features,
                additional_info={'lookback': metadata['lookback']}
            )
            log_dataset_info(
                dataset_path=str(data_dir / 'dev.npz'),
                dataset_type='dev',
                n_samples=len(dev_dataset),
                n_features=n_features
            )
            
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Dev samples: {len(dev_dataset)}")
            
            # Create model
            logger.info(f"\nInitializing {args.model.upper()} model...")
            if args.model == 'lstm':
                model = LSTMForecaster(
                    n_features=n_features,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                )
            elif args.model == 'gru':
                model = GRUForecaster(
                    n_features=n_features,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                )
            
            n_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param('model_parameters', n_params)
            logger.info(f"Model parameters: {n_params:,}")
            
            # Train
            checkpoint_path = output_dir / f'{args.model}_best.pt'
            start_time = time.time()
            
            history = train_model(
                model=model,
                train_loader=train_loader,
                dev_loader=dev_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                checkpoint_path=checkpoint_path,
                log_to_mlflow=True
            )
            
            training_time = time.time() - start_time
            logger.info(f"\nTraining completed in {training_time/60:.2f} minutes")
            
            # Log final metrics and artifacts
            mlflow.log_metric('training_time_minutes', training_time/60)
            mlflow.log_metric('final_train_loss', history['train_loss'][-1])
            mlflow.log_metric('final_dev_loss', history['dev_loss'][-1])
            mlflow.log_metric('best_dev_loss', min(history['dev_loss']))
            
            # Save and log training history
            history_path = output_dir / f'{args.model}_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            mlflow.log_artifact(str(history_path), artifact_path='training_history')
            
            # Log model checkpoint
            mlflow.log_artifact(str(checkpoint_path), artifact_path='checkpoints')
            
            # Log PyTorch model with signature
            model.eval()
            sample_input = train_dataset[0]['lookback'].unsqueeze(0).numpy()
            sample_output = train_dataset[0]['target'].unsqueeze(0).numpy()
            
            log_pytorch_model(
                model=model,
                model_name=f"{args.model}_forecaster",
                input_sample=sample_input,
                output_sample=sample_output,
                register_model=True,
                registered_model_name=f"forecaster-{args.model}"
            )
            
            logger.info(f"\n✅ MLflow tracking complete!")
            logger.info(f"📊 View results: mlflow ui --backend-store-uri mlruns")
            logger.info(f"🔗 Run ID: {run.info.run_id}")
            
    else:
        # Original training without MLflow
        logger.info("\nMLflow tracking disabled")
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            data_dir / 'train.npz',
            normalize=True,
            stats_path=data_dir / 'feature_stats.json'
        )
        dev_dataset = TimeSeriesDataset(
            data_dir / 'dev.npz',
            normalize=True,
            stats_path=data_dir / 'feature_stats.json'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        if args.model == 'lstm':
            model = LSTMForecaster(
                n_features=n_features,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        elif args.model == 'gru':
            model = GRUForecaster(
                n_features=n_features,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout
            )
        
        # Train
        checkpoint_path = output_dir / f'{args.model}_best.pt'
        history = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_path=checkpoint_path,
            log_to_mlflow=False
        )
        
        # Save history
        with open(output_dir / f'{args.model}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining complete! Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
