#!/usr/bin/env python3
"""
Train Forecasting Model

Usage:
    python train_forecaster.py --input data/processed/quarters_labeled.parquet --model lstm
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent))

from forecaster import (
    NaiveForecaster, MovingAverageForecaster,
    LSTMForecaster, GRUForecaster,
    create_sequences, save_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/quarters_labeled.parquet')
    parser.add_argument('--model', type=str, choices=['naive', 'ma', 'lstm', 'gru'], default='lstm')
    parser.add_argument('--sequence-length', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='models/checkpoints')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_parquet(args.input)
    
    # Identify metric columns
    metric_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) 
                   and col not in ['year', 'quarter', 'quarter_index', 
                                   'quarters_since_creation', 'total_quarters',
                                   'activity_score', 'is_active']]
    
    logger.info(f"Metric columns: {metric_cols}")
    
    # Create sequences
    logger.info("Creating sequences...")
    X, y, repo_ids = create_sequences(df, args.sequence_length, metric_cols)
    logger.info(f"Created {len(X)} sequences of shape {X.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Create model
    if args.model == 'naive':
        model = NaiveForecaster(random_seed=args.random_seed)
    elif args.model == 'ma':
        model = MovingAverageForecaster(window_size=3, random_seed=args.random_seed)
    elif args.model == 'lstm':
        model = LSTMForecaster(
            input_size=len(metric_cols),
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            random_seed=args.random_seed
        )
    elif args.model == 'gru':
        model = GRUForecaster(
            input_size=len(metric_cols),
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            random_seed=args.random_seed
        )
    
    # Train
    logger.info(f"Training {args.model} model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    logger.info(f"Test MSE: {metrics['mse']:.6f}")
    logger.info(f"Test MAE: {metrics['mae']:.6f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.6f}")
    
    # Save model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_dir) / f"{args.model}_forecaster.pkl"
    save_model(model, str(model_path))
    
    # Save metrics
    metrics_path = Path(args.output_dir) / f"{args.model}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in metrics.items() 
                  if not isinstance(v, (list, np.ndarray))}, f, indent=2)
    
    logger.info(f"Training complete! Model: {model_path}, Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
