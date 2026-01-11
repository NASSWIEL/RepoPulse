"""
Autoregressive evaluation of forecasting models with activity classification.

This script:
1. Loads trained forecasting models
2. Performs rolling autoregressive multi-step forecasting on test data
3. Converts numeric forecasts to activity labels using the threshold
4. Evaluates both numeric forecasting and binary classification performance
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from train_forecasters import LSTMForecaster, GRUForecaster, BaselineForecaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivityConverter:
    """Convert numeric forecasts to activity labels."""
    
    # Weights for computing activity score (same as labeling)
    FEATURE_WEIGHTS = {
        'commit_count': 1.0,
        'total_contributors': 2.0,
        'issue_count': 0.5,
        'issue_closed': 1.0,
        'pr_count': 0.8,
        'pr_merged': 1.5,
        'star_count': 0.01,
    }
    
    def __init__(self, threshold: float, feature_names: List[str]):
        """
        Args:
            threshold: Activity score threshold (from metadata)
            feature_names: List of feature names in order
        """
        self.threshold = threshold
        self.feature_names = feature_names
        
        # Create weight array matching feature order
        self.weights = np.array([
            self.FEATURE_WEIGHTS[feat] for feat in feature_names
        ])
    
    def compute_activity_scores(self, forecasts: np.ndarray) -> np.ndarray:
        """
        Compute activity scores from denormalized forecasts.
        
        Args:
            forecasts: Shape (n_samples, n_features)
        
        Returns:
            scores: Shape (n_samples,)
        """
        return np.sum(forecasts * self.weights, axis=1)
    
    def classify(self, forecasts: np.ndarray) -> np.ndarray:
        """
        Convert forecasts to binary activity labels.
        
        Args:
            forecasts: Shape (n_samples, n_features)
        
        Returns:
            labels: Shape (n_samples,), dtype bool
        """
        scores = self.compute_activity_scores(forecasts)
        return scores >= self.threshold


class AutoregressiveEvaluator:
    """Evaluate models with rolling autoregressive forecasting."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_stats: Dict[str, np.ndarray],
        activity_converter: ActivityConverter,
    ):
        self.model = model
        self.device = device
        self.feature_stats = feature_stats
        self.activity_converter = activity_converter
        
        # Extract stats as arrays
        self.mean = np.array(feature_stats['mean'])
        self.std = np.array(feature_stats['std'])
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics."""
        return (x - self.mean) / self.std
    
    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return x_norm * self.std + self.mean
    
    def forecast_one_step(self, lookback: np.ndarray) -> np.ndarray:
        """
        Forecast one step ahead given lookback window.
        
        Args:
            lookback: Shape (lookback_len, n_features), denormalized
        
        Returns:
            prediction: Shape (n_features,), denormalized
        """
        # Normalize and prepare for model
        lookback_norm = self.normalize(lookback)
        x = torch.FloatTensor(lookback_norm).unsqueeze(0).to(self.device)  # (1, lookback, features)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            pred_norm = self.model(x).squeeze(0).squeeze(0).cpu().numpy()  # (features,)
        
        # Denormalize
        pred = self.denormalize(pred_norm)
        
        # Clip negative values to 0 (counts can't be negative)
        pred = np.maximum(pred, 0)
        
        return pred
    
    def forecast_autoregressive(
        self,
        initial_lookback: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Perform autoregressive multi-step forecasting.
        
        Args:
            initial_lookback: Shape (lookback_len, n_features), denormalized
            n_steps: Number of steps to forecast
        
        Returns:
            predictions: Shape (n_steps, n_features), denormalized
        """
        lookback = initial_lookback.copy()
        predictions = []
        
        for _ in range(n_steps):
            # Forecast next step
            pred = self.forecast_one_step(lookback)
            predictions.append(pred)
            
            # Update lookback window (rolling)
            lookback = np.vstack([lookback[1:], pred])
        
        return np.array(predictions)
    
    def evaluate_batch(
        self,
        lookbacks: np.ndarray,
        targets: np.ndarray,
        max_horizon: int = 4,
    ) -> Dict[str, any]:
        """
        Evaluate model on a batch of sequences with autoregressive forecasting.
        
        Args:
            lookbacks: Shape (n_samples, lookback_len, n_features), normalized
            targets: Shape (n_samples, available_horizon, n_features), normalized
            max_horizon: Maximum forecasting horizon to evaluate
        
        Returns:
            results: Dictionary with evaluation metrics
        """
        n_samples = lookbacks.shape[0]
        n_features = lookbacks.shape[2]
        available_horizon = targets.shape[1]
        
        # Limit max_horizon to what's available in targets
        max_horizon = min(max_horizon, available_horizon)
        
        logger.info(f"Evaluating up to horizon {max_horizon} (available: {available_horizon})")
        
        # Denormalize inputs
        lookbacks_denorm = self.denormalize(lookbacks.reshape(-1, n_features)).reshape(lookbacks.shape)
        targets_denorm = self.denormalize(targets.reshape(-1, n_features)).reshape(targets.shape)
        
        # Collect predictions for each horizon
        all_predictions = {h: [] for h in range(1, max_horizon + 1)}
        all_targets = {h: [] for h in range(1, max_horizon + 1)}
        
        logger.info(f"Running autoregressive forecasting for {n_samples} samples...")
        
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i + 1}/{n_samples} samples")
            
            # Get initial lookback
            initial_lookback = lookbacks_denorm[i]  # (lookback_len, features)
            
            # Forecast multiple steps
            predictions = self.forecast_autoregressive(initial_lookback, max_horizon)
            
            # Store predictions and targets for each horizon
            for h in range(1, max_horizon + 1):
                all_predictions[h].append(predictions[h - 1])
                all_targets[h].append(targets_denorm[i, h - 1])
        
        # Convert to arrays
        for h in range(1, max_horizon + 1):
            all_predictions[h] = np.array(all_predictions[h])
            all_targets[h] = np.array(all_targets[h])
        
        # Compute metrics for each horizon
        results = {}
        for h in range(1, max_horizon + 1):
            preds = all_predictions[h]
            targs = all_targets[h]
            
            # Numeric metrics
            mse = mean_squared_error(targs, preds)
            mae = mean_absolute_error(targs, preds)
            rmse = np.sqrt(mse)
            
            # Per-feature metrics
            feature_mse = np.mean((targs - preds) ** 2, axis=0)
            feature_mae = np.mean(np.abs(targs - preds), axis=0)
            
            # Activity classification metrics
            pred_labels = self.activity_converter.classify(preds)
            true_labels = self.activity_converter.classify(targs)
            
            precision = precision_score(true_labels, pred_labels, zero_division=0)
            recall = recall_score(true_labels, pred_labels, zero_division=0)
            f1 = f1_score(true_labels, pred_labels, zero_division=0)
            
            # Scores for AUC
            pred_scores = self.activity_converter.compute_activity_scores(preds)
            true_scores = self.activity_converter.compute_activity_scores(targs)
            
            try:
                roc_auc = roc_auc_score(true_labels, pred_scores)
            except ValueError:
                roc_auc = None
            
            try:
                pr_auc = average_precision_score(true_labels, pred_scores)
            except ValueError:
                pr_auc = None
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            results[f'horizon_{h}'] = {
                'numeric': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'per_feature_mse': feature_mse.tolist(),
                    'per_feature_mae': feature_mae.tolist(),
                },
                'classification': {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'roc_auc': float(roc_auc) if roc_auc is not None else None,
                    'pr_auc': float(pr_auc) if pr_auc is not None else None,
                    'confusion_matrix': cm.tolist(),
                    'support': {
                        'inactive': int(np.sum(~true_labels)),
                        'active': int(np.sum(true_labels)),
                    }
                }
            }
        
        return results


def load_model(
    model_type: str,
    checkpoint_path: Path,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    
    if model_type == 'lstm':
        model = LSTMForecaster(input_size, hidden_size, num_layers, dropout=0.2)
    elif model_type == 'gru':
        model = GRUForecaster(input_size, hidden_size, num_layers, dropout=0.2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")
    logger.info(f"  Best dev loss: {checkpoint['dev_loss']:.6f}")
    
    return model


class BaselineWrapper(torch.nn.Module):
    """Wrapper to make BaselineForecaster compatible with PyTorch interface."""
    
    def __init__(self, baseline_forecaster):
        super().__init__()
        self.baseline = baseline_forecaster
    
    def forward(self, x):
        """Forward pass compatible with PyTorch."""
        # x is [batch, lookback, features] tensor
        x_np = x.cpu().numpy()
        pred_np = self.baseline.predict(x_np)
        return torch.FloatTensor(pred_np)
    
    def eval(self):
        """Compatibility method."""
        return self


def evaluate_baseline(
    baseline_type: str,
    test_lookbacks: np.ndarray,
    test_targets: np.ndarray,
    feature_stats: Dict[str, np.ndarray],
    activity_converter: ActivityConverter,
    max_horizon: int = 4,
) -> Dict[str, any]:
    """Evaluate baseline model."""
    
    # Create baseline model
    device = torch.device('cpu')
    baseline = BaselineForecaster(baseline_type)
    model = BaselineWrapper(baseline)
    
    # Create evaluator
    evaluator = AutoregressiveEvaluator(
        model, device, feature_stats, activity_converter
    )
    
    # Evaluate
    results = evaluator.evaluate_batch(test_lookbacks, test_targets, max_horizon)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate forecasting models')
    parser.add_argument('--model', type=str, default='lstm', 
                       choices=['lstm', 'gru', 'last', 'avg', 'moving_average'],
                       help='Model to evaluate')
    parser.add_argument('--hidden_size', type=int, default=32,
                       help='Hidden size for neural models')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of layers for neural models')
    parser.add_argument('--max_horizon', type=int, default=4,
                       help='Maximum forecasting horizon')
    parser.add_argument('--data_dir', type=Path, 
                       default=Path('data/processed/timeseries'),
                       help='Path to timeseries data directory')
    parser.add_argument('--checkpoint_dir', type=Path,
                       default=Path('models/checkpoints'),
                       help='Path to model checkpoints')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('models/results'),
                       help='Path to save results')
    
    args = parser.parse_args()
    
    # Map 'avg' to 'moving_average' for baseline
    if args.model == 'avg':
        args.model = 'moving_average'
    
    logger.info("=" * 70)
    logger.info(f"EVALUATING {args.model.upper()} FORECASTER")
    logger.info("=" * 70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(args.data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['feature_names']
    activity_threshold = metadata['activity_threshold']
    
    logger.info(f"Features: {len(feature_names)} ({', '.join(feature_names)})")
    logger.info(f"Lookback: {metadata['lookback']} quarters")
    logger.info(f"Activity threshold: {activity_threshold:.2f}")
    logger.info("")
    
    # Load feature statistics
    with open(args.data_dir / 'feature_stats.json', 'r') as f:
        feature_stats = json.load(f)
    
    # Load test data
    logger.info("Loading test data...")
    test_data = np.load(args.data_dir / 'test.npz')
    test_lookbacks = test_data['lookback_features']
    test_targets = test_data['target_metrics']
    
    logger.info(f"Test samples: {len(test_lookbacks)}")
    logger.info("")
    
    # Create activity converter
    activity_converter = ActivityConverter(activity_threshold, feature_names)
    
    # Load and evaluate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model in ['lstm', 'gru']:
        # Neural model
        checkpoint_path = args.checkpoint_dir / f'{args.model}_best.pt'
        
        model = load_model(
            args.model,
            checkpoint_path,
            len(feature_names),
            args.hidden_size,
            args.num_layers,
            device,
        )
        
        logger.info("")
        logger.info("Running evaluation...")
        
        evaluator = AutoregressiveEvaluator(
            model, device, feature_stats, activity_converter
        )
        
        results = evaluator.evaluate_batch(
            test_lookbacks, test_targets, args.max_horizon
        )
    
    else:
        # Baseline model
        logger.info("")
        logger.info("Running evaluation...")
        
        # Map model name to baseline method
        baseline_method = args.model if args.model != 'avg' else 'moving_average'
        
        results = evaluate_baseline(
            baseline_method,
            test_lookbacks,
            test_targets,
            feature_stats,
            activity_converter,
            args.max_horizon,
        )
    
    # Add metadata to results
    results['metadata'] = {
        'model': args.model,
        'hidden_size': args.hidden_size if args.model in ['lstm', 'gru'] else None,
        'num_layers': args.num_layers if args.model in ['lstm', 'gru'] else None,
        'feature_names': feature_names,
        'activity_threshold': activity_threshold,
        'n_test_samples': len(test_lookbacks),
    }
    
    # Save results
    output_path = args.output_dir / f'{args.model}_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    # Print summary for each horizon
    for h in range(1, args.max_horizon + 1):
        horizon_results = results[f'horizon_{h}']
        logger.info(f"\nHorizon {h}:")
        logger.info(f"  Numeric:")
        logger.info(f"    RMSE: {horizon_results['numeric']['rmse']:.4f}")
        logger.info(f"    MAE:  {horizon_results['numeric']['mae']:.4f}")
        logger.info(f"  Classification:")
        logger.info(f"    Precision: {horizon_results['classification']['precision']:.4f}")
        logger.info(f"    Recall:    {horizon_results['classification']['recall']:.4f}")
        logger.info(f"    F1:        {horizon_results['classification']['f1']:.4f}")
        if horizon_results['classification']['roc_auc'] is not None:
            logger.info(f"    ROC-AUC:   {horizon_results['classification']['roc_auc']:.4f}")
        if horizon_results['classification']['pr_auc'] is not None:
            logger.info(f"    PR-AUC:    {horizon_results['classification']['pr_auc']:.4f}")
    
    logger.info("")
    logger.info(f"Saved results to {output_path}")


if __name__ == '__main__':
    main()
