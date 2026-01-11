#!/usr/bin/env python3
"""
Prepare time-series data for autoregressive forecasting.

This module:
1. Loads labeled quarterly data
2. Creates lagged features for all metrics
3. Engineers temporal features
4. Handles missing values consistently
5. Splits data temporally (train/dev/test)
6. Saves prepared sequences

Usage:
    python prepare_timeseries_data.py --config config/config.yaml
    python prepare_timeseries_data.py --lookback 4 --horizon 1
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

import pandas as pd
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimeSeriesDataPreparator:
    """Prepares time-series data for autoregressive forecasting."""
    
    # Metrics to forecast
    FORECAST_FEATURES = [
        'commit_count',
        'total_contributors', 
        'issue_count',
        'issue_closed',
        'pr_count',
        'pr_merged',
        'star_count'
    ]
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        lookback: int = 4,
        horizon: int = 1
    ):
        """
        Initialize data preparator.
        
        Args:
            config_path: Path to config file
            lookback: Number of past quarters to use as features
            horizon: Number of future quarters to predict
        """
        self.config = self._load_config(config_path)
        self.lookback = lookback
        self.horizon = horizon
        self.input_path = Path(self.config["data"]["labeled_data_path"])
        self.output_dir = Path(self.config["data"]["output_path"]) / "timeseries"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare(self) -> None:
        """Main preparation pipeline."""
        logger.info("="*70)
        logger.info("TIME-SERIES DATA PREPARATION")
        logger.info("="*70)
        logger.info(f"Lookback window: {self.lookback} quarters")
        logger.info(f"Forecast horizon: {self.horizon} quarter(s)")
        
        # Load data
        logger.info(f"\nLoading labeled data from {self.input_path}")
        data = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(data)} records from {data['repo_id'].nunique()} repositories")
        
        # Fill NaN values with 0 (quarters with no activity)
        features_to_fill = self.FORECAST_FEATURES + ['activity_score']
        nan_counts = data[features_to_fill].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"\nFound NaN values in features:")
            for feat in features_to_fill:
                if nan_counts[feat] > 0:
                    logger.warning(f"  {feat}: {nan_counts[feat]} NaNs")
            logger.info("Filling NaN values with 0 (quarters with no activity)")
            data[features_to_fill] = data[features_to_fill].fillna(0)
        
        # Sort chronologically by repository
        data = data.sort_values(['repo_id', 'year', 'quarter']).reset_index(drop=True)
        
        # Engineer features
        logger.info("\nEngineering temporal features...")
        data = self._engineer_features(data)
        
        # Create sequences
        logger.info("\nCreating sequences with lagged features...")
        sequences = self._create_sequences(data)
        
        logger.info(f"Created {len(sequences)} sequences")
        
        # Temporal split
        logger.info("\nPerforming temporal train/dev/test split...")
        train, dev, test = self._temporal_split(sequences)
        
        logger.info(f"  Train: {len(train)} sequences")
        logger.info(f"  Dev:   {len(dev)} sequences")
        logger.info(f"  Test:  {len(test)} sequences")
        
        # Save splits
        self._save_splits(train, dev, test, data)
        
        logger.info("\n" + "="*70)
        logger.info("DATA PREPARATION COMPLETE")
        logger.info("="*70)
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer temporal features (removed cumulative features to avoid zero variance)."""
        
        # Quarter index (continuous time indicator)
        data['quarter_index'] = (data['year'] - data['year'].min()) * 4 + data['quarter']
        
        # Per-repository features
        for repo in data['repo_id'].unique():
            repo_mask = data['repo_id'] == repo
            repo_data = data[repo_mask].copy()
            
            # Quarters since repository creation (in this dataset)
            repo_data['quarters_since_start'] = range(len(repo_data))
            
            # NOTE: Removed cumulative features - they have zero variance and cause NaN in normalization
            # Using only per-quarter deltas/counts which have proper variance
            
            # Update main dataframe
            data.loc[repo_mask, 'quarters_since_start'] = repo_data['quarters_since_start'].values
        
        logger.info(f"  Added quarter_index and quarters_since_start (removed cumulative features)")
        
        return data
    
    def _create_sequences(self, data: pd.DataFrame) -> List[Dict]:
        """
        Create sequences with lagged features.
        
        Each sequence contains:
        - lookback_features: [lookback, n_features] past quarters
        - target_metrics: [horizon, n_features] future metrics to predict
        - target_label: binary activity label for forecast quarter
        - metadata: repo_id, year, quarter, etc.
        """
        sequences = []
        
        # Group by repository
        for repo_id, repo_data in data.groupby('repo_id'):
            repo_data = repo_data.sort_values(['year', 'quarter']).reset_index(drop=True)
            
            # Need at least lookback + horizon quarters
            if len(repo_data) < self.lookback + self.horizon:
                continue
            
            # Sliding window
            for i in range(len(repo_data) - self.lookback - self.horizon + 1):
                # Lookback window
                lookback_data = repo_data.iloc[i:i+self.lookback]
                
                # Target window
                target_data = repo_data.iloc[i+self.lookback:i+self.lookback+self.horizon]
                
                # Extract features
                lookback_features = lookback_data[self.FORECAST_FEATURES].values
                target_metrics = target_data[self.FORECAST_FEATURES].values
                target_label = target_data['is_active'].values[0]  # First quarter in horizon
                target_score = target_data['activity_score'].values[0]
                
                # Metadata
                target_quarter = target_data.iloc[0]
                
                sequence = {
                    'repo_id': repo_id,
                    'target_year': int(target_quarter['year']),
                    'target_quarter': int(target_quarter['quarter']),
                    'target_quarter_index': int(target_quarter['quarter_index']),
                    'lookback_features': lookback_features.astype(np.float32),
                    'target_metrics': target_metrics.astype(np.float32),
                    'target_label': int(target_label),
                    'target_score': float(target_score),
                    'quarters_since_start': int(target_quarter['quarters_since_start'])
                }
                
                sequences.append(sequence)
        
        return sequences
    
    def _temporal_split(
        self,
        sequences: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split sequences temporally.
        
        Strategy: Use quarter_index for temporal ordering
        - Train: earliest 70% of quarters
        - Dev: next 15% 
        - Test: final 15%
        """
        # Sort by target quarter index
        sequences = sorted(sequences, key=lambda x: (x['target_quarter_index'], x['repo_id']))
        
        # Get unique quarter indices
        quarter_indices = sorted(set(s['target_quarter_index'] for s in sequences))
        
        n_quarters = len(quarter_indices)
        train_end_idx = quarter_indices[int(n_quarters * 0.70)]
        dev_end_idx = quarter_indices[int(n_quarters * 0.85)]
        
        train = [s for s in sequences if s['target_quarter_index'] <= train_end_idx]
        dev = [s for s in sequences if train_end_idx < s['target_quarter_index'] <= dev_end_idx]
        test = [s for s in sequences if s['target_quarter_index'] > dev_end_idx]
        
        # Log split info
        logger.info(f"\nTemporal split details:")
        logger.info(f"  Total quarters: {n_quarters}")
        logger.info(f"  Train quarters: {quarter_indices[0]} to {train_end_idx}")
        logger.info(f"  Dev quarters: {train_end_idx+1} to {dev_end_idx}")
        logger.info(f"  Test quarters: {dev_end_idx+1} to {quarter_indices[-1]}")
        
        return train, dev, test
    
    def _save_splits(
        self,
        train: List[Dict],
        dev: List[Dict],
        test: List[Dict],
        original_data: pd.DataFrame
    ) -> None:
        """Save prepared data splits."""
        
        # Save sequences as numpy arrays for efficiency
        for split_name, split_data in [('train', train), ('dev', dev), ('test', test)]:
            # Convert to structured format
            n_samples = len(split_data)
            
            # Save as npz (compressed numpy format)
            np.savez_compressed(
                self.output_dir / f'{split_name}.npz',
                lookback_features=np.array([s['lookback_features'] for s in split_data]),
                target_metrics=np.array([s['target_metrics'] for s in split_data]),
                target_labels=np.array([s['target_label'] for s in split_data]),
                target_scores=np.array([s['target_score'] for s in split_data]),
                repo_ids=np.array([s['repo_id'] for s in split_data]),
                target_years=np.array([s['target_year'] for s in split_data]),
                target_quarters=np.array([s['target_quarter'] for s in split_data])
            )
            
            logger.info(f"Saved {split_name} split: {n_samples} sequences")
        
        # Save metadata
        metadata = {
            'lookback': self.lookback,
            'horizon': self.horizon,
            'n_features': len(self.FORECAST_FEATURES),
            'feature_names': self.FORECAST_FEATURES,
            'n_train': len(train),
            'n_dev': len(dev),
            'n_test': len(test),
            'n_repositories': original_data['repo_id'].nunique()
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata")
        
        # Save feature statistics for normalization (with guards for zero std)
        train_features = np.concatenate([s['lookback_features'] for s in train], axis=0)
        
        feature_mean = train_features.mean(axis=0)
        feature_std = train_features.std(axis=0)
        
        # Guard against zero/near-zero std - replace with 1.0 to disable scaling for that feature
        feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
        
        feature_stats = {
            'mean': feature_mean.tolist(),
            'std': feature_std.tolist(),
            'min': train_features.min(axis=0).tolist(),
            'max': train_features.max(axis=0).tolist()
        }
        
        with open(self.output_dir / 'feature_stats.json', 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        logger.info(f"Saved feature statistics for normalization (guarded against zero std)")
        
        # Sanity check for NaNs
        if np.isnan(feature_mean).any() or np.isnan(feature_std).any():
            logger.error("ERROR: Feature statistics contain NaN values!")
            logger.error(f"Mean NaNs: {np.isnan(feature_mean).sum()}")
            logger.error(f"Std NaNs: {np.isnan(feature_std).sum()}")
        else:
            logger.info("Feature statistics are clean (no NaNs)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare time-series data for autoregressive forecasting"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=4,
        help='Number of past quarters to use as features'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=1,
        help='Number of future quarters to predict'
    )
    
    args = parser.parse_args()
    
    # Prepare data
    preparator = TimeSeriesDataPreparator(
        config_path=args.config,
        lookback=args.lookback,
        horizon=args.horizon
    )
    preparator.prepare()
    
    logger.info("\nData preparation complete!")


if __name__ == "__main__":
    main()
