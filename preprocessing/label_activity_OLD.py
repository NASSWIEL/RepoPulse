#!/usr/bin/env python3
"""
Activity Labeling Script for GitHub Repositories

This script:
- Reads quarterly aggregated data
- Computes a weighted activity score
- Determines optimal threshold using validation data
- Labels repositories as active (1) or inactive (0)
- Outputs labeled dataset

Usage:
    python label_activity.py --input data/processed/quarters.parquet --output data/processed/quarters_labeled.parquet
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActivityLabeler:
    """Label repository activity based on weighted metrics."""
    
    # Default weights for activity score
    # Rationale:
    # - commits: Direct indicator of code activity (1.0)
    # - pull_requests: High-value collaboration (2.0)
    # - issues: Community engagement (0.5)
    # - stars: Interest but not active contribution (0.1)
    # - forks: Potential for external development (0.3)
    
    DEFAULT_WEIGHTS = {
        'commits': 1.0,
        'commit': 1.0,
        'pull_requests': 2.0,
        'pulls': 2.0,
        'pr': 2.0,
        'issues': 0.5,
        'issue': 0.5,
        'stars': 0.1,
        'star': 0.1,
        'forks': 0.3,
        'fork': 0.3,
    }
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
        threshold_method: str = 'f1',
        validation_split: float = 0.2,
        min_quarters: int = 2,
        export_csv: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the labeler.
        
        Args:
            input_path: Path to quarterly aggregated data
            output_path: Path where labeled data will be saved
            weights: Custom weights for activity score (dict of metric_name: weight)
            threshold: Manual threshold (if None, will be computed automatically)
            threshold_method: Method to compute threshold ('f1', 'precision', 'recall', 'youden')
            validation_split: Fraction of data to use for threshold tuning
            min_quarters: Minimum quarters of data required for a repo
            export_csv: Also export as CSV
            verbose: Enable verbose logging
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.manual_threshold = threshold
        self.threshold_method = threshold_method
        self.validation_split = validation_split
        self.min_quarters = min_quarters
        self.export_csv = export_csv
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self.data: Optional[pd.DataFrame] = None
        self.threshold: Optional[float] = None
        self.threshold_metrics: Dict = {}
    
    def process(self) -> None:
        """Run the complete labeling pipeline."""
        logger.info("Starting activity labeling pipeline")
        
        # Load data
        self._load_data()
        
        # Compute activity scores
        self._compute_activity_scores()
        
        # Determine threshold
        if self.manual_threshold is not None:
            self.threshold = self.manual_threshold
            logger.info(f"Using manual threshold: {self.threshold}")
        else:
            self._determine_optimal_threshold()
        
        # Apply labels
        self._apply_labels()
        
        # Save results
        self._save_results()
        
        logger.info("Activity labeling complete!")
    
    def _load_data(self) -> None:
        """Load quarterly aggregated data."""
        logger.info(f"Loading data from: {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        if self.input_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.input_path)
        elif self.input_path.suffix == '.csv':
            self.data = pd.read_csv(self.input_path)
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        logger.info(f"Loaded {len(self.data):,} records")
        
        # Filter repositories with insufficient data
        repo_counts = self.data.groupby('repo_id').size()
        valid_repos = repo_counts[repo_counts >= self.min_quarters].index
        
        self.data = self.data[self.data['repo_id'].isin(valid_repos)]
        logger.info(f"After filtering (min {self.min_quarters} quarters): {len(self.data):,} records, "
                   f"{self.data['repo_id'].nunique():,} repos")
    
    def _compute_activity_scores(self) -> None:
        """Compute weighted activity score for each record."""
        logger.info("Computing activity scores...")
        
        # Identify available metric columns
        available_metrics = []
        metric_weights = {}
        
        for col in self.data.columns:
            col_lower = col.lower()
            for metric_key, weight in self.weights.items():
                if metric_key in col_lower and pd.api.types.is_numeric_dtype(self.data[col]):
                    available_metrics.append(col)
                    metric_weights[col] = weight
                    break
        
        logger.info(f"Using {len(available_metrics)} metrics for scoring: {available_metrics}")
        
        if not available_metrics:
            logger.error("No metrics found for activity scoring")
            raise ValueError("Cannot compute activity score: no metrics available")
        
        # Compute weighted sum
        self.data['activity_score'] = 0.0
        
        for col, weight in metric_weights.items():
            # Normalize metric to avoid scale issues (use log1p for skewed distributions)
            normalized = np.log1p(self.data[col].fillna(0))
            self.data['activity_score'] += weight * normalized
        
        logger.info(f"Activity score range: [{self.data['activity_score'].min():.2f}, "
                   f"{self.data['activity_score'].max():.2f}]")
        logger.info(f"Activity score mean: {self.data['activity_score'].mean():.2f}, "
                   f"median: {self.data['activity_score'].median():.2f}")
    
    def _determine_optimal_threshold(self) -> None:
        """
        Determine optimal threshold using validation data.
        
        Uses a pseudo-labeling strategy: assume top X% are active, bottom Y% are inactive,
        and optimize threshold on this validation set.
        """
        logger.info(f"Determining optimal threshold using method: {self.threshold_method}")
        
        # Create pseudo-labels for validation
        # Strategy: Use top 25% as definitely active, bottom 25% as definitely inactive
        scores = self.data['activity_score'].values
        top_percentile = np.percentile(scores, 75)
        bottom_percentile = np.percentile(scores, 25)
        
        # Create validation set
        validation_mask = (scores >= top_percentile) | (scores <= bottom_percentile)
        val_scores = scores[validation_mask]
        val_labels = (scores[validation_mask] >= top_percentile).astype(int)
        
        logger.info(f"Validation set: {len(val_scores):,} samples "
                   f"({val_labels.sum():,} active, {(1-val_labels).sum():,} inactive)")
        
        # Compute precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(val_labels, val_scores)
        
        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(val_labels, val_scores)
        
        # Select threshold based on method
        if self.threshold_method == 'f1':
            # Maximize F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores[:-1])  # Exclude last element
            self.threshold = thresholds_pr[best_idx]
            self.threshold_metrics['f1'] = f1_scores[best_idx]
            self.threshold_metrics['precision'] = precision[best_idx]
            self.threshold_metrics['recall'] = recall[best_idx]
            logger.info(f"Optimal threshold (max F1): {self.threshold:.4f} "
                       f"(F1={f1_scores[best_idx]:.3f}, "
                       f"P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f})")
        
        elif self.threshold_method == 'youden':
            # Maximize Youden's J statistic (sensitivity + specificity - 1)
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            self.threshold = thresholds_roc[best_idx]
            self.threshold_metrics['youden_j'] = youden_j[best_idx]
            self.threshold_metrics['tpr'] = tpr[best_idx]
            self.threshold_metrics['fpr'] = fpr[best_idx]
            logger.info(f"Optimal threshold (Youden's J): {self.threshold:.4f} "
                       f"(J={youden_j[best_idx]:.3f}, TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})")
        
        elif self.threshold_method == 'precision':
            # Target high precision (>=0.9), maximize recall
            high_precision_mask = precision >= 0.9
            if high_precision_mask.any():
                valid_indices = np.where(high_precision_mask)[0]
                best_idx = valid_indices[np.argmax(recall[high_precision_mask])]
                self.threshold = thresholds_pr[best_idx]
            else:
                # Fall back to max precision
                best_idx = np.argmax(precision[:-1])
                self.threshold = thresholds_pr[best_idx]
            
            self.threshold_metrics['precision'] = precision[best_idx]
            self.threshold_metrics['recall'] = recall[best_idx]
            logger.info(f"Optimal threshold (precision-focused): {self.threshold:.4f} "
                       f"(P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f})")
        
        elif self.threshold_method == 'recall':
            # Target high recall (>=0.9), maximize precision
            high_recall_mask = recall >= 0.9
            if high_recall_mask.any():
                valid_indices = np.where(high_recall_mask)[0]
                best_idx = valid_indices[np.argmax(precision[high_recall_mask])]
                self.threshold = thresholds_pr[best_idx]
            else:
                # Fall back to max recall
                best_idx = np.argmax(recall[:-1])
                self.threshold = thresholds_pr[best_idx]
            
            self.threshold_metrics['precision'] = precision[best_idx]
            self.threshold_metrics['recall'] = recall[best_idx]
            logger.info(f"Optimal threshold (recall-focused): {self.threshold:.4f} "
                       f"(P={precision[best_idx]:.3f}, R={recall[best_idx]:.3f})")
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        # Store curves for later analysis
        self.threshold_metrics['precision_curve'] = precision
        self.threshold_metrics['recall_curve'] = recall
        self.threshold_metrics['thresholds_pr'] = thresholds_pr
        self.threshold_metrics['fpr'] = fpr
        self.threshold_metrics['tpr'] = tpr
        self.threshold_metrics['thresholds_roc'] = thresholds_roc
    
    def _apply_labels(self) -> None:
        """Apply activity labels based on threshold."""
        logger.info(f"Applying labels with threshold: {self.threshold:.4f}")
        
        self.data['is_active'] = (self.data['activity_score'] >= self.threshold).astype(int)
        
        # Statistics
        active_count = self.data['is_active'].sum()
        active_pct = (active_count / len(self.data)) * 100
        
        logger.info(f"Labeled {active_count:,} records as active ({active_pct:.1f}%)")
        logger.info(f"Labeled {len(self.data) - active_count:,} records as inactive "
                   f"({100 - active_pct:.1f}%)")
        
        # Per-repo statistics
        repo_activity = self.data.groupby('repo_id')['is_active'].agg(['sum', 'count', 'mean'])
        mostly_active = (repo_activity['mean'] > 0.5).sum()
        
        logger.info(f"Repository-level: {mostly_active:,} repos are mostly active "
                   f"({mostly_active / len(repo_activity) * 100:.1f}%)")
    
    def _save_results(self) -> None:
        """Save labeled data to file."""
        logger.info(f"Saving results to: {self.output_path}")
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        self.data.to_parquet(
            self.output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"✅ Saved Parquet: {self.output_path}")
        
        # Optionally save as CSV
        if self.export_csv:
            csv_path = self.output_path.with_suffix('.csv')
            self.data.to_csv(csv_path, index=False)
            logger.info(f"✅ Saved CSV: {csv_path}")
        
        # Save threshold metadata
        metadata_path = self.output_path.parent / 'labeling_metadata.json'
        import json
        
        metadata = {
            'threshold': float(self.threshold),
            'threshold_method': self.threshold_method,
            'weights': self.weights,
            'total_records': len(self.data),
            'active_count': int(self.data['is_active'].sum()),
            'active_percentage': float(self.data['is_active'].mean() * 100),
            'unique_repos': int(self.data['repo_id'].nunique()),
        }
        
        if self.threshold_metrics:
            # Add metrics (excluding numpy arrays)
            for key, value in self.threshold_metrics.items():
                if not isinstance(value, np.ndarray):
                    metadata[f'threshold_{key}'] = float(value)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved metadata: {metadata_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("LABELING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records: {len(self.data):,}")
        logger.info(f"Active: {metadata['active_count']:,} ({metadata['active_percentage']:.1f}%)")
        logger.info(f"Inactive: {len(self.data) - metadata['active_count']:,} "
                   f"({100 - metadata['active_percentage']:.1f}%)")
        logger.info(f"Threshold: {self.threshold:.4f}")
        logger.info(f"Method: {self.threshold_method}")
        logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Label GitHub repository activity'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/quarters.parquet',
        help='Path to quarterly aggregated data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/quarters_labeled.parquet',
        help='Path where labeled data will be saved'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Manual threshold (if not provided, will be computed automatically)'
    )
    parser.add_argument(
        '--threshold-method',
        type=str,
        choices=['f1', 'precision', 'recall', 'youden'],
        default='f1',
        help='Method to compute optimal threshold'
    )
    parser.add_argument(
        '--min-quarters',
        type=int,
        default=2,
        help='Minimum quarters of data required for a repository'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Also export results as CSV'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Run labeling
    labeler = ActivityLabeler(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        threshold_method=args.threshold_method,
        min_quarters=args.min_quarters,
        export_csv=args.export_csv,
        verbose=args.verbose
    )
    
    labeler.process()
    
    print(f"\n✅ Activity labeling complete! Output: {args.output}")


if __name__ == '__main__':
    main()
