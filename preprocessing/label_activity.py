#!/usr/bin/env python3
"""
Label quarterly repository activity as ACTIVE or INACTIVE.

This script analyzes aggregated quarterly data and assigns binary labels based on
a data-driven activity score that combines multiple metrics.

Labeling Strategy:
-----------------
1. Compute weighted activity score per quarter
2. Use percentile-based threshold (target: 20-30% active)
3. Validate distribution and balance
4. Save labeled dataset

Activity Score Formula:
----------------------
The score combines developmental and community engagement metrics:

activity_score = (
    commit_count * 1.0          # Direct development activity
    + total_contributors * 2.0   # Community size (high weight)
    + issue_count * 0.5          # Issue tracking
    + issue_closed * 1.0         # Issue resolution (higher value)
    + pr_count * 0.8             # PR creation
    + pr_merged * 1.5            # PR acceptance (highest value)
    + star_count * 0.2           # Popularity signal
    + fork_count * 0.3           # Fork activity - strong interest indicator
)

Rationale:
- total_contributors: High weight (2.0) because unique contributors indicate
  sustained community engagement
- pr_merged: Highest weight (1.5) because merged PRs represent actual accepted
  contributions (quality signal)
- issue_closed: Higher weight (1.0) than issue_count (0.5) because resolution
  matters more than just opening issues
- star_count: Moderate weight (0.2) to better reflect popularity and
  community interest as activity indicators
- fork_count: Moderate-high weight (0.3) because forks indicate serious intent
  to contribute or extend the project
- commit_count: Base weight (1.0) as fundamental activity measure

Threshold Selection:
-------------------
- Target: 20-30% of quarters labeled as ACTIVE
- Method: Percentile-based (75th-80th percentile)
- Ensures meaningful separation between active and inactive periods
- Avoids trivial labeling (all active or all inactive)

Usage:
    python label_activity.py --config config/config.yaml
    python label_activity.py --threshold 0.75  # Use 75th percentile
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActivityLabeler:
    """Labels quarterly repository activity as active or inactive."""
    
    # Feature weights for activity score
    FEATURE_WEIGHTS = {
        'commit_count': 1.0,
        'total_contributors': 2.0,  # High weight - community engagement
        'issue_count': 0.5,
        'issue_closed': 1.0,        # Higher than issue_count - resolution matters
        'pr_count': 0.8,
        'pr_merged': 1.5,           # Highest weight - accepted contributions
        'star_count': 0.2,          # Moderate weight - popularity and interest
        'fork_count': 0.3           # Moderate-high weight - serious interest
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the labeler."""
        self.config = self._load_config(config_path)
        self.input_path = Path(self.config["data"]["aggregated_data_path"])
        self.output_path = Path(self.config["data"]["labeled_data_path"])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def label(self, percentile_threshold: float = 0.75) -> pd.DataFrame:
        """
        Label quarterly data as active or inactive.
        
        Args:
            percentile_threshold: Percentile to use as threshold (0.75 = top 25%)
        
        Returns:
            DataFrame with activity scores and labels
        """
        logger.info("="*70)
        logger.info("ACTIVITY LABELING")
        logger.info("="*70)
        
        # Load aggregated data
        logger.info(f"Loading aggregated data from {self.input_path}")
        data = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(data)} quarterly records from {data['repo_id'].nunique()} repositories")
        
        # Compute activity scores
        logger.info("\nComputing weighted activity scores...")
        data = self._compute_activity_score(data)
        
        # Analyze score distribution
        self._analyze_distribution(data)
        
        # Apply threshold and label
        logger.info(f"\nApplying threshold at {percentile_threshold*100:.0f}th percentile...")
        data, threshold_value = self._apply_threshold(data, percentile_threshold)
        
        # Validate labeling
        self._validate_labels(data, threshold_value)
        
        # Save results
        self._save_labeled_data(data)
        
        return data
    
    def _compute_activity_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted activity score for each quarter."""
        
        # Initialize score
        data['activity_score'] = 0.0
        
        # Apply weighted sum
        for feature, weight in self.FEATURE_WEIGHTS.items():
            if feature in data.columns:
                data['activity_score'] += data[feature] * weight
            else:
                logger.warning(f"Feature {feature} not found in data")
        
        logger.info(f"Activity score computed using {len(self.FEATURE_WEIGHTS)} features")
        logger.info("\nFeature weights:")
        for feature, weight in self.FEATURE_WEIGHTS.items():
            logger.info(f"  {feature:25s}: {weight:5.2f}")
        
        return data
    
    def _analyze_distribution(self, data: pd.DataFrame) -> None:
        """Analyze and log activity score distribution."""
        
        logger.info("\n" + "-"*70)
        logger.info("ACTIVITY SCORE DISTRIBUTION")
        logger.info("-"*70)
        
        score = data['activity_score']
        
        logger.info(f"Mean:       {score.mean():12,.2f}")
        logger.info(f"Median:     {score.median():12,.2f}")
        logger.info(f"Std Dev:    {score.std():12,.2f}")
        logger.info(f"Min:        {score.min():12,.2f}")
        logger.info(f"Max:        {score.max():12,.2f}")
        
        logger.info("\nPercentiles:")
        for p in [25, 50, 60, 70, 75, 80, 85, 90, 95]:
            logger.info(f"  {p:2d}%: {score.quantile(p/100):12,.2f}")
        
        logger.info(f"\nZero scores: {(score == 0).sum()} ({(score == 0).sum()/len(score)*100:.1f}%)")
    
    def _apply_threshold(
        self, 
        data: pd.DataFrame, 
        percentile: float
    ) -> Tuple[pd.DataFrame, float]:
        """
        Apply percentile-based threshold to label quarters.
        
        Args:
            data: DataFrame with activity_score column
            percentile: Percentile threshold (e.g., 0.75 for top 25%)
        
        Returns:
            Tuple of (labeled_data, threshold_value)
        """
        
        # Compute threshold value
        threshold = data['activity_score'].quantile(percentile)
        
        # Apply binary label
        data['is_active'] = (data['activity_score'] >= threshold).astype(int)
        
        return data, threshold
    
    def _validate_labels(self, data: pd.DataFrame, threshold: float) -> None:
        """Validate and report on the labeling results."""
        
        logger.info("\n" + "-"*70)
        logger.info("LABELING VALIDATION")
        logger.info("-"*70)
        
        logger.info(f"\nThreshold value: {threshold:,.2f}")
        
        # Label distribution
        active_count = (data['is_active'] == 1).sum()
        inactive_count = (data['is_active'] == 0).sum()
        active_pct = active_count / len(data) * 100
        inactive_pct = inactive_count / len(data) * 100
        
        logger.info(f"\nLabel distribution:")
        logger.info(f"  ACTIVE:   {active_count:6d} quarters ({active_pct:5.1f}%)")
        logger.info(f"  INACTIVE: {inactive_count:6d} quarters ({inactive_pct:5.1f}%)")
        
        # Check if within target range (20-30%)
        if 20 <= active_pct <= 30:
            logger.info(f"\nActive proportion ({active_pct:.1f}%) is within target range (20-30%)")
        else:
            logger.warning(f"\n⚠ Active proportion ({active_pct:.1f}%) is outside target range (20-30%)")
        
        # Statistics by label
        logger.info("\nActivity score by label:")
        for label in [0, 1]:
            label_name = "INACTIVE" if label == 0 else "ACTIVE"
            subset = data[data['is_active'] == label]['activity_score']
            logger.info(f"  {label_name:8s}: mean={subset.mean():10,.2f}, "
                       f"median={subset.median():10,.2f}, "
                       f"range=[{subset.min():,.0f}, {subset.max():,.0f}]")
        
        # Per-repository distribution
        repo_label_dist = data.groupby('repo_id')['is_active'].agg(['sum', 'count'])
        repo_label_dist['pct_active'] = repo_label_dist['sum'] / repo_label_dist['count'] * 100
        
        logger.info("\nPer-repository active percentage:")
        logger.info(f"  Mean:   {repo_label_dist['pct_active'].mean():5.1f}%")
        logger.info(f"  Median: {repo_label_dist['pct_active'].median():5.1f}%")
        logger.info(f"  Min:    {repo_label_dist['pct_active'].min():5.1f}%")
        logger.info(f"  Max:    {repo_label_dist['pct_active'].max():5.1f}%")
        
        # Check for trivial cases
        all_active = (repo_label_dist['pct_active'] == 100).sum()
        all_inactive = (repo_label_dist['pct_active'] == 0).sum()
        
        logger.info(f"\nRepositories with all quarters active:   {all_active}")
        logger.info(f"Repositories with all quarters inactive: {all_inactive}")
    
    def _save_labeled_data(self, data: pd.DataFrame) -> None:
        """Save labeled dataset."""
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        parquet_path = self.output_path.with_suffix('.parquet')
        data.to_parquet(parquet_path, index=False)
        logger.info(f"\nSaved labeled data to {parquet_path}")
        
        # Save as CSV
        csv_path = self.output_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)
        logger.info(f"Saved labeled data to {csv_path}")
        
        # Save summary statistics
        summary_path = self.output_path.parent / "labeling_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("ACTIVITY LABELING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total quarters: {len(data)}\n")
            f.write(f"Active quarters: {(data['is_active']==1).sum()} ({(data['is_active']==1).sum()/len(data)*100:.1f}%)\n")
            f.write(f"Inactive quarters: {(data['is_active']==0).sum()} ({(data['is_active']==0).sum()/len(data)*100:.1f}%)\n")
            f.write(f"\nThreshold: {data['activity_score'].quantile(0.75):,.2f} (75th percentile)\n")
            f.write(f"\nFeature weights:\n")
            for feature, weight in self.FEATURE_WEIGHTS.items():
                f.write(f"  {feature:25s}: {weight:5.2f}\n")
        
        logger.info(f"Saved summary to {summary_path}")
        
        logger.info("\n" + "="*70)
        logger.info("LABELING COMPLETE")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Label quarterly repository activity as active or inactive"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.75,
        help='Percentile threshold (0.75 = top 25%% active, 0.80 = top 20%% active)'
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.5 <= args.threshold <= 0.95:
        raise ValueError("Threshold must be between 0.5 and 0.95")
    
    # Run labeling
    labeler = ActivityLabeler(config_path=args.config)
    labeler.label(percentile_threshold=args.threshold)
    
    logger.info("\nLabeling complete!")


if __name__ == "__main__":
    main()
