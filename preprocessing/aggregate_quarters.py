#!/usr/bin/env python3
"""
Aggregate GitHub repository data into quarterly bins.

This script takes raw event-level data from the repositories directory structure
and aggregates it into 3-month (quarterly) time bins, computing metrics like 
commit counts, issue counts, PR counts, and star counts for each repository 
in each quarter.

Data Structure:
  /big_data/
    repositories/
      owner__repo_name/
        commits.csv (author_date)
        stargazers.csv (starred_at)
        issues.csv (created_at)
        pull_requests.csv (created_at)

Usage:
    python aggregate_quarters.py --config config/config.yaml
    python aggregate_quarters.py --config config/config.yaml --limit 10
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuarterlyAggregator:
    """Aggregates repository data into quarterly time bins."""
    
    # File-to-timestamp column mapping
    TIMESTAMP_MAPPING = {
        'commits.csv': 'author_date',
        'stargazers.csv': 'starred_at',
        'issues.csv': 'created_at',
        'pull_requests.csv': 'created_at'
    }
    
    # Metric name mapping
    METRIC_MAPPING = {
        'commits.csv': 'commit_count',
        'stargazers.csv': 'star_count',
        'issues.csv': 'issue_count',
        'pull_requests.csv': 'pr_count'
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the aggregator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        # Point to repositories directory
        self.repos_dir = Path(self.config["data"]["raw_data_path"]) / "repositories"
        self.output_path = Path(self.config["data"]["aggregated_data_path"])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def aggregate(self, repo_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate data into quarterly bins.
        
        Args:
            repo_limit: Optional limit on number of repos to process (for testing)
        
        Returns:
            DataFrame with quarterly aggregated data
        """
        logger.info("Starting quarterly aggregation...")
        logger.info(f"Scanning repositories in {self.repos_dir}")
        
        if not self.repos_dir.exists():
            raise FileNotFoundError(f"Repository directory not found: {self.repos_dir}")
        
        # Get all repository directories
        repo_dirs = sorted([d for d in self.repos_dir.iterdir() if d.is_dir()])
        
        if repo_limit:
            repo_dirs = repo_dirs[:repo_limit]
            logger.info(f"Limited to {repo_limit} repositories for testing")
        
        logger.info(f"Found {len(repo_dirs)} repositories")
        
        # Process each repository
        all_quarters = []
        for i, repo_dir in enumerate(repo_dirs, 1):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(repo_dirs)} repositories...")
            
            try:
                repo_data = self._process_repository(repo_dir)
                if not repo_data.empty:
                    all_quarters.append(repo_data)
            except Exception as e:
                logger.error(f"Error processing {repo_dir.name}: {e}")
                continue
        
        # Combine all repositories
        if not all_quarters:
            raise ValueError("No data was aggregated from any repository")
        
        aggregated = pd.concat(all_quarters, ignore_index=True)
        logger.info(f"Aggregated to {len(aggregated)} quarterly records across {len(repo_dirs)} repositories")
        
        # Save results
        self._save_results(aggregated)
        
        return aggregated
    
    def _process_repository(self, repo_dir: Path) -> pd.DataFrame:
        """
        Process a single repository and aggregate its quarterly data.
        
        Args:
            repo_dir: Path to repository directory
        
        Returns:
            DataFrame with quarterly aggregated metrics for this repo
        """
        repo_id = repo_dir.name  # Format: owner__repo_name
        
        # Process each file type
        quarterly_data = {}
        
        for filename, timestamp_col in self.TIMESTAMP_MAPPING.items():
            file_path = repo_dir / filename
            
            if not file_path.exists():
                continue
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)
                
                # Check if timestamp column exists
                if timestamp_col not in df.columns:
                    logger.warning(f"{repo_id}/{filename}: Missing {timestamp_col} column")
                    continue
                
                # Convert to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                df = df.dropna(subset=[timestamp_col])
                
                if len(df) == 0:
                    continue
                
                # Extract year and quarter
                df['year'] = df[timestamp_col].dt.year
                df['quarter'] = df[timestamp_col].dt.quarter
                
                # Count events per quarter
                quarter_counts = df.groupby(['year', 'quarter']).size().reset_index(name='count')
                
                # Store with metric name
                metric_name = self.METRIC_MAPPING[filename]
                quarterly_data[metric_name] = quarter_counts
                
            except Exception as e:
                logger.warning(f"Error processing {repo_id}/{filename}: {e}")
                continue
        
        # Combine all metrics for this repo
        if not quarterly_data:
            return pd.DataFrame()
        
        # Start with first metric
        first_metric = list(quarterly_data.keys())[0]
        combined = quarterly_data[first_metric].copy()
        combined.rename(columns={'count': first_metric}, inplace=True)
        
        # Merge other metrics
        for metric_name in list(quarterly_data.keys())[1:]:
            metric_df = quarterly_data[metric_name].copy()
            metric_df.rename(columns={'count': metric_name}, inplace=True)
            combined = combined.merge(
                metric_df,
                on=['year', 'quarter'],
                how='outer'
            )
        
        # Fill missing values with 0 (no events in that quarter)
        for metric in self.METRIC_MAPPING.values():
            if metric in combined.columns:
                combined[metric] = combined[metric].fillna(0).astype(int)
            else:
                combined[metric] = 0
        
        # Add repository ID
        combined.insert(0, 'repo_id', repo_id)
        
        # Sort by year and quarter
        combined = combined.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        return combined
    
    def _save_results(self, data: pd.DataFrame) -> None:
        """Save aggregated results."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet (more efficient)
        parquet_path = self.output_path.with_suffix('.parquet')
        data.to_parquet(parquet_path, index=False)
        logger.info(f"Saved aggregated data to {parquet_path}")
        
        # Also save as CSV for easy inspection
        csv_path = self.output_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregated data to {csv_path}")
        
        # Print summary statistics
        logger.info("\nAggregation Summary:")
        logger.info(f"  Total repositories: {data['repo_id'].nunique()}")
        logger.info(f"  Total quarters: {len(data)}")
        logger.info(f"  Year range: {data['year'].min()} - {data['year'].max()}")
        logger.info(f"  Quarters per repo (avg): {len(data) / data['repo_id'].nunique():.1f}")
        logger.info("\nMetric statistics:")
        for metric in ['commit_count', 'star_count', 'issue_count', 'pr_count']:
            if metric in data.columns:
                logger.info(f"  {metric}: mean={data[metric].mean():.2f}, median={data[metric].median():.0f}, max={data[metric].max()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate repository data into quarterly bins"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of repositories to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Run aggregation
    aggregator = QuarterlyAggregator(config_path=args.config)
    aggregator.aggregate(repo_limit=args.limit)
    
    logger.info("Aggregation complete!")


if __name__ == "__main__":
    main()
