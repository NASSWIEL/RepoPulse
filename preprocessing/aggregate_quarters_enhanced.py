#!/usr/bin/env python3
"""
Enhanced Quarterly Aggregation Script for GitHub Repository Activity

This script aggregates ALL available time-series features from raw GitHub data
into quarterly bins, capturing comprehensive activity signals for forecasting.

Features Extracted:
-----------------
COMMIT ACTIVITY:
  - commit_count: Number of commits
  - commit_authors: Number of unique commit authors
  - commit_comments: Total comments on commits
  
ISSUE ACTIVITY:
  - issue_count: Number of new issues opened
  - issue_closed: Number of issues closed
  - issue_comments: Total comments on issues
  - issue_participants: Number of unique issue participants
  
PULL REQUEST ACTIVITY:
  - pr_count: Number of new PRs opened
  - pr_merged: Number of PRs merged
  - pr_closed_unmerged: Number of PRs closed without merging
  - pr_comments: Total comments on PRs
  - pr_contributors: Number of unique PR contributors
  
STAR ACTIVITY:
  - star_count: Number of new stars received
  - star_users: Number of unique users who starred

FORK ACTIVITY:
  - fork_count: Cumulative fork count at quarter end

DERIVED METRICS:
  - total_contributors: Union of commit authors, issue participants, PR contributors
  - engagement_score: Total comments across all activities
  - pr_acceptance_rate: Merged PRs / Total closed PRs
  - issue_resolution_rate: Closed issues / Total issues

Data Structure:
  /big_data/repositories/owner__repo_name/
    ├── commits.csv (author_date, author_login, comment_count)
    ├── stargazers.csv (starred_at, user_login)
    ├── issues.csv (created_at, closed_at, state, comments, user_login)
    ├── pull_requests.csv (created_at, merged_at, closed_at, state, comments, user_login)
    └── repository.csv (forks_count, updated_at - cumulative metrics)

Usage:
    python aggregate_quarters_enhanced.py --config config/config.yaml
    python aggregate_quarters_enhanced.py --limit 10
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Set
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedQuarterlyAggregator:
    """Aggregates comprehensive repository activity into quarterly bins."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the aggregator."""
        self.config = self._load_config(config_path)
        self.repos_dir = Path(self.config["data"]["raw_data_path"]) / "repositories"
        self.output_path = Path(self.config["data"]["aggregated_data_path"])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def aggregate(self, repo_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate all features into quarterly bins.
        
        Args:
            repo_limit: Optional limit on number of repos to process
        
        Returns:
            DataFrame with comprehensive quarterly aggregated data
        """
        logger.info("Starting enhanced quarterly aggregation...")
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
        errors = 0
        
        for i, repo_dir in enumerate(repo_dirs, 1):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(repo_dirs)} repositories (errors: {errors})...")
            
            try:
                repo_data = self._process_repository(repo_dir)
                if not repo_data.empty:
                    all_quarters.append(repo_data)
            except Exception as e:
                errors += 1
                logger.error(f"Error processing {repo_dir.name}: {e}")
                continue
        
        # Combine all repositories
        if not all_quarters:
            raise ValueError("No data was aggregated from any repository")
        
        aggregated = pd.concat(all_quarters, ignore_index=True)
        logger.info(f"Aggregated {len(aggregated)} quarterly records from {len(repo_dirs)} repositories ({errors} errors)")
        
        # Compute derived metrics
        aggregated = self._compute_derived_metrics(aggregated)
        
        # Save results
        self._save_results(aggregated)
        
        return aggregated
    
    def _process_repository(self, repo_dir: Path) -> pd.DataFrame:
        """
        Process a single repository and extract all quarterly features.
        
        Args:
            repo_dir: Path to repository directory
        
        Returns:
            DataFrame with all quarterly features for this repo
        """
        repo_id = repo_dir.name
        
        # Extract features from each file type
        commits_data = self._process_commits(repo_dir)
        issues_data = self._process_issues(repo_dir)
        prs_data = self._process_pull_requests(repo_dir)
        stars_data = self._process_stargazers(repo_dir)
        forks_data = self._process_repository_info(repo_dir)
        
        # Collect all DataFrames
        dfs = [df for df in [commits_data, issues_data, prs_data, stars_data, forks_data] if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all on (year, quarter)
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.merge(df, on=['year', 'quarter'], how='outer')
        
        # Fill missing values with 0 for count-based metrics
        count_cols = [c for c in combined.columns if c not in ['year', 'quarter']]
        combined[count_cols] = combined[count_cols].fillna(0).astype(int)
        
        # Add repository ID
        combined.insert(0, 'repo_id', repo_id)
        
        # Sort by time
        combined = combined.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        return combined
    
    def _process_commits(self, repo_dir: Path) -> pd.DataFrame:
        """Extract commit-based features."""
        file_path = repo_dir / 'commits.csv'
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp
            df['author_date'] = pd.to_datetime(df['author_date'], errors='coerce')
            df = df.dropna(subset=['author_date'])
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Extract time components
            df['year'] = df['author_date'].dt.year
            df['quarter'] = df['author_date'].dt.quarter
            
            # Aggregate by quarter
            quarterly = df.groupby(['year', 'quarter']).agg(
                commit_count=('author_date', 'size'),
                commit_authors=('author_login', 'nunique'),
                commit_comments=('comment_count', 'sum')
            ).reset_index()
            
            return quarterly
            
        except Exception as e:
            logger.warning(f"Error processing commits in {repo_dir.name}: {e}")
            return pd.DataFrame()
    
    def _process_issues(self, repo_dir: Path) -> pd.DataFrame:
        """Extract issue-based features."""
        file_path = repo_dir / 'issues.csv'
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
            
            # Issues opened (by created_at)
            df_opened = df.dropna(subset=['created_at']).copy()
            df_opened['year'] = df_opened['created_at'].dt.year
            df_opened['quarter'] = df_opened['created_at'].dt.quarter
            
            opened = df_opened.groupby(['year', 'quarter']).agg(
                issue_count=('created_at', 'size'),
                issue_comments=('comments', 'sum'),
                issue_participants=('user_login', 'nunique')
            ).reset_index()
            
            # Issues closed (by closed_at)
            df_closed = df.dropna(subset=['closed_at']).copy()
            df_closed['year'] = df_closed['closed_at'].dt.year
            df_closed['quarter'] = df_closed['closed_at'].dt.quarter
            
            closed = df_closed.groupby(['year', 'quarter']).agg(
                issue_closed=('closed_at', 'size')
            ).reset_index()
            
            # Merge opened and closed
            quarterly = opened.merge(closed, on=['year', 'quarter'], how='outer')
            quarterly = quarterly.fillna(0)
            
            return quarterly
            
        except Exception as e:
            logger.warning(f"Error processing issues in {repo_dir.name}: {e}")
            return pd.DataFrame()
    
    def _process_pull_requests(self, repo_dir: Path) -> pd.DataFrame:
        """Extract pull request features."""
        file_path = repo_dir / 'pull_requests.csv'
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['merged_at'] = pd.to_datetime(df['merged_at'], errors='coerce')
            df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
            
            # PRs opened (by created_at)
            df_opened = df.dropna(subset=['created_at']).copy()
            df_opened['year'] = df_opened['created_at'].dt.year
            df_opened['quarter'] = df_opened['created_at'].dt.quarter
            
            opened = df_opened.groupby(['year', 'quarter']).agg(
                pr_count=('created_at', 'size'),
                pr_comments=('comments', 'sum'),
                pr_contributors=('user_login', 'nunique')
            ).reset_index()
            
            # PRs merged (by merged_at)
            df_merged = df.dropna(subset=['merged_at']).copy()
            df_merged['year'] = df_merged['merged_at'].dt.year
            df_merged['quarter'] = df_merged['merged_at'].dt.quarter
            
            merged = df_merged.groupby(['year', 'quarter']).agg(
                pr_merged=('merged_at', 'size')
            ).reset_index()
            
            # PRs closed without merge (closed_at exists but merged_at is null)
            df_closed_unmerged = df[
                df['closed_at'].notna() & df['merged_at'].isna()
            ].copy()
            
            if len(df_closed_unmerged) > 0:
                df_closed_unmerged['year'] = pd.to_datetime(df_closed_unmerged['closed_at']).dt.year
                df_closed_unmerged['quarter'] = pd.to_datetime(df_closed_unmerged['closed_at']).dt.quarter
                
                closed_unmerged = df_closed_unmerged.groupby(['year', 'quarter']).agg(
                    pr_closed_unmerged=('closed_at', 'size')
                ).reset_index()
            else:
                closed_unmerged = pd.DataFrame(columns=['year', 'quarter', 'pr_closed_unmerged'])
            
            # Merge all PR metrics
            quarterly = opened.merge(merged, on=['year', 'quarter'], how='outer')
            quarterly = quarterly.merge(closed_unmerged, on=['year', 'quarter'], how='outer')
            quarterly = quarterly.fillna(0)
            
            return quarterly
            
        except Exception as e:
            logger.warning(f"Error processing PRs in {repo_dir.name}: {e}")
            return pd.DataFrame()
    
    def _process_stargazers(self, repo_dir: Path) -> pd.DataFrame:
        """Extract stargazer features."""
        file_path = repo_dir / 'stargazers.csv'
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp
            df['starred_at'] = pd.to_datetime(df['starred_at'], errors='coerce')
            df = df.dropna(subset=['starred_at'])
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Extract time components
            df['year'] = df['starred_at'].dt.year
            df['quarter'] = df['starred_at'].dt.quarter
            
            # Aggregate by quarter
            quarterly = df.groupby(['year', 'quarter']).agg(
                star_count=('starred_at', 'size'),
                star_users=('user_login', 'nunique')
            ).reset_index()
            
            return quarterly
            
        except Exception as e:
            logger.warning(f"Error processing stargazers in {repo_dir.name}: {e}")
            return pd.DataFrame()
    
    def _process_repository_info(self, repo_dir: Path) -> pd.DataFrame:
        """Extract cumulative fork count from repository snapshot."""
        file_path = repo_dir / 'repository.csv'
        if not file_path.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            
            # The repository.csv contains a snapshot with forks_count and fetched_at
            if 'forks_count' not in df.columns or 'fetched_at' not in df.columns:
                return pd.DataFrame()
            
            # Convert timestamp
            df['fetched_at'] = pd.to_datetime(df['fetched_at'], errors='coerce')
            df = df.dropna(subset=['fetched_at'])
            
            if len(df) == 0:
                return pd.DataFrame()
            
            # Extract time components
            df['year'] = df['fetched_at'].dt.year
            df['quarter'] = df['fetched_at'].dt.quarter
            
            # For each quarter, use the latest fork_count value
            # (forks_count is cumulative, so we take the last snapshot per quarter)
            quarterly = df.groupby(['year', 'quarter']).agg(
                fork_count=('forks_count', 'last')
            ).reset_index()
            
            return quarterly
            
        except Exception as e:
            logger.warning(f"Error processing repository info in {repo_dir.name}: {e}")
            return pd.DataFrame()
    
    def _compute_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute total_contributors and filter to keep only essential features."""
        
        # Total contributors (max of commit_authors, issue_participants, pr_contributors)
        # Note: This is an approximation since we can't compute true union without user-level data
        contributor_cols = ['commit_authors', 'issue_participants', 'pr_contributors']
        available_cols = [c for c in contributor_cols if c in data.columns]
        
        if available_cols:
            data['total_contributors'] = data[available_cols].max(axis=1)
        else:
            data['total_contributors'] = 0
        
        # Keep only essential features
        essential_features = [
            'repo_id', 'year', 'quarter',
            'commit_count', 'total_contributors', 
            'issue_count', 'issue_closed',
            'pr_count', 'pr_merged', 
            'star_count', 'fork_count'
        ]
        
        # Select only available essential features
        available_features = [f for f in essential_features if f in data.columns]
        data = data[available_features]
        
        return data
    
    def _save_results(self, data: pd.DataFrame) -> None:
        """Save aggregated results with comprehensive metadata."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet
        parquet_path = self.output_path.with_suffix('.parquet')
        data.to_parquet(parquet_path, index=False)
        logger.info(f"Saved aggregated data to {parquet_path}")
        
        # Save as CSV
        csv_path = self.output_path.with_suffix('.csv')
        data.to_csv(csv_path, index=False)
        logger.info(f"Saved aggregated data to {csv_path}")
        
        # Print comprehensive statistics
        logger.info("\n" + "="*70)
        logger.info("AGGREGATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total repositories: {data['repo_id'].nunique()}")
        logger.info(f"Total quarterly records: {len(data)}")
        logger.info(f"Year range: {data['year'].min()} - {data['year'].max()}")
        logger.info(f"Average quarters per repo: {len(data) / data['repo_id'].nunique():.1f}")
        
        logger.info("\n" + "-"*70)
        logger.info("FEATURE STATISTICS (Essential Features Only)")
        logger.info("-"*70)
        
        # Essential features only
        features = {
            'Commit Activity': ['commit_count', 'total_contributors'],
            'Issue Activity': ['issue_count', 'issue_closed'],
            'PR Activity': ['pr_count', 'pr_merged'],
            'Star Activity': ['star_count']
        }
        
        for group_name, feature_list in features.items():
            logger.info(f"\n{group_name}:")
            for feature in feature_list:
                if feature in data.columns:
                    col = data[feature]
                    logger.info(f"  {feature:25s}: mean={col.mean():8.1f}, "
                              f"median={col.median():6.0f}, "
                              f"max={col.max():8.0f}")
        
        logger.info("\n" + "="*70)
        logger.info(f"Feature count: {len([c for c in data.columns if c not in ['repo_id', 'year', 'quarter']])}")
        logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced quarterly aggregation with comprehensive features"
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
    aggregator = EnhancedQuarterlyAggregator(config_path=args.config)
    aggregator.aggregate(repo_limit=args.limit)
    
    logger.info("Enhanced aggregation complete!")


if __name__ == "__main__":
    main()
