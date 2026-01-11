#!/usr/bin/env python3
"""
Quarterly Aggregation Script for GitHub Repository Activity

This script:
- Reads raw GitHub repository data
- Normalizes timestamps to UTC
- Bins data into quarterly intervals (Q1-Q4)
- Aggregates metrics per repository per quarter
- Handles missing quarters with configurable imputation
- Outputs processed data in Parquet/CSV format

Usage:
    python aggregate_quarters.py --input /path/to/raw --output data/processed/quarters.parquet
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuarterlyAggregator:
    """Aggregate GitHub repository data into quarterly bins."""
    
    # Metric aggregation strategies
    CUMULATIVE_METRICS = ['stars', 'forks', 'watchers', 'subscribers', 'size']
    EVENT_METRICS = ['commits', 'commit', 'pull_requests', 'pulls', 'issues', 'contributors']
    RATE_METRICS = []  # Can add if needed
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        imputation_strategy: str = 'zero',
        export_csv: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the aggregator.
        
        Args:
            input_path: Path to directory with raw data files
            output_path: Path where processed data will be saved
            start_date: Start date for filtering (YYYY-MM-DD)
            end_date: End date for filtering (YYYY-MM-DD)
            imputation_strategy: How to handle missing quarters ('zero', 'forward_fill', 'interpolate')
            export_csv: Also export as CSV alongside Parquet
            verbose: Enable verbose logging
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.imputation_strategy = imputation_strategy
        self.export_csv = export_csv
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.quarterly_data: Optional[pd.DataFrame] = None
    
    def process(self) -> None:
        """Run the complete aggregation pipeline."""
        logger.info("Starting quarterly aggregation pipeline")
        
        # Load raw data
        self.raw_data = self._load_raw_data()
        
        if self.raw_data is None or len(self.raw_data) == 0:
            logger.error("No data loaded. Exiting.")
            return
        
        logger.info(f"Loaded {len(self.raw_data):,} raw records")
        
        # Normalize timestamps
        self._normalize_timestamps()
        
        # Filter by date range
        if self.start_date or self.end_date:
            self._filter_date_range()
        
        # Create quarterly bins
        self._create_quarterly_bins()
        
        # Aggregate by repository and quarter
        self._aggregate_metrics()
        
        # Handle missing quarters
        self._handle_missing_quarters()
        
        # Add derived features
        self._add_derived_features()
        
        # Save results
        self._save_results()
        
        logger.info("Quarterly aggregation complete!")
    
    def _load_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Load and combine raw data files.
        
        Returns:
            Combined DataFrame of all raw data
        """
        logger.info(f"Loading raw data from: {self.input_path}")
        
        if not self.input_path.exists():
            logger.error(f"Input path does not exist: {self.input_path}")
            return None
        
        dataframes = []
        
        # Scan for files
        file_patterns = ['*.csv', '*.json', '*.parquet', '*.csv.gz', '*.json.gz', '*.jsonl']
        
        for pattern in file_patterns:
            for file_path in self.input_path.rglob(pattern):
                try:
                    logger.info(f"Reading: {file_path.name}")
                    df = self._read_file(file_path)
                    
                    if df is not None and not df.empty:
                        dataframes.append(df)
                        logger.debug(f"  → Loaded {len(df):,} rows, {len(df.columns)} columns")
                    
                except Exception as e:
                    logger.warning(f"Could not read {file_path.name}: {e}")
                    continue
        
        if not dataframes:
            logger.error("No data files could be loaded")
            return None
        
        # Combine all dataframes
        logger.info(f"Combining {len(dataframes)} data files...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logger.info(f"Combined dataset: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def _read_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read a single data file.
        
        Args:
            file_path: Path to file
            
        Returns:
            DataFrame or None
        """
        try:
            if file_path.suffix == '.parquet':
                return pd.read_parquet(file_path)
            
            elif '.csv' in file_path.suffixes:
                return pd.read_csv(file_path, low_memory=False)
            
            elif '.json' in file_path.suffixes or '.jsonl' in file_path.suffixes:
                # Try JSON lines first
                try:
                    return pd.read_json(file_path, lines=True)
                except:
                    return pd.read_json(file_path)
            
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None
    
    def _normalize_timestamps(self) -> None:
        """
        Normalize timestamp columns to UTC datetime.
        """
        logger.info("Normalizing timestamps...")
        
        # Identify time columns
        time_columns = [col for col in self.raw_data.columns 
                       if any(kw in col.lower() for kw in 
                             ['time', 'date', 'timestamp', 'created', 'updated', 'at'])]
        
        if not time_columns:
            logger.warning("No timestamp columns found. Looking for datetime types...")
            time_columns = self.raw_data.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not time_columns:
            logger.error("Cannot proceed without timestamp column")
            raise ValueError("No timestamp columns found in data")
        
        logger.info(f"Time columns found: {time_columns}")
        
        # Select primary timestamp (prefer created_at or similar)
        primary_time_col = None
        for col in ['created_at', 'timestamp', 'date', 'time'] + time_columns:
            if col in self.raw_data.columns:
                primary_time_col = col
                break
        
        if primary_time_col is None:
            primary_time_col = time_columns[0]
        
        logger.info(f"Using '{primary_time_col}' as primary timestamp")
        
        # Convert to datetime
        self.raw_data['timestamp'] = pd.to_datetime(
            self.raw_data[primary_time_col],
            utc=True,
            errors='coerce'
        )
        
        # Remove rows with invalid timestamps
        invalid_count = self.raw_data['timestamp'].isna().sum()
        if invalid_count > 0:
            logger.warning(f"Removing {invalid_count:,} rows with invalid timestamps")
            self.raw_data = self.raw_data.dropna(subset=['timestamp'])
        
        logger.info(f"Timestamp range: {self.raw_data['timestamp'].min()} to {self.raw_data['timestamp'].max()}")
    
    def _filter_date_range(self) -> None:
        """Filter data by specified date range."""
        logger.info(f"Filtering date range: {self.start_date} to {self.end_date}")
        
        if self.start_date:
            before_count = len(self.raw_data)
            self.raw_data = self.raw_data[self.raw_data['timestamp'] >= self.start_date]
            logger.info(f"  Removed {before_count - len(self.raw_data):,} rows before {self.start_date}")
        
        if self.end_date:
            before_count = len(self.raw_data)
            self.raw_data = self.raw_data[self.raw_data['timestamp'] <= self.end_date]
            logger.info(f"  Removed {before_count - len(self.raw_data):,} rows after {self.end_date}")
        
        logger.info(f"Remaining records: {len(self.raw_data):,}")
    
    def _create_quarterly_bins(self) -> None:
        """Create quarterly time bins."""
        logger.info("Creating quarterly bins...")
        
        # Extract quarter information
        self.raw_data['year'] = self.raw_data['timestamp'].dt.year
        self.raw_data['quarter'] = self.raw_data['timestamp'].dt.quarter
        
        # Create quarter start/end dates
        self.raw_data['quarter_start'] = pd.to_datetime(
            self.raw_data['year'].astype(str) + '-' + 
            ((self.raw_data['quarter'] - 1) * 3 + 1).astype(str) + '-01'
        )
        
        # Quarter end: last day of quarter
        self.raw_data['quarter_end'] = (
            self.raw_data['quarter_start'] + pd.offsets.QuarterEnd(0)
        )
        
        # Create a unique quarter identifier
        self.raw_data['quarter_id'] = (
            self.raw_data['year'].astype(str) + 'Q' + 
            self.raw_data['quarter'].astype(str)
        )
        
        logger.info(f"Created bins for {self.raw_data['quarter_id'].nunique()} unique quarters")
    
    def _identify_repo_column(self) -> str:
        """Identify the repository identifier column."""
        repo_candidates = []
        
        for col in self.raw_data.columns:
            col_lower = col.lower()
            if 'repo' in col_lower or 'repository' in col_lower:
                if 'id' in col_lower:
                    return col  # Prefer ID
                repo_candidates.append(col)
        
        if repo_candidates:
            return repo_candidates[0]
        
        # Fallback to any ID or name column
        for col in self.raw_data.columns:
            if col.lower() in ['id', 'name', 'project']:
                return col
        
        raise ValueError("Cannot identify repository column. Please ensure data has repo identifier.")
    
    def _identify_metric_columns(self) -> Dict[str, List[str]]:
        """
        Identify metric columns and their aggregation strategy.
        
        Returns:
            Dictionary mapping aggregation strategy to column names
        """
        columns = self.raw_data.columns.tolist()
        
        metrics = {
            'cumulative': [],
            'event': [],
            'rate': []
        }
        
        # Check each column
        for col in columns:
            col_lower = col.lower()
            
            # Skip non-numeric and non-metric columns
            if col in ['timestamp', 'year', 'quarter', 'quarter_start', 'quarter_end', 'quarter_id']:
                continue
            
            if not pd.api.types.is_numeric_dtype(self.raw_data[col]):
                continue
            
            # Classify metric
            if any(kw in col_lower for kw in ['star', 'fork', 'watch', 'subscriber', 'size']):
                metrics['cumulative'].append(col)
            elif any(kw in col_lower for kw in ['commit', 'pull', 'issue', 'contributor', 'pr']):
                metrics['event'].append(col)
            else:
                # Default to event for counting
                metrics['event'].append(col)
        
        return metrics
    
    def _aggregate_metrics(self) -> None:
        """Aggregate metrics by repository and quarter."""
        logger.info("Aggregating metrics by repository and quarter...")
        
        # Identify repository column
        repo_col = self._identify_repo_column()
        logger.info(f"Using '{repo_col}' as repository identifier")
        
        # Standardize repo column name
        self.raw_data['repo_id'] = self.raw_data[repo_col]
        
        # Identify metric columns
        metrics = self._identify_metric_columns()
        logger.info(f"Metrics - Cumulative: {len(metrics['cumulative'])}, "
                   f"Event: {len(metrics['event'])}, Rate: {len(metrics['rate'])}")
        
        # Build aggregation dict
        agg_dict = {}
        
        # Cumulative metrics: take last value in quarter
        for col in metrics['cumulative']:
            agg_dict[col] = 'last'
        
        # Event metrics: sum all events in quarter
        for col in metrics['event']:
            agg_dict[col] = 'sum'
        
        # Rate metrics: average over quarter
        for col in metrics['rate']:
            agg_dict[col] = 'mean'
        
        if not agg_dict:
            logger.warning("No metrics identified for aggregation")
            agg_dict = {'timestamp': 'count'}  # At least count events
        
        # Group and aggregate
        groupby_cols = ['repo_id', 'year', 'quarter', 'quarter_start', 'quarter_end', 'quarter_id']
        
        self.quarterly_data = self.raw_data.groupby(groupby_cols, as_index=False).agg(agg_dict)
        
        logger.info(f"Aggregated to {len(self.quarterly_data):,} repository-quarter records")
        logger.info(f"Unique repositories: {self.quarterly_data['repo_id'].nunique():,}")
    
    def _handle_missing_quarters(self) -> None:
        """
        Handle missing quarters for each repository.
        
        Imputation strategies:
        - 'zero': Fill with zeros
        - 'forward_fill': Carry forward last known value
        - 'interpolate': Linear interpolation
        """
        logger.info(f"Handling missing quarters with strategy: {self.imputation_strategy}")
        
        if self.imputation_strategy == 'none':
            logger.info("Skipping imputation (strategy='none')")
            return
        
        # Get all unique repos and quarters
        all_repos = self.quarterly_data['repo_id'].unique()
        all_quarters = self.quarterly_data[['year', 'quarter', 'quarter_start', 'quarter_end', 'quarter_id']].drop_duplicates()
        all_quarters = all_quarters.sort_values(['year', 'quarter'])
        
        logger.info(f"Filling missing quarters for {len(all_repos):,} repos across {len(all_quarters)} quarters")
        
        # Create complete index
        complete_index = pd.MultiIndex.from_product(
            [all_repos, all_quarters['quarter_id'].values],
            names=['repo_id', 'quarter_id']
        )
        
        # Set index and reindex
        indexed_data = self.quarterly_data.set_index(['repo_id', 'quarter_id'])
        complete_data = indexed_data.reindex(complete_index)
        
        # Fill in quarter metadata
        quarter_metadata = all_quarters.set_index('quarter_id')
        for col in ['year', 'quarter', 'quarter_start', 'quarter_end']:
            complete_data[col] = complete_data.index.get_level_values('quarter_id').map(
                quarter_metadata[col]
            )
        
        # Apply imputation strategy
        metric_cols = [col for col in complete_data.columns 
                      if col not in ['year', 'quarter', 'quarter_start', 'quarter_end']]
        
        if self.imputation_strategy == 'zero':
            complete_data[metric_cols] = complete_data[metric_cols].fillna(0)
        
        elif self.imputation_strategy == 'forward_fill':
            complete_data[metric_cols] = complete_data.groupby('repo_id')[metric_cols].ffill()
            complete_data[metric_cols] = complete_data[metric_cols].fillna(0)  # Fill any remaining
        
        elif self.imputation_strategy == 'interpolate':
            complete_data[metric_cols] = complete_data.groupby('repo_id')[metric_cols].apply(
                lambda x: x.interpolate(method='linear', limit_direction='both')
            )
            complete_data[metric_cols] = complete_data[metric_cols].fillna(0)
        
        # Reset index
        self.quarterly_data = complete_data.reset_index()
        
        logger.info(f"After imputation: {len(self.quarterly_data):,} records")
    
    def _add_derived_features(self) -> None:
        """Add derived features like quarters_since_creation."""
        logger.info("Adding derived features...")
        
        # Sort by repo and quarter
        self.quarterly_data = self.quarterly_data.sort_values(['repo_id', 'year', 'quarter'])
        
        # Quarters since first appearance (proxy for creation)
        self.quarterly_data['quarter_index'] = self.quarterly_data.groupby('repo_id').cumcount()
        self.quarterly_data['quarters_since_creation'] = self.quarterly_data['quarter_index']
        
        # Add total quarters column for context
        repo_quarter_counts = self.quarterly_data.groupby('repo_id').size()
        self.quarterly_data['total_quarters'] = self.quarterly_data['repo_id'].map(repo_quarter_counts)
        
        logger.info("Added: quarter_index, quarters_since_creation, total_quarters")
    
    def _save_results(self) -> None:
        """Save processed data to file."""
        logger.info(f"Saving results to: {self.output_path}")
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        self.quarterly_data.to_parquet(
            self.output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        logger.info(f"✅ Saved Parquet: {self.output_path}")
        
        # Optionally save as CSV
        if self.export_csv:
            csv_path = self.output_path.with_suffix('.csv')
            self.quarterly_data.to_csv(csv_path, index=False)
            logger.info(f"✅ Saved CSV: {csv_path}")
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        logger.info(f"Total records: {len(self.quarterly_data):,}")
        logger.info(f"Unique repositories: {self.quarterly_data['repo_id'].nunique():,}")
        logger.info(f"Quarter range: {self.quarterly_data['quarter_id'].min()} to {self.quarterly_data['quarter_id'].max()}")
        logger.info(f"Date range: {self.quarterly_data['quarter_start'].min()} to {self.quarterly_data['quarter_end'].max()}")
        
        # Metric columns
        metric_cols = [col for col in self.quarterly_data.columns 
                      if pd.api.types.is_numeric_dtype(self.quarterly_data[col]) 
                      and col not in ['year', 'quarter', 'quarter_index', 'quarters_since_creation', 'total_quarters']]
        
        if metric_cols:
            logger.info(f"\nMetric columns ({len(metric_cols)}):")
            for col in metric_cols[:10]:  # Show first 10
                mean_val = self.quarterly_data[col].mean()
                logger.info(f"  {col}: mean = {mean_val:.2f}")
        
        logger.info("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Aggregate GitHub repository data into quarterly bins'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/info/raid-etu/m2/s2308975/big_data',
        help='Path to directory containing raw data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/quarters.parquet',
        help='Path where processed data will be saved'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for filtering (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for filtering (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--imputation',
        type=str,
        choices=['zero', 'forward_fill', 'interpolate', 'none'],
        default='zero',
        help='Strategy for handling missing quarters'
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
    
    # Run aggregation
    aggregator = QuarterlyAggregator(
        input_path=args.input,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        imputation_strategy=args.imputation,
        export_csv=args.export_csv,
        verbose=args.verbose
    )
    
    aggregator.process()
    
    print(f"\n✅ Quarterly aggregation complete! Output: {args.output}")


if __name__ == '__main__':
    main()
