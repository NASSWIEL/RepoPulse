#!/usr/bin/env python3
"""
Data Inspection Script for GitHub Repository Activity Pipeline

This script scans a directory containing raw GitHub repository data and:
- Identifies file formats (CSV, JSON, parquet, etc.)
- Infers schemas and data types
- Extracts sample rows
- Detects potential data quality issues
- Generates a comprehensive report

Usage:
    python inspect_data.py --input /path/to/data --output report.md
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import gzip
import zipfile

import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataInspector:
    """Inspect and analyze raw data files."""
    
    def __init__(self, data_path: str, output_path: str, sample_size: int = 5):
        """
        Initialize the data inspector.
        
        Args:
            data_path: Path to directory containing raw data
            output_path: Path where the report will be written
            sample_size: Number of sample rows to include per file
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.sample_size = sample_size
        self.report_sections = []
        
    def inspect(self) -> None:
        """Run the full inspection pipeline."""
        logger.info(f"Starting inspection of data in: {self.data_path}")
        
        if not self.data_path.exists():
            logger.error(f"Data path does not exist: {self.data_path}")
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Add header
        self.report_sections.append("# GitHub Repository Data Inspection Report\n")
        self.report_sections.append(f"**Data Source:** `{self.data_path}`\n")
        self.report_sections.append(f"**Generated:** {pd.Timestamp.now()}\n\n")
        
        # Scan directory
        files = self._scan_directory()
        
        if not files:
            logger.warning("No data files found in the specified directory")
            self.report_sections.append("## ⚠️ Warning\n\nNo data files found in the specified directory.\n\n")
        else:
            logger.info(f"Found {len(files)} files to inspect")
            
            # Inspect each file
            for file_path in files:
                self._inspect_file(file_path)
        
        # Write report
        self._write_report()
        logger.info(f"Inspection complete. Report written to: {self.output_path}")
    
    def _scan_directory(self) -> List[Path]:
        """
        Scan directory for data files.
        
        Returns:
            List of file paths to inspect
        """
        supported_extensions = {
            '.csv', '.json', '.parquet', '.txt',
            '.csv.gz', '.json.gz', '.jsonl', '.jsonl.gz',
            '.tsv', '.tsv.gz', '.zip'
        }
        
        files = []
        for root, dirs, filenames in os.walk(self.data_path):
            for filename in filenames:
                file_path = Path(root) / filename
                
                # Check if file has supported extension
                if any(str(file_path).endswith(ext) for ext in supported_extensions):
                    files.append(file_path)
        
        return sorted(files)
    
    def _inspect_file(self, file_path: Path) -> None:
        """
        Inspect a single file and add results to report.
        
        Args:
            file_path: Path to file to inspect
        """
        logger.info(f"Inspecting: {file_path.name}")
        
        self.report_sections.append(f"## File: `{file_path.relative_to(self.data_path)}`\n\n")
        
        try:
            # Determine file type and read data
            df, file_info = self._read_file(file_path)
            
            if df is None:
                self.report_sections.append("**Status:** Could not read file\n\n")
                return
            
            # Add basic file info
            self.report_sections.append(f"- **Format:** {file_info['format']}\n")
            self.report_sections.append(f"- **Size:** {file_info['size_mb']:.2f} MB\n")
            self.report_sections.append(f"- **Rows:** {len(df):,}\n")
            self.report_sections.append(f"- **Columns:** {len(df.columns)}\n\n")
            
            # Schema analysis
            self._analyze_schema(df)
            
            # Identify time/repo/metric fields
            self._identify_key_fields(df)
            
            # Sample rows
            self._add_sample_rows(df)
            
            # Data quality checks
            self._check_data_quality(df)
            
            self.report_sections.append("\n---\n\n")
            
        except Exception as e:
            logger.error(f"Error inspecting {file_path.name}: {e}")
            self.report_sections.append(f"**Error:** {str(e)}\n\n")
    
    def _read_file(self, file_path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Read file based on extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (DataFrame, file_info_dict)
        """
        file_info = {
            'format': 'unknown',
            'size_mb': file_path.stat().st_size / (1024 * 1024)
        }
        
        try:
            if file_path.suffix == '.parquet':
                file_info['format'] = 'Parquet'
                df = pd.read_parquet(file_path)
                
            elif file_path.suffix == '.csv' or file_path.name.endswith('.csv.gz'):
                file_info['format'] = 'CSV' + (' (gzipped)' if file_path.name.endswith('.gz') else '')
                df = pd.read_csv(file_path, nrows=10000)  # Limit rows for inspection
                
            elif file_path.suffix == '.tsv' or file_path.name.endswith('.tsv.gz'):
                file_info['format'] = 'TSV' + (' (gzipped)' if file_path.name.endswith('.gz') else '')
                df = pd.read_csv(file_path, sep='\t', nrows=10000)
                
            elif file_path.suffix == '.json' or file_path.name.endswith('.json.gz'):
                file_info['format'] = 'JSON' + (' (gzipped)' if file_path.name.endswith('.gz') else '')
                # Try reading as JSON lines first
                try:
                    df = pd.read_json(file_path, lines=True, nrows=10000)
                except:
                    df = pd.read_json(file_path, nrows=10000)
                    
            elif file_path.suffix == '.jsonl' or file_path.name.endswith('.jsonl.gz'):
                file_info['format'] = 'JSON Lines' + (' (gzipped)' if file_path.name.endswith('.gz') else '')
                df = pd.read_json(file_path, lines=True, nrows=10000)
                
            else:
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return None, file_info
            
            return df, file_info
            
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None, file_info
    
    def _analyze_schema(self, df: pd.DataFrame) -> None:
        """
        Analyze and report schema information.
        
        Args:
            df: DataFrame to analyze
        """
        self.report_sections.append("### Schema\n\n")
        self.report_sections.append("| Column | Type | Non-Null | Unique | Sample Values |\n")
        self.report_sections.append("|--------|------|----------|--------|---------------|\n")
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            non_null_pct = (non_null / len(df)) * 100
            unique = df[col].nunique()
            
            # Get sample values (non-null)
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ', '.join([str(v)[:30] for v in sample_vals])
            
            self.report_sections.append(
                f"| `{col}` | {dtype} | {non_null:,} ({non_null_pct:.1f}%) | "
                f"{unique:,} | {sample_str} |\n"
            )
        
        self.report_sections.append("\n")
    
    def _identify_key_fields(self, df: pd.DataFrame) -> None:
        """
        Identify time, repository, and metric fields.
        
        Args:
            df: DataFrame to analyze
        """
        self.report_sections.append("### Identified Key Fields\n\n")
        
        # Time fields (common names and datetime types)
        time_keywords = ['time', 'date', 'timestamp', 'created', 'updated', 'at']
        time_fields = [
            col for col in df.columns 
            if any(kw in col.lower() for kw in time_keywords) 
            or pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        
        if time_fields:
            self.report_sections.append(f"**Time fields:** {', '.join(f'`{f}`' for f in time_fields)}\n\n")
        else:
            self.report_sections.append("⚠️ **No obvious time fields detected**\n\n")
        
        # Repository fields
        repo_keywords = ['repo', 'repository', 'project', 'name', 'id']
        repo_fields = [
            col for col in df.columns 
            if any(kw in col.lower() for kw in repo_keywords)
        ]
        
        if repo_fields:
            self.report_sections.append(f"**Repository fields:** {', '.join(f'`{f}`' for f in repo_fields)}\n\n")
        else:
            self.report_sections.append("⚠️ **No obvious repository fields detected**\n\n")
        
        # Numeric metric fields
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_keywords = ['star', 'fork', 'commit', 'pull', 'issue', 'contributor', 'watch', 'count']
        metric_fields = [
            col for col in numeric_cols 
            if any(kw in col.lower() for kw in metric_keywords)
        ]
        
        if metric_fields:
            self.report_sections.append(f"**Metric fields:** {', '.join(f'`{f}`' for f in metric_fields)}\n\n")
        else:
            self.report_sections.append(f"**Numeric fields:** {', '.join(f'`{f}`' for f in numeric_cols[:10])}\n\n")
    
    def _add_sample_rows(self, df: pd.DataFrame) -> None:
        """
        Add sample rows to report.
        
        Args:
            df: DataFrame to sample
        """
        self.report_sections.append("### Sample Rows\n\n")
        
        # Limit columns displayed to avoid overly wide tables
        display_cols = df.columns[:10].tolist()
        if len(df.columns) > 10:
            self.report_sections.append(f"*(Showing first 10 of {len(df.columns)} columns)*\n\n")
        
        sample_df = df[display_cols].head(self.sample_size)
        
        # Convert to markdown table
        self.report_sections.append(sample_df.to_markdown(index=False))
        self.report_sections.append("\n\n")
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """
        Check for data quality issues.
        
        Args:
            df: DataFrame to check
        """
        self.report_sections.append("### Data Quality Issues\n\n")
        
        issues_found = False
        
        # Missing values
        missing = df.isnull().sum()
        high_missing = missing[missing > len(df) * 0.5]
        if not high_missing.empty:
            issues_found = True
            self.report_sections.append("⚠️ **High missing values (>50%):**\n")
            for col, count in high_missing.items():
                pct = (count / len(df)) * 100
                self.report_sections.append(f"- `{col}`: {count:,} ({pct:.1f}%)\n")
            self.report_sections.append("\n")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues_found = True
            dup_pct = (duplicates / len(df)) * 100
            self.report_sections.append(
                f"⚠️ **Duplicate rows:** {duplicates:,} ({dup_pct:.1f}%)\n\n"
            )
        
        # Check for mixed types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to infer if column has mixed types
                sample = df[col].dropna().head(100)
                types = sample.apply(type).unique()
                if len(types) > 1:
                    issues_found = True
                    self.report_sections.append(
                        f"⚠️ **Mixed types in `{col}`:** {[t.__name__ for t in types]}\n\n"
                    )
            except:
                pass
        
        if not issues_found:
            self.report_sections.append("No major data quality issues detected\n\n")
    
    def _write_report(self) -> None:
        """Write the complete report to file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            f.write(''.join(self.report_sections))
        
        # Also add assumptions section
        with open(self.output_path, 'a') as f:
            f.write(self._generate_assumptions())
    
    def _generate_assumptions(self) -> str:
        """Generate assumptions section for the report."""
        assumptions = """
## Assumptions & Design Decisions

Based on the data inspection, the following assumptions are made for subsequent processing:

### Data Schema Assumptions

1. **Time Fields**: We expect one of the following fields to represent timestamps:
   - `created_at`, `updated_at`, `timestamp`, `date`, or similar
   - If multiple time fields exist, we prioritize creation/event time over update time
   - Timestamps should be ISO 8601 format or Unix epoch

2. **Repository Identification**: 
   - Repositories are identified by `repo_id`, `repository_id`, `id`, or `repo_name`
   - If both ID and name exist, ID is preferred for uniqueness
   - Repository names may change over time; IDs are assumed stable

3. **Metrics Aggregation**:
   - **Cumulative metrics** (stars, forks, watchers): Use the last value in each quarter
   - **Event metrics** (commits, pull_requests, issues): Sum all events in each quarter
   - **Rate metrics** (if present): Average over the quarter
   
4. **Missing Data Handling**:
   - Missing quarters for a repository: Assumed to have zero activity
   - Missing values within a record: Forward-fill for cumulative metrics, zero for event counts
   - Repositories with no activity after 4 consecutive quarters: May be considered inactive

### Processing Assumptions

5. **Quarterly Binning**:
   - Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
   - Data is binned by event timestamp (not update timestamp)
   - Quarter boundaries use UTC timezone

6. **Activity Labeling**:
   - **Active** repositories show meaningful engagement (commits, PRs, issues)
   - Threshold will be determined via validation data analysis
   - Weighted score: `commits × 1.0 + pull_requests × 2.0 + issues × 0.5 + stars × 0.1`
   - Rationale: PRs indicate code contribution (high value), commits show development, issues show engagement, stars show interest but not activity

7. **Time Range**:
   - Default analysis range: 2015-01-01 to latest available date
   - Training/validation split: Temporal split (earlier data for training)
   - Rolling evaluation: Minimum 3 years training data before first prediction

### Known Limitations

- If repository creation dates are unavailable, we use first observed timestamp
- Very new repositories (<2 quarters of data) may not have reliable forecasts
- Archived/deleted repositories may show as inactive but are not distinguishable from naturally inactive repos

### Data Quality Notes

- Any specific issues found during inspection are noted in the file-specific sections above
- Manual validation recommended for edge cases and extreme values

"""
        return assumptions


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Inspect raw GitHub repository data and generate report'
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
        default='data_inspection/report.md',
        help='Path where report will be written'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5,
        help='Number of sample rows to include per file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Run inspection
    inspector = DataInspector(args.input, args.output, args.sample_size)
    inspector.inspect()
    
    print(f"\nInspection complete! Report saved to: {args.output}")


if __name__ == '__main__':
    main()
