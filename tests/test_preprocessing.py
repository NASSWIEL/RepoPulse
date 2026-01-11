"""
Unit Tests for Preprocessing Functions
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

sys.path.append(str(Path(__file__).parent.parent / 'preprocessing'))
sys.path.append(str(Path(__file__).parent.parent / 'models'))


class TestQuarterlyAggregation(unittest.TestCase):
    """Test quarterly aggregation functionality."""
    
    def setUp(self):
        """Create synthetic test data."""
        # Generate synthetic repo data
        dates = pd.date_range('2018-01-01', '2018-12-31', freq='D')
        self.test_df = pd.DataFrame({
            'repo_id': np.random.choice(['repo_1', 'repo_2', 'repo_3'], len(dates)),
            'timestamp': dates,
            'commits': np.random.randint(0, 10, len(dates)),
            'stars': np.random.randint(0, 100, len(dates)),
            'issues': np.random.randint(0, 5, len(dates))
        })
    
    def test_timestamp_normalization(self):
        """Test timestamp parsing and normalization."""
        from aggregate_quarters import QuarterlyAggregator
        
        # Create temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'input'
            output_path = Path(tmpdir) / 'output' / 'quarters.parquet'
            input_path.mkdir(parents=True, exist_ok=True)
            
            # Save test data
            self.test_df.to_csv(input_path / 'test.csv', index=False)
            
            # Run aggregation
            agg = QuarterlyAggregator(str(input_path), str(output_path))
            agg.process()
            
            # Check output exists
            self.assertTrue(output_path.exists())
            
            # Load and verify
            result = pd.read_parquet(output_path)
            self.assertIn('quarter_id', result.columns)
            self.assertIn('year', result.columns)
            self.assertIn('quarter', result.columns)
    
    def test_quarterly_binning(self):
        """Test that data is correctly binned into quarters."""
        # Add quarter info
        self.test_df['year'] = self.test_df['timestamp'].dt.year
        self.test_df['quarter'] = self.test_df['timestamp'].dt.quarter
        
        # Group by repo and quarter
        grouped = self.test_df.groupby(['repo_id', 'year', 'quarter'])
        
        # Check all repos have quarterly aggregates
        self.assertGreater(len(grouped), 0)
        
        # Check quarters are in 1-4
        for (repo, year, quarter), group in grouped:
            self.assertIn(quarter, [1, 2, 3, 4])


class TestActivityLabeling(unittest.TestCase):
    """Test activity labeling functionality."""
    
    def setUp(self):
        """Create synthetic quarterly data."""
        quarters = pd.date_range('2018-01-01', '2018-12-31', freq='QS')
        
        self.test_df = pd.DataFrame({
            'repo_id': np.repeat(['repo_1', 'repo_2'], len(quarters)),
            'quarter_start': list(quarters) * 2,
            'commits': np.random.randint(0, 50, len(quarters) * 2),
            'stars': np.random.randint(0, 100, len(quarters) * 2),
            'issues': np.random.randint(0, 20, len(quarters) * 2)
        })
    
    def test_activity_score_computation(self):
        """Test activity score calculation."""
        from label_activity import ActivityLabeler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'quarters.parquet'
            output_path = Path(tmpdir) / 'labeled.parquet'
            
            self.test_df.to_parquet(input_path)
            
            labeler = ActivityLabeler(str(input_path), str(output_path))
            labeler._load_data()
            labeler._compute_activity_scores()
            
            # Check activity score exists
            self.assertIn('activity_score', labeler.data.columns)
            self.assertEqual(len(labeler.data), len(self.test_df))
    
    def test_binary_labels(self):
        """Test binary label assignment."""
        from label_activity import ActivityLabeler
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'quarters.parquet'
            output_path = Path(tmpdir) / 'labeled.parquet'
            
            self.test_df.to_parquet(input_path)
            
            labeler = ActivityLabeler(str(input_path), str(output_path), threshold=1.0)
            labeler.process()
            
            # Load result
            result = pd.read_parquet(output_path)
            
            # Check labels exist and are binary
            self.assertIn('is_active', result.columns)
            self.assertTrue(set(result['is_active'].unique()).issubset({0, 1}))


class TestSequenceCreation(unittest.TestCase):
    """Test time series sequence creation."""
    
    def test_create_sequences(self):
        """Test sequence creation from dataframe."""
        from forecaster import create_sequences
        
        # Create test data
        quarters = pd.date_range('2018-01-01', '2019-12-31', freq='QS')
        df = pd.DataFrame({
            'repo_id': ['repo_1'] * len(quarters),
            'quarter_start': quarters,
            'metric1': np.random.randn(len(quarters)),
            'metric2': np.random.randn(len(quarters))
        })
        
        # Create sequences
        X, y, repo_ids = create_sequences(df, sequence_length=2, 
                                         target_columns=['metric1', 'metric2'])
        
        # Check shapes
        self.assertEqual(X.shape[1], 2)  # sequence length
        self.assertEqual(X.shape[2], 2)  # number of features
        self.assertEqual(y.shape[1], 2)  # number of features
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(repo_ids))


if __name__ == '__main__':
    unittest.main()
