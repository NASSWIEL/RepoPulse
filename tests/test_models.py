"""
Unit Tests for Forecasting Models
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'models'))

from forecaster import (
    NaiveForecaster, MovingAverageForecaster,
    LSTMForecaster, GRUForecaster
)
from classifier import ActivityClassifier


class TestBaselineForecasters(unittest.TestCase):
    """Test baseline forecasting models."""
    
    def setUp(self):
        """Create synthetic data."""
        self.n_samples = 100
        self.seq_len = 5
        self.n_features = 3
        
        self.X = np.random.randn(self.n_samples, self.seq_len, self.n_features)
        self.y = np.random.randn(self.n_samples, self.n_features)
    
    def test_naive_forecaster(self):
        """Test naive forecaster."""
        model = NaiveForecaster()
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        
        # Check shape
        self.assertEqual(predictions.shape, (self.n_samples, self.n_features))
        
        # Check it returns last value
        np.testing.assert_array_equal(predictions, self.X[:, -1, :])
    
    def test_moving_average_forecaster(self):
        """Test moving average forecaster."""
        model = MovingAverageForecaster(window_size=3)
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        
        # Check shape
        self.assertEqual(predictions.shape, (self.n_samples, self.n_features))
        
        # Check values are averages
        expected = np.mean(self.X[:, -3:, :], axis=1)
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_evaluate(self):
        """Test evaluation metrics."""
        model = NaiveForecaster()
        model.fit(self.X, self.y)
        
        metrics = model.evaluate(self.X, self.y)
        
        # Check metrics exist
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        
        # Check values are non-negative
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
        self.assertGreaterEqual(metrics['rmse'], 0)


class TestNeuralForecasters(unittest.TestCase):
    """Test neural forecasting models."""
    
    def setUp(self):
        """Create synthetic data."""
        self.n_samples = 50
        self.seq_len = 4
        self.n_features = 2
        
        self.X_train = np.random.randn(self.n_samples, self.seq_len, self.n_features)
        self.y_train = np.random.randn(self.n_samples, self.n_features)
        self.X_test = np.random.randn(10, self.seq_len, self.n_features)
        self.y_test = np.random.randn(10, self.n_features)
    
    def test_lstm_forecaster(self):
        """Test LSTM forecaster."""
        model = LSTMForecaster(
            input_size=self.n_features,
            hidden_size=8,
            num_layers=1,
            epochs=2,
            batch_size=16
        )
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        # Check shape
        self.assertEqual(predictions.shape, (10, self.n_features))
        
        # Check model is fitted
        self.assertTrue(model.is_fitted)
    
    def test_gru_forecaster(self):
        """Test GRU forecaster."""
        model = GRUForecaster(
            input_size=self.n_features,
            hidden_size=8,
            num_layers=1,
            epochs=2,
            batch_size=16
        )
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        # Check shape
        self.assertEqual(predictions.shape, (10, self.n_features))
    
    def test_device_selection(self):
        """Test device selection (CPU/GPU)."""
        model = LSTMForecaster(
            input_size=self.n_features,
            hidden_size=8,
            device='cpu'
        )
        
        self.assertEqual(model.device.type, 'cpu')


class TestActivityClassifier(unittest.TestCase):
    """Test binary classifier."""
    
    def setUp(self):
        """Create synthetic classification data."""
        self.n_samples = 100
        self.n_features = 5
        
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
    
    def test_logistic_classifier(self):
        """Test logistic regression classifier."""
        classifier = ActivityClassifier(model_type='logistic')
        classifier.fit(self.X, self.y)
        
        predictions = classifier.predict(self.X)
        probas = classifier.predict_proba(self.X)
        
        # Check shapes
        self.assertEqual(predictions.shape, (self.n_samples,))
        self.assertEqual(probas.shape, (self.n_samples, 2))
        
        # Check predictions are binary
        self.assertTrue(set(predictions).issubset({0, 1}))
        
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(
            probas.sum(axis=1), 
            np.ones(self.n_samples)
        )
    
    def test_random_forest_classifier(self):
        """Test random forest classifier."""
        classifier = ActivityClassifier(
            model_type='rf',
            n_estimators=10
        )
        classifier.fit(self.X, self.y)
        
        predictions = classifier.predict(self.X)
        self.assertEqual(predictions.shape, (self.n_samples,))
    
    def test_evaluate_metrics(self):
        """Test evaluation metrics."""
        classifier = ActivityClassifier(model_type='logistic')
        classifier.fit(self.X, self.y)
        
        metrics = classifier.evaluate(self.X, self.y)
        
        # Check metrics exist
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('roc_auc', metrics)
        
        # Check values are in valid range [0, 1]
        for metric in ['precision', 'recall', 'f1', 'roc_auc']:
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)


if __name__ == '__main__':
    unittest.main()
