#!/usr/bin/env python3
"""
Train Binary Classifier for Activity Prediction

Usage:
    python train_classifier.py --input data/processed/quarters_labeled.parquet
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent))

from classifier import ActivityClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/quarters_labeled.parquet')
    parser.add_argument('--model', type=str, choices=['logistic', 'rf', 'gbm'], default='logistic')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    df = pd.read_parquet(args.input)
    
    # Features: use numeric metrics and activity score
    feature_cols = [col for col in df.columns 
                   if pd.api.types.is_numeric_dtype(df[col]) 
                   and col not in ['year', 'quarter', 'is_active']]
    
    X = df[feature_cols].fillna(0).values
    y = df['is_active'].values
    
    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    
    # Train
    logger.info(f"Training {args.model} classifier...")
    classifier = ActivityClassifier(model_type=args.model, random_seed=args.random_seed)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)
    
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1: {metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
    
    # Save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_dir) / f"{args.model}_classifier.pkl"
    classifier.save(str(model_path))
    
    # Save metrics
    metrics_path = Path(args.output_dir) / f"{args.model}_classifier_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training complete! Model: {model_path}")


if __name__ == '__main__':
    main()
