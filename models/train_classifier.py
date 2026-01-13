#!/usr/bin/env python3
"""
Train Binary Classifier for Activity Prediction with MLflow Tracking

MLflow Integration:
- Logs all model hyperparameters
- Tracks classification metrics (precision, recall, F1, AUC)
- Saves model artifacts with signature
- Enables model comparison and registry

Usage:
    python train_classifier.py --input data/processed/quarters_labeled.parquet --model logistic
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

import sys
sys.path.append(str(Path(__file__).parent))

from classifier import ActivityClassifier

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow_utils import (
    setup_mlflow_experiment,
    log_sklearn_model,
    log_dataset_info
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main training script with comprehensive MLflow integration.
    
    MLflow captures:
    1. Model type and hyperparameters
    2. Dataset statistics and class distribution
    3. All classification metrics (precision, recall, F1, ROC-AUC, PR-AUC)
    4. Trained model with signature for deployment
    5. Confusion matrix and classification report
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/processed/quarters_labeled.parquet')
    parser.add_argument('--model', type=str, choices=['logistic', 'rf', 'gbm'], default='logistic')
    parser.add_argument('--output-dir', type=str, default='models/checkpoints')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--no_mlflow', action='store_true', help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
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
    
    use_mlflow = not args.no_mlflow
    
    if use_mlflow:
        # MLflow Experiment Setup
        run_name = f"classifier-{args.model}"
        
        with setup_mlflow_experiment(
            config=config,
            experiment_type='classification',
            run_name=run_name,
            tags={'model_type': args.model, 'task': 'binary_classification'}
        ) as run:
            logger.info(f"\n📊 MLflow Run ID: {run.info.run_id}")
            logger.info(f"📊 MLflow Experiment: {run.info.experiment_id}")
            
            # Log hyperparameters
            mlflow.log_params({
                'model_type': args.model,
                'test_size': args.test_size,
                'random_seed': args.random_seed,
                'n_features': len(feature_cols),
                'n_samples': len(X),
                'class_0_count': int(np.sum(y == 0)),
                'class_1_count': int(np.sum(y == 1)),
                'class_imbalance_ratio': float(np.sum(y == 1) / np.sum(y == 0))
            })
            
            # Log feature names
            mlflow.log_text('\n'.join(feature_cols), 'features.txt')
            
            # Log dataset information
            log_dataset_info(
                dataset_path=str(args.input),
                dataset_type='train',
                n_samples=len(X_train),
                n_features=len(feature_cols),
                additional_info={
                    'positive_class_count': int(np.sum(y_train == 1)),
                    'negative_class_count': int(np.sum(y_train == 0))
                }
            )
            log_dataset_info(
                dataset_path=str(args.input),
                dataset_type='test',
                n_samples=len(X_test),
                n_features=len(feature_cols),
                additional_info={
                    'positive_class_count': int(np.sum(y_test == 1)),
                    'negative_class_count': int(np.sum(y_test == 0))
                }
            )
            
            # Train
            logger.info(f"Training {args.model} classifier...")
            classifier = ActivityClassifier(model_type=args.model, random_seed=args.random_seed)
            classifier.fit(X_train, y_train)
            
            # Evaluate
            logger.info("Evaluating on test set...")
            metrics = classifier.evaluate(X_test, y_test)
            
            # Log all classification metrics
            mlflow.log_metrics({
                'test_precision': metrics['precision'],
                'test_recall': metrics['recall'],
                'test_f1': metrics['f1'],
                'test_accuracy': metrics['accuracy'],
                'test_roc_auc': metrics['roc_auc'],
                'test_pr_auc': metrics['pr_auc']
            })
            
            # Log confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, classifier.predict(X_test))
            mlflow.log_metrics({
                'confusion_matrix_tn': int(cm[0, 0]),
                'confusion_matrix_fp': int(cm[0, 1]),
                'confusion_matrix_fn': int(cm[1, 0]),
                'confusion_matrix_tp': int(cm[1, 1])
            })
            
            logger.info(f"Test Precision: {metrics['precision']:.4f}")
            logger.info(f"Test Recall: {metrics['recall']:.4f}")
            logger.info(f"Test F1: {metrics['f1']:.4f}")
            logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
            
            # Save locally
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            model_path = Path(args.output_dir) / f"{args.model}_classifier.pkl"
            classifier.save(str(model_path))
            
            # Save metrics
            metrics_path = Path(args.output_dir) / f"{args.model}_classifier_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Log artifacts to MLflow
            mlflow.log_artifact(str(model_path), artifact_path='models')
            mlflow.log_artifact(str(metrics_path), artifact_path='metrics')
            
            # Log sklearn model with signature
            log_sklearn_model(
                model=classifier.model,
                model_name=f"{args.model}_classifier",
                input_sample=X_test[:5],
                output_sample=y_test[:5],
                register_model=True,
                registered_model_name=f"activity-classifier-{args.model}"
            )
            
            logger.info(f"\n✅ MLflow tracking complete!")
            logger.info(f"📊 View results: mlflow ui --backend-store-uri mlruns")
            logger.info(f"🔗 Run ID: {run.info.run_id}")
    
    else:
        # Original training without MLflow
        logger.info("\nMLflow tracking disabled")
        
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
    
    logger.info(f"\nTraining complete! Model: {model_path}")


if __name__ == '__main__':
    main()
