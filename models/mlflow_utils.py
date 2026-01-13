#!/usr/bin/env python3
"""
MLflow Utilities for Experiment Tracking and Model Management

This module provides helper functions for consistent MLflow integration across
the project, following MLOps best practices.

Key Features:
- Experiment setup and configuration
- Automatic parameter and metric logging
- Model artifact management
- Model signature generation
- Model registry integration

Usage:
    from mlflow_utils import setup_mlflow, log_model_with_signature
    
    with setup_mlflow(config, "forecasting") as run:
        # Training code
        log_metrics({"mse": 0.05})
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import json

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import torch
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec

logger = logging.getLogger(__name__)


def initialize_mlflow(config: Dict[str, Any]) -> None:
    """
    Initialize MLflow with configuration settings.
    
    Args:
        config: Configuration dictionary with mlflow settings
    
    MLflow Best Practice: Centralized initialization ensures consistent tracking URI
    and artifact location across all experiments.
    """
    mlflow_config = config.get('mlflow', {})
    
    # Set tracking URI (local directory or remote server)
    tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Create artifacts directory if it doesn't exist
    artifact_location = mlflow_config.get('artifact_location', 'mlruns/artifacts')
    Path(artifact_location).mkdir(parents=True, exist_ok=True)
    
    logger.info("MLflow initialized successfully")


@contextmanager
def setup_mlflow_experiment(
    config: Dict[str, Any],
    experiment_type: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
):
    """
    Context manager for MLflow experiment setup and cleanup.
    
    Args:
        config: Configuration dictionary
        experiment_type: Type of experiment (forecasting, classification, preprocessing)
        run_name: Optional custom run name
        tags: Optional additional tags
    
    Yields:
        mlflow.ActiveRun: Active MLflow run context
    
    MLflow Best Practice: Use context manager to ensure proper cleanup and
    automatic run termination even if exceptions occur.
    
    Example:
        with setup_mlflow_experiment(config, "forecasting", "LSTM-v1") as run:
            # Training code here
            mlflow.log_param("hidden_size", 64)
    """
    initialize_mlflow(config)
    
    # Get experiment configuration
    mlflow_config = config.get('mlflow', {})
    experiments = mlflow_config.get('experiments', {})
    experiment_name = experiments.get(experiment_type, f"{experiment_type}-models")
    
    # Set experiment (creates if doesn't exist)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    
    # Prepare tags
    default_tags = mlflow_config.get('default_tags', {})
    run_tags = {**default_tags}
    if tags:
        run_tags.update(tags)
    run_tags['experiment_type'] = experiment_type
    
    # Start run
    with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        yield run
        logger.info(f"MLflow run completed: {run.info.run_id}")


def log_params_from_config(config: Dict[str, Any], prefix: str = "") -> None:
    """
    Log all configuration parameters to MLflow.
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names (for nested configs)
    
    MLflow Best Practice: Log all hyperparameters for complete experiment reproducibility.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            log_params_from_config(value, prefix=f"{prefix}{key}.")
        elif value is not None and not isinstance(value, (list, tuple)):
            param_name = f"{prefix}{key}"
            # MLflow has 250 char limit for param values
            param_value = str(value)[:250]
            mlflow.log_param(param_name, param_value)


def log_training_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Log training metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Training step/epoch number
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
    
    MLflow Best Practice: Log metrics at each epoch to track training progress
    and enable early stopping decisions.
    """
    for name, value in metrics.items():
        metric_name = f"{prefix}{name}"
        mlflow.log_metric(metric_name, value, step=step)


def create_model_signature(
    input_data: np.ndarray,
    output_data: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> ModelSignature:
    """
    Create MLflow model signature for input/output schema validation.
    
    Args:
        input_data: Sample input data
        output_data: Sample output data
        feature_names: Optional feature names
    
    Returns:
        ModelSignature object
    
    MLflow Best Practice: Define model signatures to ensure type safety and
    enable automatic validation during inference.
    """
    return infer_signature(input_data, output_data)


def log_pytorch_model(
    model: torch.nn.Module,
    model_name: str,
    input_sample: np.ndarray,
    output_sample: np.ndarray,
    conda_env: Optional[Dict] = None,
    register_model: bool = False,
    registered_model_name: Optional[str] = None
) -> None:
    """
    Log PyTorch model with signature and optional registration.
    
    Args:
        model: Trained PyTorch model
        model_name: Name for the model artifact
        input_sample: Sample input for signature inference
        output_sample: Sample output for signature inference
        conda_env: Optional conda environment specification
        register_model: Whether to register model in MLflow Model Registry
        registered_model_name: Name for registered model
    
    MLflow Best Practice: Log models with signatures and register production-ready
    models for deployment tracking and versioning.
    """
    # Create signature
    signature = create_model_signature(input_sample, output_sample)
    
    # Log model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        signature=signature,
        conda_env=conda_env,
        registered_model_name=registered_model_name if register_model else None
    )
    
    logger.info(f"Model logged: {model_name}")
    if register_model:
        logger.info(f"Model registered: {registered_model_name}")
    
    return model_info


def log_sklearn_model(
    model: Any,
    model_name: str,
    input_sample: np.ndarray,
    output_sample: np.ndarray,
    register_model: bool = False,
    registered_model_name: Optional[str] = None
) -> None:
    """
    Log scikit-learn model with signature and optional registration.
    
    Args:
        model: Trained scikit-learn model
        model_name: Name for the model artifact
        input_sample: Sample input for signature inference
        output_sample: Sample output for signature inference
        register_model: Whether to register model in MLflow Model Registry
        registered_model_name: Name for registered model
    
    MLflow Best Practice: Use framework-specific logging for optimal model
    serialization and deployment support.
    """
    signature = create_model_signature(input_sample, output_sample)
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_name,
        signature=signature,
        registered_model_name=registered_model_name if register_model else None
    )
    
    logger.info(f"Model logged: {model_name}")
    if register_model:
        logger.info(f"Model registered: {registered_model_name}")
    
    return model_info


def log_artifacts_from_dict(artifacts: Dict[str, Any], artifact_dir: str = "artifacts") -> None:
    """
    Log multiple artifacts (plots, configs, results) to MLflow.
    
    Args:
        artifacts: Dictionary mapping artifact names to file paths
        artifact_dir: Directory name for artifacts in MLflow
    
    MLflow Best Practice: Log all relevant artifacts (plots, configs, predictions)
    for complete experiment documentation.
    """
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        for name, filepath in artifacts.items():
            src = Path(filepath)
            if src.exists():
                dst = tmpdir_path / src.name
                shutil.copy(src, dst)
        
        # Log entire directory
        mlflow.log_artifacts(tmpdir, artifact_path=artifact_dir)
        logger.info(f"Artifacts logged to {artifact_dir}")


def log_dataset_info(
    dataset_path: str,
    dataset_type: str,
    n_samples: int,
    n_features: int,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Log dataset information as parameters and tags.
    
    Args:
        dataset_path: Path to dataset
        dataset_type: Type of dataset (train/val/test)
        n_samples: Number of samples
        n_features: Number of features
        additional_info: Additional dataset metadata
    
    MLflow Best Practice: Document data provenance and statistics for
    reproducibility and debugging.
    """
    mlflow.log_param(f"dataset_path_{dataset_type}", dataset_path)
    mlflow.log_param(f"n_samples_{dataset_type}", n_samples)
    mlflow.log_param(f"n_features_{dataset_type}", n_features)
    
    if additional_info:
        for key, value in additional_info.items():
            mlflow.log_param(f"dataset_{dataset_type}_{key}", str(value)[:250])


def log_model_to_registry(
    model_uri: str,
    model_name: str,
    stage: str = "Staging",
    description: Optional[str] = None
) -> None:
    """
    Register or update model in MLflow Model Registry.
    
    Args:
        model_uri: URI of the logged model
        model_name: Name for registered model
        stage: Model stage (None, Staging, Production, Archived)
        description: Optional model description
    
    MLflow Best Practice: Use Model Registry for production model tracking,
    versioning, and stage transitions.
    """
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Register model
    result = mlflow.register_model(model_uri, model_name)
    
    # Transition to specified stage
    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage=stage
        )
        logger.info(f"Model {model_name} v{result.version} transitioned to {stage}")
    
    # Add description
    if description:
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=description
        )


def get_best_run(experiment_name: str, metric: str, ascending: bool = True) -> Dict:
    """
    Get the best run from an experiment based on a metric.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric name to optimize
        ascending: If True, minimize metric; if False, maximize
    
    Returns:
        Dictionary with run information
    
    MLflow Best Practice: Programmatically identify best models for
    automated deployment pipelines.
    """
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment {experiment_name}")
    
    best_run = runs[0]
    
    return {
        'run_id': best_run.info.run_id,
        'metric_value': best_run.data.metrics.get(metric),
        'params': best_run.data.params,
        'tags': best_run.data.tags
    }


def compare_runs(run_ids: List[str], metrics: List[str]) -> Dict:
    """
    Compare multiple runs across specified metrics.
    
    Args:
        run_ids: List of run IDs to compare
        metrics: List of metric names to compare
    
    Returns:
        Dictionary mapping run_ids to their metrics
    
    MLflow Best Practice: Enable systematic model comparison for
    selection and evaluation.
    """
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    comparison = {}
    
    for run_id in run_ids:
        run = client.get_run(run_id)
        comparison[run_id] = {
            'name': run.data.tags.get('mlflow.runName', run_id[:8]),
            'metrics': {m: run.data.metrics.get(m) for m in metrics}
        }
    
    return comparison
