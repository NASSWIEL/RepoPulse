# RepoPulse

This project tackles the problem of forecasting GitHub repository activity through autoregressive time-series modeling. We extract quarterly metrics from repository event logs and train models to predict future activity levels. The goal is to classify repositories as active or inactive one quarter ahead, which could inform decisions about project maintenance, resource allocation, or automated health monitoring.

The approach compares neural sequence models (LSTM, GRU) against simple baselines (last quarter persistence, moving average). Our experiments reveal that repository activity exhibits strong temporal autocorrelation, making simple statistical models surprisingly effective.

## Motivation

Open-source repositories follow unpredictable lifecycles. Some maintain steady contribution patterns while others go dormant without warning. Forecasting activity could help maintainers anticipate resource needs, identify projects at risk of abandonment, or guide contributor recommendations. This work evaluates whether deep learning offers advantages over statistical baselines for this forecasting task.

## Methodology

### Data Collection

The input consists of GitHub event logs organized by repository. Each repository directory contains four JSON files with newline-delimited records:

- `commits.json`: Commit timestamps and author information
- `issues.json`: Issue creation/closure events
- `pull_requests.json`: PR creation/merge events  
- `stars.json`: Star timestamps

We process these logs to extract temporal patterns rather than treating repositories as static snapshots. The dataset covers 900 repositories with historical data spanning multiple years.

### Feature Extraction

We aggregate events into calendar quarters (Q1-Q4) to balance temporal resolution with statistical stability. For each repository-quarter pair, we compute eight metrics:

1. **commit_count**: Total commits in the quarter
2. **total_contributors**: Unique authors contributing commits
3. **issue_count**: Issues opened
4. **issue_closed**: Issues closed
5. **pr_count**: Pull requests opened
6. **pr_merged**: Pull requests merged
7. **star_count**: Stars received
8. **fork_count**: Forks created

These metrics capture different aspects of project health: development velocity (commits), community engagement (contributors), project management (issues), code quality (PR merges), popularity (stars), and project adoption (forks).

### Activity Labeling

We define repository activity through a weighted scoring function that reflects the relative importance of different engagement signals:

```
activity_score = commit_count × 1.0 
               + total_contributors × 2.0 
               + issue_count × 0.5 
               + issue_closed × 1.0 
               + pr_count × 0.8 
               + pr_merged × 1.5 
               + star_count × 0.2
               + fork_count × 0.3
```



### Preprocessing Pipeline

The preprocessing chain consists of three stages:

1. **Aggregation** (`aggregate_quarters_enhanced.py`): Parse JSON logs, bin events by quarter, compute metrics. Missing data (repositories with no events in a quarter) are filled with zeros rather than excluded, preserving temporal continuity.

2. **Labeling** (`label_activity.py`): Apply the activity scoring formula and threshold to generate binary labels. The output is a labeled dataset with 31,638 repository-quarter records.

3. **Sequence Generation** (`prepare_timeseries_data.py`): Convert the flat dataset into sequences suitable for time-series forecasting. Each training example consists of a 4-quarter lookback window (input) and a 1-quarter forecast horizon (target). We apply z-score normalization per feature using training set statistics, with a floor value (1e-8) on standard deviations to prevent numerical instability.

The data is split temporally based on quarter indices: **85% train, 10% dev (validation), 5% test**. This temporal splitting ensures models are evaluated on future data they haven't seen, simulating real-world deployment where predictions must be made for upcoming quarters.

### Models

**Baseline Models:**
- **Last Quarter**: Copies the most recent quarter's values forward (persistence model)
- **Moving Average**: Averages the 4-quarter lookback window

These baselines require no training and provide a reality check on whether learned models add value.

**Neural Models:**
- **LSTM**: 1-layer network with 16 hidden units, 0.3 dropout, optimized for dataset size
- **GRU**: Identical architecture using GRU cells instead of LSTM

Both models take sequences of shape (batch, 4, 8) and output predictions of shape (batch, 1, 8). They are trained with MSE loss on the continuous metrics, using the Adam optimizer with a learning rate of 0.001 and batch size of 64. Early stopping with patience of 10 epochs prevents overfitting. Model capacity is carefully tuned to avoid overfitting given the 7,004 training samples.

### Evaluation Strategy

We evaluate using autoregressive forecasting: the model predicts one quarter ahead, then uses that prediction (not the ground truth) as input for the next step. This mimics real deployment where future data is unavailable. Predictions are denormalized back to the original scale and clipped to non-negative values.

For activity classification, we apply the same scoring function and threshold to predicted metrics, converting numeric forecasts to binary labels. This tests whether models capture the patterns that determine activity status, not just raw metric values.

Metrics include:
- **Numeric**: RMSE, MAE (on continuous features)
- **Classification**: F1, Recall, ROC-AUC (on binary activity labels)

The focus is on classification performance since that's the practical use case.

## Installation

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn pyyaml mlflow

# Or use requirements.txt
pip install -r requirements.txt
```

Requires Python 3.9+.


### Viewing Results

```bash
# Start MLflow UI (from project root)
mlflow ui --backend-store-uri mlruns

# Access at http://localhost:5000
```

The UI provides:
- Interactive experiment comparison
- Metric visualization over time
- Parameter correlation analysis
- Model artifact browser

### Training with MLflow

```bash
# Train with MLflow tracking (default)
python models/train_forecasters.py --model lstm --epochs 50 --hidden_size 64

# Train without MLflow
python models/train_forecasters.py --model lstm --no_mlflow

# Train classifier with tracking
python models/train_classifier.py --model logistic
```

### Programmatic Access

See [notebooks/mlflow_demo.ipynb](notebooks/mlflow_demo.ipynb) for examples of:
- Querying experiments and comparing runs
- Loading models from registry
- Extracting best hyperparameters
- Generating comparison visualizations
- Exporting results for reports

### MLflow Configuration

Configuration is in [config/config.yaml](config/config.yaml):

```yaml
mlflow:
  tracking_uri: "mlruns"                         # Local tracking directory
  experiment_name: "github-activity-forecasting"  # Main experiment
  
  experiments:
    forecasting: "forecasting-models"             # LSTM/GRU experiments
    classification: "activity-classification"      # Classifier experiments
  
  registry:
    enabled: true                                  # Enable model registry
```

### For Presentation

**Key points to highlight:**
1. Every training run is tracked with complete hyperparameters
2. Easy comparison of 10+ model configurations
3. Best model automatically identified and registered
4. Complete reproducibility - can recreate any result
5. Production-ready with model signatures and versioning

**Demo Flow:**
1. Show MLflow UI with multiple runs
2. Compare metrics across models (LSTM vs GRU)
3. View training curves and convergence
4. Load best model from registry
5. Explain how this supports MLOps workflow

## Usage

Run the full pipeline with MLflow tracking:

```bash
# 1. Aggregate raw event logs into quarterly metrics
python preprocessing/aggregate_quarters_enhanced.py

# 2. Compute activity scores and labels
python preprocessing/label_activity.py

# 3. Generate time-series sequences
python models/prepare_timeseries_data.py --lookback 4 --horizon 1

# 4. Train neural models (with MLflow tracking)
python models/train_forecasters.py --model lstm --epochs 50 --hidden_size 64
python models/train_forecasters.py --model gru --epochs 50 --hidden_size 32

# 5. Train classifier (with MLflow tracking)
python models/train_classifier.py --model logistic
python models/train_classifier.py --model rf

# 6. View results in MLflow UI
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000

# 7. Evaluate models
python models/evaluate_forecasters.py --model lstm
python models/evaluate_forecasters.py --model gru
python models/evaluate_forecasters.py --model last  # baseline
python models/evaluate_forecasters.py --model avg   # baseline
```



## Project Structure

```
├── config/config.yaml              # Configuration (includes MLflow settings)
├── preprocessing/
│   ├── aggregate_quarters_enhanced.py  # Quarterly aggregation
│   ├── label_activity.py              # Activity labeling
│   └── inspect_data.py
├── models/
│   ├── mlflow_utils.py            # MLflow helper functions (NEW)
│   ├── prepare_timeseries_data.py # Sequence generation
│   ├── train_forecasters.py       # Training with MLflow tracking
│   ├── train_classifier.py        # Classification with MLflow
│   ├── evaluate_forecasters.py
│   ├── forecaster.py              # Model definitions
│   └── checkpoints/               # Saved models
├── notebooks/
│   ├── mlflow_demo.ipynb          # MLflow usage examples (NEW)
│   ├── explore.ipynb              # Data exploration
│   └── experiments.ipynb
├── data/
│   ├── raw/                       # Input event logs (not included)
│   └── processed/                 # Generated datasets
├── mlruns/                        # MLflow tracking data (NEW)
│   ├── experiments/
│   ├── artifacts/
│   └── models/
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies (includes mlflow)
└── PROJECT_REPORT.tex             # Technical report
```

## MLflow Artifacts Logged

For each training run, MLflow captures:

**Forecasting Models:**
- Hyperparameters: model_type, hidden_size, num_layers, dropout, learning_rate, etc.
- Metrics: train_loss, dev_loss, best_dev_loss (logged per epoch)
- Artifacts: Model checkpoints, training history JSON, PyTorch model with signature
- System: Training time, GPU/CPU usage, model parameter count

**Classification Models:**
- Hyperparameters: model_type, n_features, test_size, random_seed
- Metrics: precision, recall, f1, accuracy, roc_auc, pr_auc, confusion_matrix
- Artifacts: Trained model pickle, metrics JSON, sklearn model with signature
- Dataset: Sample counts, class distribution, imbalance ratio

## 🚀 Inference API Service

### FastAPI Service for Repository Activity Prediction

The project includes a clean FastAPI service that loads the best model checkpoint and provides interactive prediction capabilities.

**Quick Start:**
```bash
# 1. Install FastAPI dependencies
pip install fastapi uvicorn pydantic

# 2. Start the API server
uvicorn api_service:app --reload

# 3. Access interactive documentation
open http://localhost:8000/docs
```

**Test the API:**
```bash
# Run automated tests
python test_api_simple.py

# Or use curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": [
      [120, 55, 80, 65, 85, 42, 5, 0],
      [140, 60, 90, 70, 95, 45, 6, 0],
      [130, 58, 85, 68, 90, 44, 5, 0],
      [150, 62, 95, 75, 100, 48, 7, 0]
    ]
  }'
```

### API Endpoints

- **POST /predict** - Predict repository activity status from time-series
- **GET /health** - Check API health and model status  
- **GET /model/info** - Get model architecture details
- **GET /docs** - Interactive Swagger UI documentation

### Features

✅ Loads best-performing model checkpoint (GRU) at startup  
✅ Automatic feature normalization using training statistics  
✅ Activity classification (active/inactive) with confidence scores  
✅ Input validation for correct time-series shape (4 quarters × 8 features)  
✅ Interactive Swagger UI for testing  
✅ Clear, documented API responses  

### How It Works

1. **Input**: 4 quarters of historical metrics (8 features per quarter)
2. **Processing**: Model forecasts metrics for next quarter
3. **Scoring**: Computes activity score using weighted metrics
4. **Classification**: Classifies as active/inactive (threshold: 1319.5)
5. **Output**: Predicted metrics + activity status + confidence

## License

MIT

