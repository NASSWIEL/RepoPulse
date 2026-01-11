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

We process these logs to extract temporal patterns rather than treating repositories as static snapshots. The dataset covers 500 repositories with historical data spanning multiple years.

### Feature Extraction

We aggregate events into calendar quarters (Q1-Q4) to balance temporal resolution with statistical stability. For each repository-quarter pair, we compute seven metrics:

1. **commit_count**: Total commits in the quarter
2. **total_contributors**: Unique authors contributing commits
3. **issue_count**: Issues opened
4. **issue_closed**: Issues closed
5. **pr_count**: Pull requests opened
6. **pr_merged**: Pull requests merged
7. **star_count**: Stars received

These metrics capture different aspects of project health: development velocity (commits), community engagement (contributors), project management (issues), code quality (PR merges), and popularity (stars).

### Activity Labeling

We define repository activity through a weighted scoring function that reflects the relative importance of different engagement signals:

```
activity_score = commit_count × 1.0 
               + total_contributors × 2.0 
               + issue_count × 0.5 
               + issue_closed × 1.0 
               + pr_count × 0.8 
               + pr_merged × 1.5 
               + star_count × 0.01
```

The weights encode several assumptions: contributors indicate sustained community (weight 2.0), merged PRs represent quality contributions (1.5), while stars can be ephemeral (0.01). We compute the 75th percentile threshold across all quarters (766.27) and label quarters above this as "active" (1) and below as "inactive" (0). This yields a 21.7% active rate, creating moderate class imbalance.

### Preprocessing Pipeline

The preprocessing chain consists of three stages:

1. **Aggregation** (`aggregate_quarters_enhanced.py`): Parse JSON logs, bin events by quarter, compute metrics. Missing data (repositories with no events in a quarter) are filled with zeros rather than excluded, preserving temporal continuity.

2. **Labeling** (`label_activity.py`): Apply the activity scoring formula and threshold to generate binary labels. The output is a labeled dataset with 16,697 repository-quarter records.

3. **Sequence Generation** (`prepare_timeseries_data.py`): Convert the flat dataset into sequences suitable for time-series forecasting. Each training example consists of a 4-quarter lookback window (input) and a 1-quarter forecast horizon (target). We apply z-score normalization per feature using training set statistics, with a floor value (1e-8) on standard deviations to prevent numerical instability.

The data is split temporally: 70% train, 15% validation, 15% test, yielding 3,593 / 4,928 / 6,186 sequences. Temporal splitting ensures models are evaluated on future data they haven't seen.

### Models

**Baseline Models:**
- **Last Quarter**: Copies the most recent quarter's values forward (persistence model)
- **Moving Average**: Averages the 4-quarter lookback window

These baselines require no training and provide a reality check on whether learned models add value.

**Neural Models:**
- **LSTM**: 2-layer network with 32 hidden units per layer, 0.2 dropout between layers
- **GRU**: Identical architecture using GRU cells instead of LSTM

Both models take sequences of shape (batch, 4, 7) and output predictions of shape (batch, 1, 7). They are trained with MSE loss on the continuous metrics, using the Adam optimizer with a learning rate of 0.001 and batch size of 64. Early stopping with patience of 10 epochs prevents overfitting.

### Evaluation Strategy

We evaluate using autoregressive forecasting: the model predicts one quarter ahead, then uses that prediction (not the ground truth) as input for the next step. This mimics real deployment where future data is unavailable. Predictions are denormalized back to the original scale and clipped to non-negative values.

For activity classification, we apply the same scoring function and threshold to predicted metrics, converting numeric forecasts to binary labels. This tests whether models capture the patterns that determine activity status, not just raw metric values.

Metrics include:
- **Numeric**: RMSE, MAE (on continuous features)
- **Classification**: F1, Recall, ROC-AUC (on binary activity labels)

The focus is on classification performance since that's the practical use case.

## Installation

```bash
pip install torch pandas numpy scikit-learn pyyaml
```

Requires Python 3.9+.

## Usage

Run the full pipeline:

```bash
# 1. Aggregate raw event logs into quarterly metrics
python preprocessing/aggregate_quarters_enhanced.py

# 2. Compute activity scores and labels
python preprocessing/label_activity.py

# 3. Generate time-series sequences
python models/prepare_timeseries_data.py --lookback 4 --horizon 1

# 4. Train neural models
python models/train_forecasters.py --model lstm --epochs 30 --batch_size 64
python models/train_forecasters.py --model gru --epochs 30 --batch_size 64

# 5. Evaluate all models
python models/evaluate_forecasters.py --model lstm --hidden_size 32 --num_layers 2
python models/evaluate_forecasters.py --model gru --hidden_size 32 --num_layers 2
python models/evaluate_forecasters.py --model last
python models/evaluate_forecasters.py --model avg
```


## Project Structure

```
├── config/config.yaml              # Configuration parameters
├── preprocessing/
│   ├── aggregate_quarters_enhanced.py
│   ├── label_activity.py
│   └── inspect_data.py
├── models/
│   ├── prepare_timeseries_data.py
│   ├── train_forecasters.py
│   ├── evaluate_forecasters.py
│   ├── forecaster.py              # Model definitions
│   └── checkpoints/               # Saved models
├── data/
│   ├── raw/                       # Input event logs (not included)
│   └── processed/                 # Generated datasets
├── tests/                         # Unit tests
└── PROJECT_REPORT.tex             # Technical report
```

## License

MIT

