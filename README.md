# GitHub Repository Activity Forecasting

Autoregressive forecasting pipeline for predicting GitHub repository activity using quarterly time-series data. Compares LSTM/GRU neural networks against baseline methods for both numeric forecasting and binary activity classification.

## Overview

This project analyzes GitHub repository activity patterns by aggregating event data into quarterly metrics and forecasting future activity. The system labels repositories as "active" or "inactive" based on weighted activity scores and evaluates model performance using both regression and classification metrics.

**Key findings:** Simple baseline models (last quarter, moving average) significantly outperform neural networks for activity classification, achieving F1 scores of 0.998 vs 0.67-0.69 for LSTM/GRU. This reveals strong temporal persistence in repository activity patterns.

## Requirements

- Python 3.9+
- PyTorch
- pandas, numpy, scikit-learn
- PyYAML

```bash
pip install torch pandas numpy scikit-learn pyyaml
```

## Quick Start

### 1. Aggregate Data

Process raw GitHub event data into quarterly metrics:

```bash
python preprocessing/aggregate_quarters_enhanced.py
```

Input: JSON event files per repository (commits, issues, pull requests, stars)  
Output: `data/processed/quarterly_aggregated.parquet`

### 2. Label Activity

Compute activity scores and binary labels:

```bash
python preprocessing/label_activity.py
```

Applies weighted scoring formula using 75th percentile threshold (21.7% of quarters labeled active).

### 3. Prepare Sequences

Generate time-series sequences with 4-quarter lookback:

```bash
python models/prepare_timeseries_data.py --lookback 4 --horizon 1
```

Creates normalized train/dev/test splits (3,593 / 4,928 / 6,186 sequences).

### 4. Train Models

```bash
# LSTM
python models/train_forecasters.py --model lstm --epochs 30 --batch_size 64 --hidden_size 32 --num_layers 2

# GRU
python models/train_forecasters.py --model gru --epochs 30 --batch_size 64 --hidden_size 32 --num_layers 2
```

### 5. Evaluate

Autoregressive forecasting evaluation:

```bash
# Neural models
python models/evaluate_forecasters.py --model lstm --hidden_size 32 --num_layers 2
python models/evaluate_forecasters.py --model gru --hidden_size 32 --num_layers 2

# Baselines
python models/evaluate_forecasters.py --model last
python models/evaluate_forecasters.py --model avg
```

## Data Format

### Input Structure

```
repositories/
└── owner__repo_name/
    ├── commits.json        # Newline-delimited JSON
    ├── issues.json
    ├── pull_requests.json
    └── stars.json
```

### Aggregated Features (7 metrics per quarter)

- `commit_count`: Number of commits
- `total_contributors`: Unique contributors
- `issue_count`: Issues opened
- `issue_closed`: Issues closed
- `pr_count`: Pull requests opened
- `pr_merged`: Pull requests merged
- `star_count`: Stars received

## Methodology

### Activity Scoring

Weighted combination of metrics reflecting engagement importance:

```
score = commit_count × 1.0 
      + total_contributors × 2.0 
      + issue_count × 0.5 
      + issue_closed × 1.0 
      + pr_count × 0.8 
      + pr_merged × 1.5 
      + star_count × 0.01
```

Contributors weighted highest (2.0) as indicator of sustained community engagement. Merged PRs (1.5) valued over opened PRs (0.8) as quality signal. Stars weighted lowest (0.01) due to volatility without corresponding development activity.

### Models

**Baseline:**
- Last Quarter: Persistence model using most recent quarter
- Moving Average: Mean of 4-quarter lookback window

**Neural:**
- LSTM: 2-layer network, 32 hidden units, 0.2 dropout
- GRU: 2-layer network, 32 hidden units, 0.2 dropout

**Training:** MSE loss, Adam optimizer, batch size 64, early stopping (patience=10)

### Evaluation

Rolling autoregressive forecasting simulates deployment conditions where only historical data is available. Predictions are denormalized, clipped to non-negative values, and used to update the lookback window for subsequent steps.

## Results

Performance on 6,186 test sequences (1-quarter ahead):

| Model | RMSE | MAE | F1 | Recall | ROC-AUC |
|-------|------|-----|-----|--------|---------|
| LSTM | 876,785 | 172,153 | 0.67 | 0.51 | 0.46 |
| GRU | 876,682 | 172,051 | 0.69 | 0.53 | 0.48 |
| Last Quarter | 895,669 | 88,929 | **0.998** | 0.998 | **0.964** |
| Moving Average | 848,127 | 112,821 | **0.998** | 1.000 | 0.954 |

Baseline methods dramatically outperform neural models for activity classification. Repository activity exhibits strong autocorrelation that simple persistence models capture effectively. Neural models struggle with limited training data (3,593 sequences), high variance in metrics (max values 100× larger than means), and class imbalance (21.7% active).

## Configuration

Edit `config/config.yaml` for:
- Data paths
- Feature selections  
- Model hyperparameters
- Training parameters

## Project Structure

```
├── config/
│   └── config.yaml
├── preprocessing/
│   ├── aggregate_quarters_enhanced.py
│   ├── label_activity.py
│   └── inspect_data.py
├── models/
│   ├── prepare_timeseries_data.py
│   ├── train_forecasters.py
│   └── evaluate_forecasters.py
├── data/
│   ├── raw/                    # Not included in repository
│   └── processed/              # Generated outputs
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
└── PROJECT_REPORT.tex          # Technical report (LaTeX)
```

## Implementation Details

**Data Preprocessing:**
- Temporal aggregation into quarters (Q1-Q4)
- NaN filling with zeros for inactive quarters
- Z-score normalization with floor values (1e-8) to prevent division by zero
- Temporal train/dev/test split (70/15/15)

**Sequence Generation:**
- 4-quarter lookback window
- 1-quarter forecast horizon
- Autoregressive multi-step capability

**Model Training:**
- Best dev loss: LSTM 0.696, GRU 0.670
- Checkpoints saved to `models/checkpoints/`
- Training history logged

## Troubleshooting

**NaN values during training:**
- Verify data aggregation filled NaN with zeros
- Check `data/processed/timeseries/feature_stats.json` for invalid statistics
- Ensure normalization floor values applied

**Memory errors:**
- Reduce batch size: `--batch_size 32`
- Reduce model capacity: `--hidden_size 16`

**Poor performance:**
- Baseline models may be more appropriate for this task
- Verify temporal splits maintain chronological order
- Check data quality and distributions

## Testing

```bash
python -m pytest tests/
```

## Documentation

See `PROJECT_REPORT.tex` for comprehensive technical documentation including methodology, experimental design, and analysis.

Compile PDF:
```bash
pdflatex PROJECT_REPORT.tex
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
@techreport{github2026forecasting,
  title={Autoregressive Forecasting of GitHub Repository Activity: A Comparative Study of Neural and Baseline Methods},
  author={Big Data Project},
  year={2026}
}
```
