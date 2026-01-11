#!/bin/bash
# Complete Pipeline Execution Script
# Usage: bash scripts/run_pipeline.sh

set -e  # Exit on error

echo "========================================"
echo "GitHub Repository Activity Forecasting"
echo "========================================"
echo ""

# Configuration
RAW_DATA_PATH="/info/raid-etu/m2/s2308975/big_data"
OUTPUT_DIR="/info/raid-etu/m2/s2405959/BigData"
QUARTERS_FILE="$OUTPUT_DIR/data/processed/quarters.parquet"
LABELED_FILE="$OUTPUT_DIR/data/processed/quarters_labeled.parquet"
MODEL_DIR="$OUTPUT_DIR/models/checkpoints"

# Create directories
mkdir -p "$OUTPUT_DIR/data/processed"
mkdir -p "$OUTPUT_DIR/data_inspection"
mkdir -p "$MODEL_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "Step 1/6: Data Inspection"
echo "-------------------------"
python preprocessing/inspect_data.py \
    --input "$RAW_DATA_PATH" \
    --output "$OUTPUT_DIR/data_inspection/report.md" \
    --verbose

echo ""
echo "✅ Data inspection complete"
echo ""

echo "Step 2/6: Quarterly Aggregation"
echo "-------------------------------"
python preprocessing/aggregate_quarters.py \
    --input "$RAW_DATA_PATH" \
    --output "$QUARTERS_FILE" \
    --imputation zero \
    --export-csv \
    --verbose

echo ""
echo "✅ Quarterly aggregation complete"
echo ""

echo "Step 3/6: Activity Labeling"
echo "---------------------------"
python preprocessing/label_activity.py \
    --input "$QUARTERS_FILE" \
    --output "$LABELED_FILE" \
    --threshold-method f1 \
    --min-quarters 2 \
    --export-csv \
    --verbose

echo ""
echo "✅ Activity labeling complete"
echo ""

echo "Step 4/6: Train Forecasting Models"
echo "-----------------------------------"

# Naive baseline
echo "  Training Naive forecaster..."
python models/train_forecaster.py \
    --input "$LABELED_FILE" \
    --model naive \
    --sequence-length 4 \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

# Moving Average
echo "  Training Moving Average forecaster..."
python models/train_forecaster.py \
    --input "$LABELED_FILE" \
    --model ma \
    --sequence-length 4 \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

# LSTM
echo "  Training LSTM forecaster..."
python models/train_forecaster.py \
    --input "$LABELED_FILE" \
    --model lstm \
    --sequence-length 4 \
    --hidden-size 64 \
    --epochs 50 \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

# GRU
echo "  Training GRU forecaster..."
python models/train_forecaster.py \
    --input "$LABELED_FILE" \
    --model gru \
    --sequence-length 4 \
    --hidden-size 64 \
    --epochs 50 \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

echo ""
echo "✅ Forecasting models trained"
echo ""

echo "Step 5/6: Train Classification Models"
echo "--------------------------------------"

# Logistic Regression
echo "  Training Logistic Regression classifier..."
python models/train_classifier.py \
    --input "$LABELED_FILE" \
    --model logistic \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

# Random Forest
echo "  Training Random Forest classifier..."
python models/train_classifier.py \
    --input "$LABELED_FILE" \
    --model rf \
    --output-dir "$MODEL_DIR" \
    --random-seed 42

echo ""
echo "✅ Classification models trained"
echo ""

echo "Step 6/6: Generate Summary Report"
echo "----------------------------------"

# Create summary report
REPORT_FILE="$OUTPUT_DIR/RESULTS_SUMMARY.md"

cat > "$REPORT_FILE" << EOF
# Pipeline Execution Summary

**Date:** $(date)

## Data Statistics

- **Raw Data Source:** $RAW_DATA_PATH
- **Output Directory:** $OUTPUT_DIR

EOF

# Add quarterly data stats
if [ -f "$QUARTERS_FILE" ]; then
    echo "### Quarterly Aggregated Data" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    python -c "import pandas as pd; df = pd.read_parquet('$QUARTERS_FILE'); print(f'Records: {len(df):,}'); print(f'Repositories: {df[\"repo_id\"].nunique():,}'); print(f'Date Range: {df[\"quarter_start\"].min()} to {df[\"quarter_end\"].max()}')" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Add labeled data stats
if [ -f "$LABELED_FILE" ]; then
    echo "### Labeled Data" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    python -c "import pandas as pd; df = pd.read_parquet('$LABELED_FILE'); active = df['is_active'].sum(); print(f'Total Records: {len(df):,}'); print(f'Active: {active:,} ({active/len(df)*100:.1f}%)'); print(f'Inactive: {len(df)-active:,} ({(1-active/len(df))*100:.1f}%)')" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# List trained models
echo "## Trained Models" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
ls -lh "$MODEL_DIR" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Add metrics if available
echo "## Model Performance" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

for metrics_file in "$MODEL_DIR"/*_metrics.json; do
    if [ -f "$metrics_file" ]; then
        echo "### $(basename $metrics_file .json)" >> "$REPORT_FILE"
        echo '```json' >> "$REPORT_FILE"
        cat "$metrics_file" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi
done

# Add next steps
cat >> "$REPORT_FILE" << EOF

## Next Steps

1. Review data inspection report: \`data_inspection/report.md\`
2. Explore threshold analysis: \`jupyter notebook notebooks/threshold_analysis.ipynb\`
3. Run experiments: \`jupyter notebook notebooks/experiments.ipynb\`
4. Tune hyperparameters for better performance
5. Implement rolling cross-validation for robust evaluation

## Files Generated

- Quarterly aggregated data: \`$QUARTERS_FILE\`
- Labeled data: \`$LABELED_FILE\`
- Model checkpoints: \`$MODEL_DIR/\`
- Data inspection: \`data_inspection/report.md\`
- This summary: \`$REPORT_FILE\`

EOF

echo ""
echo "✅ Summary report generated: $REPORT_FILE"
echo ""

echo "========================================"
echo "Pipeline execution complete! 🎉"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review: cat $REPORT_FILE"
echo "2. Inspect data: cat $OUTPUT_DIR/data_inspection/report.md"
echo "3. Run notebooks: jupyter notebook notebooks/"
echo ""
