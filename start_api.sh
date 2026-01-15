#!/bin/bash
# Quick start script for the API service

echo "================================================"
echo "Repository Activity Prediction API"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Check if FastAPI is installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing FastAPI dependencies..."
    pip install -q fastapi uvicorn pydantic
fi

# Check if model checkpoint exists
if [ ! -f "models/checkpoints/gru_best.pt" ]; then
    echo "❌ Model checkpoint not found: models/checkpoints/gru_best.pt"
    echo "   Please train the model first"
    exit 1
fi

# Check if feature stats exist
if [ ! -f "data/processed/timeseries/feature_stats.json" ]; then
    echo "❌ Feature stats not found: data/processed/timeseries/feature_stats.json"
    echo "   Please run data preprocessing first"
    exit 1
fi

echo "✅ All requirements satisfied"
echo ""
echo "Starting API server..."
echo "  - Health check: http://localhost:8000/health"
echo "  - Swagger UI: http://localhost:8000/docs"
echo "  - Model info: http://localhost:8000/model/info"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

# Start the server
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
