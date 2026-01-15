# 🚀 API Quick Reference

## Start the API

```bash
# Option 1: Using the startup script
./start_api.sh

# Option 2: Direct command
uvicorn api_service:app --reload
```

Access at: **http://localhost:8000/docs**

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict` | POST | Predict activity status from time-series |
| `/health` | GET | Check API health |
| `/model/info` | GET | Get model details |
| `/docs` | GET | Interactive Swagger UI |

---

## Example Request

```bash
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

---

## Input Format

**4 quarters × 8 metrics:**

1. `commit_count` - Number of commits
2. `contributor_count` - Unique contributors
3. `issue_count` - Issues opened
4. `pr_count` - Pull requests
5. `star_count` - Stars received
6. `watch_count` - Watchers
7. `release_count` - Releases
8. `fork_count` - Forks

---

## Response Format

```json
{
  "predicted_metrics": {
    "commit_count": 158.3,
    "contributor_count": 64.2,
    "issue_count": 98.1,
    "pr_count": 77.5,
    "star_count": 103.6,
    "watch_count": 49.8,
    "release_count": 7.3,
    "fork_count": 2.1
  },
  "activity_score": 1456.8,
  "activity_status": "active",
  "confidence": {
    "score_threshold": 1319.5,
    "score_distance": 137.3,
    "classification": "medium"
  },
  "model_info": {
    "model_type": "gru",
    "device": "cpu"
  }
}
```

---

## Activity Classification

- **Threshold**: 1319.5 (75th percentile from training data)
- **Active**: score ≥ 1319.5
- **Inactive**: score < 1319.5

---

## Testing

```bash
# Run automated tests
python test_api_simple.py

# Test health
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/model/info
```

---

## Features

✅ Loads best GRU checkpoint at startup  
✅ Automatic z-score normalization  
✅ Input validation (shape: 4×8)  
✅ Activity classification with confidence  
✅ Interactive Swagger UI  
✅ Clear error messages  

---

## Troubleshooting

**Model not found:**
```bash
# Check checkpoint exists
ls models/checkpoints/gru_best.pt
```

**Feature stats missing:**
```bash
# Check stats file
ls data/processed/timeseries/feature_stats.json
```

**Port already in use:**
```bash
# Use different port
uvicorn api_service:app --port 8001
```

---

## Architecture

```
Input (4×8 time-series)
        ↓
  Normalization
        ↓
    GRU Model
        ↓
  Denormalization
        ↓
  Activity Scoring
        ↓
  Classification
        ↓
Output (active/inactive + metrics)
```

---

**Model**: GRU (64 hidden units, 2 layers)  
**Framework**: PyTorch + FastAPI  
**Documentation**: http://localhost:8000/docs
