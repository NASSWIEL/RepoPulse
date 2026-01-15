#!/usr/bin/env python3
"""
Visual demo of API usage - Shows example predictions
"""

print("""
╔══════════════════════════════════════════════════════════════════════════╗
║         Repository Activity Prediction API - Usage Guide                ║
╚══════════════════════════════════════════════════════════════════════════╝

📋 STEP 1: Start the API
─────────────────────────────────────────────────────────────────────────
$ ./start_api.sh
# OR
$ uvicorn api_service:app --reload

✅ Server runs at: http://localhost:8000


📋 STEP 2: Access Interactive Documentation
─────────────────────────────────────────────────────────────────────────
Open in browser: http://localhost:8000/docs

You'll see:
  ✓ /predict       - Main prediction endpoint
  ✓ /health        - Check API status
  ✓ /model/info    - Model details
  ✓ Try it out     - Interactive testing


📋 STEP 3: Test with Sample Data
─────────────────────────────────────────────────────────────────────────

Example 1: ACTIVE Repository (High Activity)
────────────────────────────────────────────
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "sequence": [
      [200, 80, 150, 120, 180, 70, 10, 2],
      [220, 85, 160, 130, 200, 75, 12, 3],
      [210, 82, 155, 125, 190, 72, 11, 2],
      [230, 88, 165, 135, 210, 78, 13, 3]
    ]
  }'

Expected Output:
{
  "activity_status": "active",
  "activity_score": 2450.3,
  "predicted_metrics": {
    "commit_count": 235.2,
    "contributor_count": 90.1,
    ...
  }
}


Example 2: INACTIVE Repository (Low Activity)
──────────────────────────────────────────────
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "sequence": [
      [10, 5, 8, 3, 15, 8, 1, 0],
      [12, 6, 10, 4, 18, 9, 1, 0],
      [9, 5, 7, 3, 16, 8, 0, 0],
      [11, 6, 9, 4, 17, 9, 1, 0]
    ]
  }'

Expected Output:
{
  "activity_status": "inactive",
  "activity_score": 125.8,
  "predicted_metrics": {
    "commit_count": 11.3,
    "contributor_count": 6.2,
    ...
  }
}


📋 STEP 4: Run Automated Tests
─────────────────────────────────────────────────────────────────────────
$ python test_api_simple.py

Tests include:
  ✓ Health check
  ✓ Model info
  ✓ Active repository prediction
  ✓ Inactive repository prediction
  ✓ Moderate repository prediction
  ✓ Invalid input handling


📋 Understanding the Input Format
─────────────────────────────────────────────────────────────────────────
Each sequence contains 4 quarters with 8 metrics per quarter:

Quarter Structure (8 metrics):
  [0] commit_count        - Number of commits
  [1] contributor_count   - Unique contributors
  [2] issue_count         - Issues opened
  [3] pr_count            - Pull requests
  [4] star_count          - Stars received
  [5] watch_count         - Watchers
  [6] release_count       - Releases
  [7] fork_count          - Forks

Full Input:
  [
    [Q1: metric0, metric1, ..., metric7],  ← Quarter 1
    [Q2: metric0, metric1, ..., metric7],  ← Quarter 2
    [Q3: metric0, metric1, ..., metric7],  ← Quarter 3
    [Q4: metric0, metric1, ..., metric7]   ← Quarter 4 (most recent)
  ]


📋 Understanding the Output
─────────────────────────────────────────────────────────────────────────
The API returns:

1. predicted_metrics     - Forecasted values for next quarter (Q5)
2. activity_score        - Weighted sum of predicted metrics
3. activity_status       - "active" or "inactive"
4. confidence            - How confident is the prediction?
5. model_info            - Which model was used?

Classification Rule:
  ✓ active   → score ≥ 1319.5 (75th percentile threshold)
  ✓ inactive → score < 1319.5


📋 Interactive Swagger UI Features
─────────────────────────────────────────────────────────────────────────
Visit http://localhost:8000/docs to:

  1. Click "Try it out" on /predict endpoint
  2. Edit the JSON request body with your data
  3. Click "Execute"
  4. See the response with predicted activity status
  5. Experiment with different input patterns


📋 Common Use Cases
─────────────────────────────────────────────────────────────────────────

✓ Monitor repository health
  → Submit recent quarterly metrics
  → Get activity forecast and classification

✓ Identify at-risk projects
  → Track declining activity scores
  → Flag repositories becoming inactive

✓ Resource allocation
  → Predict which repos need more attention
  → Prioritize maintenance efforts

✓ Trend analysis
  → Compare predicted vs actual metrics
  → Identify growth or decline patterns


📋 Troubleshooting
─────────────────────────────────────────────────────────────────────────
Issue: "Model not found"
  → Check: ls models/checkpoints/gru_best.pt
  → Solution: Train the model first

Issue: "Feature stats not found"
  → Check: ls data/processed/timeseries/feature_stats.json
  → Solution: Run preprocessing pipeline

Issue: "Port already in use"
  → Solution: uvicorn api_service:app --port 8001

Issue: "Invalid input shape"
  → Ensure: 4 quarters × 8 metrics
  → Check: All quarters have exactly 8 values


📋 Quick Links
─────────────────────────────────────────────────────────────────────────
  API Docs:       http://localhost:8000/docs
  Health Check:   http://localhost:8000/health
  Model Info:     http://localhost:8000/model/info
  
  Code:           api_service.py
  Tests:          test_api_simple.py
  Guide:          API_REFERENCE.md
  Summary:        IMPLEMENTATION_COMPLETE.md


╔══════════════════════════════════════════════════════════════════════════╗
║  🚀 Ready to predict repository activity status!                         ║
║     Start with: ./start_api.sh                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
