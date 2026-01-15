#!/usr/bin/env python3
"""
Quick test script for the API service
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("Testing /model/info endpoint")
    print("="*60)
    
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


def test_prediction_active():
    """Test prediction with active repository pattern."""
    print("\n" + "="*60)
    print("Testing /predict endpoint - Active Repository")
    print("="*60)
    
    # High activity pattern
    data = {
        "sequence": [
            [200.0, 80.0, 150.0, 120.0, 180.0, 70.0, 10.0, 2.0],
            [220.0, 85.0, 160.0, 130.0, 200.0, 75.0, 12.0, 3.0],
            [210.0, 82.0, 155.0, 125.0, 190.0, 72.0, 11.0, 2.0],
            [230.0, 88.0, 165.0, 135.0, 210.0, 78.0, 13.0, 3.0]
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print(f"\n📊 Summary:")
    print(f"  Activity Status: {result['activity_status'].upper()}")
    print(f"  Activity Score: {result['activity_score']:.2f}")
    print(f"  Confidence: {result['confidence']['classification']}")


def test_prediction_inactive():
    """Test prediction with inactive repository pattern."""
    print("\n" + "="*60)
    print("Testing /predict endpoint - Inactive Repository")
    print("="*60)
    
    # Low activity pattern
    data = {
        "sequence": [
            [10.0, 5.0, 8.0, 3.0, 15.0, 8.0, 1.0, 0.0],
            [12.0, 6.0, 10.0, 4.0, 18.0, 9.0, 1.0, 0.0],
            [9.0, 5.0, 7.0, 3.0, 16.0, 8.0, 0.0, 0.0],
            [11.0, 6.0, 9.0, 4.0, 17.0, 9.0, 1.0, 0.0]
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print(f"\n📊 Summary:")
    print(f"  Activity Status: {result['activity_status'].upper()}")
    print(f"  Activity Score: {result['activity_score']:.2f}")
    print(f"  Confidence: {result['confidence']['classification']}")


def test_prediction_moderate():
    """Test prediction with moderate repository pattern."""
    print("\n" + "="*60)
    print("Testing /predict endpoint - Moderate Repository")
    print("="*60)
    
    # Moderate activity pattern (near threshold)
    data = {
        "sequence": [
            [100.0, 50.0, 75.0, 60.0, 80.0, 40.0, 5.0, 1.0],
            [120.0, 55.0, 80.0, 65.0, 90.0, 42.0, 6.0, 1.0],
            [110.0, 52.0, 78.0, 62.0, 85.0, 41.0, 5.0, 1.0],
            [130.0, 58.0, 85.0, 70.0, 95.0, 45.0, 7.0, 2.0]
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2))
    
    print(f"\n📊 Summary:")
    print(f"  Activity Status: {result['activity_status'].upper()}")
    print(f"  Activity Score: {result['activity_score']:.2f}")
    print(f"  Confidence: {result['confidence']['classification']}")


def test_invalid_input():
    """Test error handling with invalid input."""
    print("\n" + "="*60)
    print("Testing /predict endpoint - Invalid Input")
    print("="*60)
    
    # Wrong shape
    data = {
        "sequence": [
            [100.0, 50.0, 75.0],  # Only 3 features instead of 8
        ]
    }
    
    response = requests.post(f"{API_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("\n")
    print("🚀 Repository Activity Prediction API - Test Suite")
    print("="*60)
    print(f"API URL: {API_URL}")
    print("="*60)
    
    try:
        # Test all endpoints
        test_health()
        test_model_info()
        test_prediction_active()
        test_prediction_inactive()
        test_prediction_moderate()
        test_invalid_input()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        print("\n💡 Try the interactive Swagger UI:")
        print(f"   {API_URL}/docs")
        print("\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("   Make sure the API is running:")
        print("   uvicorn api_service:app --reload")
        print()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
