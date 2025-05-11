"""
Test cases for the model API service.
"""
import os
import json
import pytest
import subprocess
import time
import requests
import numpy as np
import pickle


@pytest.fixture(scope="module")
def api_server():
    """Start the API server for testing."""
    # Skip if model doesn't exist yet
    if not os.path.exists('models/model.pkl'):
        pytest.skip("Model file not found")
    
    # Ensure model is copied to the deploy directory
    os.makedirs('deploy', exist_ok=True)
    if not os.path.exists('deploy/model.pkl') and os.path.exists('models/model.pkl'):
        import shutil
        shutil.copy('models/model.pkl', 'deploy/model.pkl')
    
    # Start server
    server = subprocess.Popen(
        ["python", "src/deploy/serve.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it time to start
    time.sleep(2)
    
    yield server
    
    # Cleanup
    server.terminate()


def test_health_endpoint(api_server):
    """Test the health endpoint."""
    try:
        response = requests.get('http://localhost:5000/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    except requests.exceptions.ConnectionError:
        pytest.fail("Could not connect to API server")


def test_prediction_endpoint(api_server):
    """Test the prediction endpoint."""
    # Load the model to compare results
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create test data - adjust feature count for your model
    feature_count = 3  # Change this to match your model's input dimensions
    test_features = np.random.rand(feature_count).tolist()
    
    # Get prediction from API
    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json={'features': test_features}
        )
        
        # Verify response
        assert response.status_code == 200
        api_prediction = response.json()['prediction']
        
        # Get prediction directly from model
        direct_prediction = model.predict(np.array([test_features])).tolist()
        
        # Compare results
        assert api_prediction == direct_prediction
    except requests.exceptions.ConnectionError:
        pytest.fail("Could not connect to API server")