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
import signal
import sys

@pytest.fixture(scope="module")
def api_server():
    """Start the API server for testing."""
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Ensure deploy directory exists
    os.makedirs('deploy', exist_ok=True)
    
    # Copy model file if needed
    if os.path.exists('models/model.pkl') and not os.path.exists('deploy/model.pkl'):
        import shutil
        print("Copying model to deploy directory")
        shutil.copy('models/model.pkl', 'deploy/model.pkl')
    
    # Start server in subprocess
    print("Starting API server for testing...")
    server_process = subprocess.Popen(
        [sys.executable, "src/deploy/serve.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server time to start
    time.sleep(5)
    
    # Check if the process is still running
    if server_process.poll() is not None:
        stdout, stderr = server_process.communicate()
        print("Server output:", stdout)
        print("Server error:", stderr)
        pytest.fail("Server process exited unexpectedly")
    
    # Check if server is responsive
    success = False
    for i in range(3):
        try:
            resp = requests.get("http://localhost:5100/", timeout=1)
            if resp.status_code == 200:
                success = True
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    
    if not success:
        server_process.terminate()
        pytest.fail("Could not connect to server after startup")
    
    yield server_process
    
    # Cleanup
    print("Stopping API server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
    
    print("API server stopped")

def test_health_endpoint(api_server):
    """Test the health endpoint."""
    try:
        response = requests.get('http://localhost:5100/health', timeout=2)
        print(f"Health response: {response.status_code} - {response.text}")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    except Exception as e:
        pytest.fail(f"Error testing health endpoint: {str(e)}")

def test_prediction_endpoint(api_server):
    """Test the prediction endpoint."""
    # Create test data
    test_features = [0.5, 0.5, 0.5]
    
    try:
        # Get prediction from API
        response = requests.post(
            'http://localhost:5100/predict',
            json={'features': test_features},
            timeout=2
        )
        
        print(f"Prediction response: {response.status_code} - {response.text}")
        
        # Verify response
        assert response.status_code == 200
        assert 'prediction' in response.json()
    except Exception as e:
        pytest.fail(f"Error testing prediction endpoint: {str(e)}")