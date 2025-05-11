"""
Test cases for the trained model.
"""
import os
import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score


def test_model_file_exists():
    """Test that model file exists."""
    assert os.path.exists('models/model.pkl'), "Model file not found"
    assert os.path.getsize('models/model.pkl') > 0, "Model file is empty"


def test_model_performance():
    """Test that model performance is above the threshold."""
    # Skip if model or processed data doesn't exist yet
    if not os.path.exists('models/model.pkl') or not os.path.exists('data/processed.csv'):
        pytest.skip("Model or processed data file not found")
    
    # Load the model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    data = pd.read_csv('data/processed.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # Assert performance threshold (adjust based on your expectation)
    assert accuracy >= 0.7, f"Model accuracy {accuracy} below threshold 0.7"


def test_model_stability():
    """Test model stability by running multiple predictions with seed."""
    # Skip if model doesn't exist yet
    if not os.path.exists('models/model.pkl'):
        pytest.skip("Model file not found")
    
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Create consistent test data
    np.random.seed(42)
    # Adjust feature count based on your actual model
    feature_count = 3  # Change this to match your model's input dimensions
    test_data = np.random.rand(10, feature_count)
    
    # Run predictions multiple times to check for stability
    predictions = []
    for _ in range(3):
        preds = model.predict(test_data)
        predictions.append(preds)
    
    # Check that all predictions are the same
    for i in range(1, len(predictions)):
        assert np.array_equal(predictions[0], predictions[i]), "Model produces non-deterministic results"