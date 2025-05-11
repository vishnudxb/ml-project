"""
Simple Flask API for serving ML model predictions.
"""
import pickle
import numpy as np
import os
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model
MODEL_PATH = 'model.pkl'
if not os.path.exists(MODEL_PATH) and os.path.exists('../models/model.pkl'):
    MODEL_PATH = '../models/model.pkl'

print(f"Loading model from {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions."""
    # Get data from request
    data = request.get_json(force=True)
    features = data.get('features', [])
    
    # Validate input
    if not features or not isinstance(features, list):
        return jsonify({'error': 'Invalid input format. Expected "features" as list'}), 400
    
    # Make prediction
    try:
        features_array = np.array([features])
        prediction = model.predict(features_array).tolist()
        prediction_proba = None
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_array).tolist()
        
        response = {
            'prediction': prediction,
            'prediction_proba': prediction_proba
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    print("Starting model serving API on http://localhost:5000")
    print("Try it with: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\"features\": [0.5, 0.1, 0.2]}'")
    app.run(host='0.0.0.0', port=5000)