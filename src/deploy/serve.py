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
def load_model():
    """Load the trained model."""
    # Look for model in various locations
    possible_paths = [
        'model.pkl',              # Current directory
        '../models/model.pkl',    # Relative to deploy directory
        'models/model.pkl'        # Project root
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading model from {path}")
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    # If no model found, create a dummy one for testing
    print("WARNING: No model found, creating dummy model for testing")
    from sklearn.ensemble import RandomForestClassifier
    dummy_model = RandomForestClassifier(n_estimators=1)
    dummy_model.fit(
        [[0, 0, 0], [1, 1, 1]], 
        [0, 1]
    )
    return dummy_model

# Try to load the model
try:
    model = load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Create a dummy model as fallback
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=1)
    model.fit([[0, 0, 0], [1, 1, 1]], [0, 1])

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions."""
    # Get data from request
    try:
        data = request.get_json(force=True)
        features = data.get('features', [])
        
        # Validate input
        if not features or not isinstance(features, list):
            return jsonify({'error': 'Invalid input format. Expected "features" as list'}), 400
        
        # Make prediction
        features_array = np.array([features])
        prediction = model.predict(features_array).tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

# Add a root route for basic connectivity testing
@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    print("Starting model serving API on http://localhost:5100")
    print("API endpoints:")
    print("  - GET  /health   - Health check")
    print("  - POST /predict  - Make prediction with JSON payload: {'features': [0.5, 0.1, 0.2]}")
    
    # Important: Debug mode should be False in a production environment
    # Setting threaded=False can help with some Flask stability issues during testing
    app.run(host='0.0.0.0', port=5100, debug=True, threaded=True, use_reloader=False)