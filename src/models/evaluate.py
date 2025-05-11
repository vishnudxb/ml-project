"""
Model evaluation script for ML pipeline.
Evaluates a trained model and produces performance metrics.
"""
import argparse
import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model_file, data_file, output_metrics_file, threshold=0.5):
    """Evaluate a trained model and save metrics to disk."""
    print(f"Evaluating model {model_file} on data {data_file}")
    
    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load data
    df = pd.read_csv(data_file)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, average='weighted', zero_division=0)),
    }
    
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = dict(zip(X.columns, importances.tolist()))
        metrics["feature_importance"] = feature_importance
    
    # Save metrics
    with open(output_metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary
    print(f"Evaluation complete. Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # Check if model meets threshold
    meets_threshold = metrics['accuracy'] >= threshold
    if not meets_threshold:
        print(f"WARNING: Model accuracy {metrics['accuracy']:.4f} is below threshold {threshold}")
    
    print(f"Metrics saved to {output_metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ML model')
    parser.add_argument('--model', required=True, help='Input model file')
    parser.add_argument('--data', required=True, help='Input data file')
    parser.add_argument('--output', required=True, help='Output metrics file')
    parser.add_argument('--threshold', type=float, default=0.75, help='Minimum accuracy threshold')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output, args.threshold)