"""
Model training script for ML pipeline.
Trains a model on processed data and saves it to disk.
"""
import argparse
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(data_file, output_model_file, random_seed=42):
    """Train a machine learning model and save it to disk."""
    print(f"Training model using {data_file}")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    # Train model (using RandomForest as an example)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_seed
    )
    model.fit(X_train, y_train)
    
    # Print training metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Save model
    with open(output_model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML model')
    parser.add_argument('--data', required=True, help='Input processed data file')
    parser.add_argument('--output', required=True, help='Output model file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    train_model(args.data, args.output, args.seed)