"""
Data preprocessing script for ML pipeline.
Takes raw data and prepares it for model training.
"""
import argparse
import pandas as pd
import numpy as np


def preprocess_data(input_file, output_file):
    """Preprocess the raw data file and save the processed version."""
    print(f"Preprocessing data from {input_file} to {output_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Basic preprocessing steps
    # 1. Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # 2. Convert categorical features
    for column in df.select_dtypes(include=['object']).columns:
        if column != 'target':  # Skip target column if it's categorical
            df[column] = pd.factorize(df[column])[0]
    
    # 3. Feature scaling
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if column != 'target':  # Skip target column
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessing complete. Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for ML pipeline')
    parser.add_argument('--input', required=True, help='Input raw data file')
    parser.add_argument('--output', required=True, help='Output processed data file')
    
    args = parser.parse_args()
    preprocess_data(args.input, args.output)