"""
Data validation utilities for ML pipeline.
Used to ensure data quality before model training.
"""
import pandas as pd
import numpy as np


def validate_schema(data, required_columns):
    """Validate that data contains all required columns."""
    return all(column in data.columns for column in required_columns)


def check_for_nulls(data, columns):
    """Check that specified columns have no null values."""
    return not data[columns].isnull().any().any()


def check_for_outliers(data, columns, z_threshold=3.0):
    """
    Check for outliers in specified columns using Z-score method.
    Returns True if no extreme outliers are found.
    """
    for column in columns:
        if data[column].dtype.kind in 'ifc':  # integer, float, complex
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            if (z_scores > z_threshold).sum() / len(data) > 0.05:  # More than 5% outliers
                return False
    return True


def validate_data(data_file, schema=None):
    """Complete data validation workflow."""
    data = pd.read_csv(data_file)
    
    # Default schema if none provided
    if schema is None:
        schema = data.columns.tolist()
    
    # Run validation checks
    schema_valid = validate_schema(data, schema)
    nulls_valid = check_for_nulls(data, schema)
    outliers_valid = check_for_outliers(data, 
                                       [c for c in schema if data[c].dtype.kind in 'ifc'])
    
    # Return overall validation result
    return schema_valid and nulls_valid and outliers_valid