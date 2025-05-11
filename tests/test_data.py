"""
Test cases for data validation.
"""
import os
import pandas as pd
import pytest
from src.data.validation import validate_schema, check_for_nulls, check_for_outliers


def test_data_file_exists():
    """Test that data file exists."""
    assert os.path.exists('data/raw.csv'), "Raw data file not found"
    if os.path.exists('data/processed.csv'):
        assert os.path.getsize('data/processed.csv') > 0, "Processed data file is empty"


def test_data_schema():
    """Test that the data conforms to expected schema."""
    # Skip if processed data doesn't exist yet
    if not os.path.exists('data/processed.csv'):
        pytest.skip("Processed data file not found")
    
    data = pd.read_csv('data/processed.csv')
    # Adjust these columns based on your actual dataset
    required_columns = ['feature_1', 'feature_2', 'target']
    assert validate_schema(data, required_columns)


def test_no_missing_values():
    """Test that there are no missing values in critical columns."""
    # Skip if processed data doesn't exist yet
    if not os.path.exists('data/processed.csv'):
        pytest.skip("Processed data file not found")
    
    data = pd.read_csv('data/processed.csv')
    # Adjust these columns based on your actual dataset
    critical_columns = ['feature_1', 'feature_2', 'target']
    assert check_for_nulls(data, critical_columns)


def test_outlier_detection():
    """Test that outliers are within acceptable range."""
    # Skip if processed data doesn't exist yet
    if not os.path.exists('data/processed.csv'):
        pytest.skip("Processed data file not found")
    
    data = pd.read_csv('data/processed.csv')
    # Adjust these columns based on your actual dataset
    outlier_columns = ['feature_1', 'feature_2']
    assert check_for_outliers(data, outlier_columns, z_threshold=3.0)