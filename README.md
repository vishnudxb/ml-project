# ML-Project: Beginner-Friendly ML CI/CD Pipeline

This repository contains a simple, local-friendly CI/CD pipeline for machine learning projects. It demonstrates how to manage the ML lifecycle from data processing to model deployment without requiring cloud services.

# Project Overview

This pipeline addresses key ML CI/CD challenges:

- Data and model versioning (using DVC)
- Non-deterministic outcomes (using fixed seeds)
- Computationally expensive tests (through selective testing)
- Model-specific metrics (via custom evaluation)

# Essential Files

- Makefile: The central component that orchestrates the entire pipeline
- dvc.yaml: Defines the DVC pipeline stages
- src/data/preprocess.py: Transforms raw data into processed features
- src/data/validation.py: Validates data quality
- src/models/train.py: Trains the model with reproducible settings
- src/models/evaluate.py: Evaluates model performance
- src/deploy/serve.py: Deploys model as a local Flask API
- tests/: Contains tests for each pipeline component

# Getting Started

## Prerequisites

- Python 3.8+
- Git (recommended, but optional)

### Setup

1. Clone the repository (or download if not using Git)

```

git clone https://github.com/vishnudxb/ml-project
cd ml-project

```

2. Install dependencies

`pip install -r requirements.txt`

3. Initialize the project

`make setup` 

4. Add your dataset

- Place your CSV file as data/raw.csv
- For this example pipeline, it should have columns named 'feature_1', 'feature_2', and 'target'
- Track it with DVC: `dvc add data/raw.csv`

### Using the Pipeline

#### Run the complete pipeline

`make all`

This executes the entire workflow: data validation → training → evaluation → API testing.

#### Run individual stages

```

make data-validation  # Validate data quality
make train            # Train the model
make evaluate         # Evaluate model performance
make test-api         # Test the API
make deploy           # Deploy as a local Flask API

```

### Understanding the Workflow

#### 1. Data Processing:

- `preprocess.py` handles missing values, encoding, and feature scaling
- `validation.py` ensures data meets quality standards


#### 2. Model Training:

- `train.py` creates a model with fixed random seeds for reproducibility
- The model is saved to `models/model.pkl`


#### 3. Model Evaluation:

`evaluate.py` assesses model quality using multiple metrics
Results are saved to `metrics/metrics.json`


#### 4. Model Serving:

`serve.py` provides a simple Flask API
Test it with: `curl -X POST http://localhost:5100/predict -H "Content-Type: application/json" -d '{"features": [0.5, 0.1, 0.2]}'`


### Optional Components

- `MLproject`: Used if you want to integrate with MLflow
- `.github/workflows/ci.yml`: Add this if you want GitHub-based CI/CD
- DVC remote storage: Configure a remote store if working with a team

### Customizing for Your Project

#### 1. Adjust the data processing:

- Modify `preprocess.py` for your specific data needs
- Update column names in `test_data.py`


#### 2.Change the model:

Edit `train.py` to use a different algorithm
Update feature count in tests


#### 3. Modify evaluation metrics:

Customize metrics in `evaluate.py`
Adjust thresholds in tests

### Troubleshooting

- Tests failing? Check if your data matches the expected schema
- API not working? Ensure model.pkl is copied to deploy/
- DVC issues? Try dvc init --no-scm if not using Git

### Next Steps

Once comfortable with this local pipeline, consider:

1. Adding a remote DVC storage
2. Implementing GitHub Actions for CI/CD
3. Deploying to cloud services

This beginner-friendly approach lets you master ML CI/CD fundamentals before moving to more complex, cloud-based solutions.
