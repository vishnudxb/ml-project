# Makefile for local ML CI/CD pipeline

.PHONY: setup data-validation train evaluate test-api deploy all clean

# First-time setup
setup:
	pip install -r requirements.txt
	dvc init --no-scm
	dvc config core.analytics false
	mkdir -p /tmp/dvc-storage
	dvc remote add -d localremote /tmp/dvc-storage
	mkdir -p data models metrics

# Data preparation and validation
data-validation:
	python3 -m pytest tests/test_data.py -v

# Model training stage
train: data-validation
	dvc repro train
	
# Model evaluation stage  
evaluate: train
	mkdir -p metrics
	dvc repro evaluate
	python3 -m pytest tests/test_model.py -v
	@echo "Model metrics:"
	@cat metrics/metrics.json

# Test API endpoint
test-api: evaluate
	python3 -m pytest tests/test_api.py -v

# Deploy model locally as a Flask API
deploy: test-api
	mkdir -p deploy
	cp models/model.pkl deploy/
	cd deploy && python3 ../src/deploy/serve.py

# Run entire pipeline
all: setup data-validation train evaluate test-api

# Clean up
clean:
	rm -rf metrics/*.json
	rm -rf deploy/*.pkl