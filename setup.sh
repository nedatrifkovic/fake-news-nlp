#!/bin/bash

# Fake News NLP - Complete Setup Script
# This script downloads data, runs EDA, preprocessing, training, and testing

set -e  # Exit on any error

echo "Starting Fake News NLP Pipeline Setup..."

# Check if data exists
if [ ! -f "data/raw/fake.csv" ] || [ ! -f "data/raw/true.csv" ]; then
    echo "Data files not found!"
    echo "Please download the datasets from Kaggle:"
    echo "   - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv"
    echo "   - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv"
    echo "   Place them in: data/raw/"
    echo ""
    echo "💡 Or run this script after placing the data files."
    exit 1
fi

echo "Data files found!"

# Step 1: Run EDA notebook
echo "Step 1: Running EDA notebook..."
uv run python -c "
import subprocess
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Load and execute the notebook
with open('notebooks/eda.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

print('EDA notebook completed successfully!')
"

# Step 2: Run preprocessing
echo "Step 2: Running preprocessing..."
uv run python src/preprocessing.py
echo "Preprocessing completed!"

# Step 3: Generate features
echo " Step 3: Generating features..."
uv run python src/features.py
echo "Features generated!"

# Step 4: Train models
echo "Step 4: Training models..."
PYTHONPATH=src uv run python src/train.py
echo "Models trained!"

# Step 5: Test prediction
echo "Step 5: Testing prediction..."
echo "Testing with sample text: 'Breaking news: Government announces new reforms.'"
PYTHONPATH=src uv run python src/predict.py \
  --text "Breaking news: Government announces new reforms." \
  --model logreg_tfidf.pkl \
  --feature tfidf

echo ""
echo " Complete pipeline executed successfully!"
echo " Check the following directories for outputs:"
echo "   - data/interim/ (EDA output)"
echo "   - data/processed/ (preprocessed data)"
echo "   - models/ml_models/ (trained models)"
echo "   - models/vectorizers/ (feature extractors)"
echo ""
echo " You can now run predictions using:"
echo "   PYTHONPATH=src uv run python src/predict.py --help"
