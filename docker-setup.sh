#!/bin/bash

# Docker Setup Script for Fake News NLP
# This script builds the Docker image and runs the complete pipeline

set -e

echo "Docker Setup for Fake News NLP"

# Build Docker image
echo "Building Docker image..."
docker build -t fake-news-nlp .

echo "Docker image built successfully!"

# Create data directory if it doesn't exist
mkdir -p data/raw

echo ""
echo " Next steps:"
echo "1. Download the datasets from Kaggle:"
echo "   - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv"
echo "   - https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv"
echo "   - Place them in: data/raw/"
echo ""
echo "2. Run the complete pipeline:"
echo "   docker run -it --rm -v \$(pwd)/data:/app/data fake-news-nlp ./setup.sh"
echo ""
echo "3. Or run individual steps:"
echo "   # EDA notebook"
echo "   docker run -it --rm -v \$(pwd)/data:/app/data -p 8888:8888 fake-news-nlp uv run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "   # Preprocessing"
echo "   docker run -it --rm -v \$(pwd)/data:/app/data fake-news-nlp uv run python src/preprocessing.py"
echo ""
echo "   # Training"
echo "   docker run -it --rm -v \$(pwd)/data:/app/data fake-news-nlp PYTHONPATH=src uv run python src/train.py"
echo ""
echo "   # Prediction"
echo "   docker run -it --rm -v \$(pwd)/data:/app/data fake-news-nlp PYTHONPATH=src uv run python src/predict.py --text 'Your news text here' --model logreg_tfidf.pkl --feature tfidf"
