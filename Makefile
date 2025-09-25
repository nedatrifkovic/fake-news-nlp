# Fake News NLP - Makefile
# Easy commands for project management

.PHONY: help install setup docker-build docker-run docker-setup clean test

# Default target
help:
	@echo "Fake News NLP - Available Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Run complete pipeline (after downloading data)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container with complete pipeline"
	@echo "  make docker-setup - Setup Docker environment"
	@echo "  make clean       - Clean generated files"
	@echo "  make test        - Test prediction"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Download datasets from Kaggle to data/raw/"
	@echo "  - Place fake.csv and true.csv in data/raw/"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo " Dependencies installed!"

# Run complete pipeline
setup:
	@echo " Running complete pipeline..."
	chmod +x setup.sh
	./setup.sh

# Build Docker image
docker-build:
	@echo " Building Docker image..."
	docker build -t fake-news-nlp .
	@echo " Docker image built!"

# Run Docker with complete pipeline
docker-run:
	@echo " Running Docker with complete pipeline..."
	docker run -it --rm -v $(PWD)/data:/app/data fake-news-nlp ./setup.sh

# Setup Docker environment
docker-setup:
	@echo " Setting up Docker environment..."
	chmod +x docker-setup.sh
	./docker-setup.sh

# Clean generated files
clean:
	@echo " Cleaning generated files..."
	rm -rf data/interim/*.csv
	rm -rf data/processed/*.csv
	rm -rf models/ml_models/*.pkl
	rm -rf models/vectorizers/*.pkl
	rm -rf models/vectorizers/*.model
	rm -rf models/vectorizers/*.npy
	rm -rf reports/images/*.png
	@echo " Cleaned!"

# Test prediction
test:
	@echo " Testing prediction..."
	PYTHONPATH=src uv run python src/predict.py \
		--text "Breaking news: Government announces new reforms." \
		--model logreg_tfidf.pkl \
		--feature tfidf
	@echo " Test completed!"
