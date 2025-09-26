# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY docs/ ./docs/
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY setup.sh docker-setup.sh Makefile ./

# Install Python dependencies
RUN uv sync --frozen

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed data/samples
RUN mkdir -p models/ml_models models/vectorizers
RUN mkdir -p reports/images

# Make scripts executable
RUN chmod +x setup.sh docker-setup.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV NLTK_DATA=/app/nltk_data

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command
CMD ["uv", "run", "python", "src/preprocessing.py"]
