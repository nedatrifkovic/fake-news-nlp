# Fake News Detection (NLP Project)

## Overview
This project aims to classify news articles as **fake** or **real** using Natural Language Processing (NLP) techniques and machine learning models.  
It includes data preprocessing, feature extraction, model training, and evaluation.

## Tech Stack
- Python, uv
- pandas, scikit-learn
- nltk / spaCy
- gensim
- matplotlib, seaborn

## Project Structure
- `data/` → raw and processed datasets
- `notebooks/` → Jupyter notebook for EDA
- `src/` → main Python source code (preprocessing, features, models, training, prediction)
- `reports/` → results and visualizations

## How to Run
1. Clone this repo  
   ```bash
   git clone https://github.com/yourusername/fake-news-nlp.git
   cd fake-news-nlp
   ```
2. Create a virtual environment and install dependencies   
   ```bash
    uv venv
    source .venv/bin/activate
    uv sync
   ``` 
3. Run training
```bash
uv run python src/train.py
```   
4. Run prediction
```bash
uv run python src/predict.py "Your news text here"
```

