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

## Dataset
The dataset is **not included in this repository** due to size limitations.  
You must download it manually from Kaggle:

- [Fake News Dataset (fake.csv)](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv)  
- [True News Dataset (true.csv)](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv)  

Place the files in:  
- `fake.csv` → `data/raw/`  
- `true.csv` → `data/raw/`  

*Dataset last accessed: September 2025*  

For more details about the data directory structure, see [`data/README.md`](data/README.md).

## Data Preparation

1. After downloading raw data from Kaggle
2. Run the **EDA notebook** to generate the **interim dataset**:  

   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
3. Run the **preprocessing script** to generate the **processed dataset** ready for modeling:

   ```bash
   uv run python src/preprocessing.py
   ```

## Documentation

For more details about the project structure and data flow, see our documentation:

- [Data Pipeline](docs/data_pipeline.md)



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

