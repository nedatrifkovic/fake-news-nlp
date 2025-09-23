# Work in progress


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

The prediction script supports three different ways to make predictions:

### Option 1: Direct text input from command line
```bash
uv run python src/predict.py \
  --text "Breaking news: Government announces new reforms." \
  --model logreg_tfidf.pkl \
  --feature tfidf
```

### Option 2: Predict from a text file
Create a text file in `data/samples/` directory with one news text per line:
```bash
# Create the samples directory if it doesn't exist
mkdir -p data/samples

# Create a sample file with multiple texts (one per line)
echo "Stock markets rise after positive jobs report." > data/samples/sample_texts.txt
echo "Scientists discover cure for aging in lab tests." >> data/samples/sample_texts.txt
```

Then run prediction:
```bash
uv run python src/predict.py \
  --input_file data/samples/sample_texts.txt \
  --model xgb_tfidf.pkl \
  --feature tfidf
```

### Option 3: Interactive input (no arguments)
```bash
uv run python src/predict.py --model logreg_tfidf.pkl --feature tfidf
```
The script will prompt you to enter the news text for prediction.

### Available Models and Features
- **Models**: `logreg_tfidf.pkl`, `rf_w2v.pkl`, `xgb_tfidf.pkl`
- **Features**: `tfidf`, `w2v`

