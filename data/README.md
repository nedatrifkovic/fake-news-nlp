# Data Directory

This folder stores all datasets used in the **Fake News NLP** project.

## Structure
- `data/raw/` → Original files downloaded from external sources (unmodified)  
- `data/interim/` → Intermediate data generated during preprocessing (e.g., cleaned text, merged datasets)  
- `data/processed/` → Final datasets ready for modeling  
- `data/samples/` → Sample text files for prediction testing (each line is a separate news text)  

## Download Instructions
⚠️ Data files are **not included in this repository** due to size limitations.  
You need to download them manually from Kaggle:

- [Fake News Dataset (fake.csv)](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv)  
- [True News Dataset (true.csv)](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv)  

After downloading, place the files in the following locations:
- `fake.csv` → `data/raw/`  
- `true.csv` → `data/raw/`  

## Samples Folder
The `data/samples/` folder contains text files for testing predictions:
- Each line in the file represents a separate news text for prediction
- Example: `sample_texts.txt` contains multiple news articles, one per line
- You can create your own sample files by adding news texts, each on a new line
- Use these files with the prediction script: `uv run python src/predict.py --input_file data/samples/your_file.txt --model model_name.pkl --feature feature_type`

## Notes
- The entire `data/` folder is excluded from version control (see `.gitignore`).  
- You must download the datasets manually before running any notebooks or training scripts.  
- Dataset last accessed: **September 2025** (please check Kaggle for updates).  
