# Data Directory

This folder stores all datasets used in the **Fake News NLP** project.

## Structure
- `data/raw/` → Original files downloaded from external sources (no modifications).  
- `data/interim/` → Intermediate data created during preprocessing (e.g., cleaned text, merged datasets).  
- `data/processed/` → Final datasets ready for modeling.  

## Download Instructions
Data files are **not included in this repository** due to size limitations.  
Please download them manually from Kaggle:  

[Fake News Detection Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=fake.csv)
[True News Detection Dataset](https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?select=true.csv)

After downloading, place the files in the following locations:
- `fake.csv` → `data/raw/`  
- `true.csv` → `data/raw/`  

## Notes
- The entire `data/` folder is **excluded from version control** (see `.gitignore`).  
- You need to download the files manually before running any notebooks or training scripts.  
