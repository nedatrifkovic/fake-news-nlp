data/raw/
 ├── fake.csv
 └── true.csv
        │
        ▼
notebooks/eda.ipynb
  - Concatenate fake + true
  - Drop duplicates & nulls
  - Save interim dataset
        │
        ▼
data/interim/interim.csv
  Columns: title, text, subject, date, label, title_and_text
        │
        ▼
src/preprocessing.py
  - Remove stopwords
  - Lowercase
  - Lemmatization
  - Combine into new column
        │
        ▼
data/processed/processed.csv
  Columns: title_and_text, subject, cleaned_text, lemmatized_text, label
        │
        ▼
src/features.py  (Feature Engineering)
  - TF-IDF / CountVectorizer on lemmatized_text
  - Word embeddings (Word2Vec/Gensim)
        │
        ▼
src/train.py
  - Split data
  - Train models (Logistic Regression, RandomForest, etc.)
  - Evaluate
        │
        ▼
Models & reports
  - Save trained models
  - Visualizations & metrics
