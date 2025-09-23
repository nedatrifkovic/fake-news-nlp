# Imports
import pandas as pd
import spacy
from tqdm import tqdm
import re
import numpy as np

# nltk stopwords
import nltk

nltk.data.path.append("nltk_data")  # lokalni folder za nltk
try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir="nltk_data")
    STOPWORDS = set(stopwords.words("english"))

# Load spaCy English model (disable parser and ner for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# Cleaning and preprocessing functions
def clean_text(text: str) -> str:
    """Lowercase, remove non-letter characters, remove stopwords"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # remove non-letters
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS]
    return " ".join(filtered_words)


def lemm_text(text: str) -> str:
    """Lemmatize text using spaCy"""
    doc = nlp(text)
    lemmatized_words = [
        token.lemma_ for token in doc if token.lemma_ not in STOPWORDS
    ]
    return " ".join(lemmatized_words)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning, lemmatization, and normalize subject"""
    tqdm.pandas()

    # Ensure combined column exists
    if "title_and_text" not in df.columns:
        df["title_and_text"] = df["title"].fillna('') + " " + df["text"].fillna('')

    # Apply cleaning
    df["cleaned_text"] = df["title_and_text"].progress_apply(clean_text)

    # Apply lemmatization
    df["lemmatized_text"] = df["cleaned_text"].progress_apply(lemm_text)

   # Fill NaN with empty string first
    df['lemmatized_text'] = df['lemmatized_text'].fillna('')

    # Handle empty lemmatized_text
    empty_rows = df[df['lemmatized_text'].str.strip() == '']
    if len(empty_rows) > 0:
        print(f"Found {len(empty_rows)} empty lemmatized_text rows. Dropping them.")
        df['lemmatized_text'].replace('', np.nan, inplace=True)
        df.dropna(subset=['lemmatized_text'], inplace=True)
    else:
        print("No empty lemmatized_text rows found.")


    # Normalize subject
    if "subject" in df.columns:
        df["subject"] = (
            df["subject"]
            .str.lower()
            .str.replace(r"[^a-z]", "", regex=True)
            .replace(
                {
                    "politicsnews": "politics",
                    "governmentnews": "government",
                    "usnews": "us",
                    "middleeast": "middleeast",
                }
            )
        )

    return df

# Main execution
if __name__ == "__main__":
    print("Running preprocessing on interim dataset...")

    # Load interim dataset
    df = pd.read_csv("data/interim/interim.csv")

    # Apply preprocessing
    df = preprocess_dataframe(df)
    
    # Save processed dataset
    processed_path = "data/processed/processed.csv"
    df_final = df[["title_and_text", "lemmatized_text", "subject", "label"]]
    df_final.to_csv(processed_path, index=False)

    print(f"Processed dataset saved to {processed_path}")
    print("Preprocessing complete.")
