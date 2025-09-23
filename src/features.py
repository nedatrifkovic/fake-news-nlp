# src/features.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import os
import pickle

# Create directories if they don't exist
os.makedirs("models/vectorizers", exist_ok=True)

# Load processed dataset
df = pd.read_csv("data/processed/processed.csv")
texts = df["lemmatized_text"].tolist()
labels = df["label"].values


# TF-IDF
def get_tfidf_features(texts, max_features=5000):
    """Generate TF-IDF features for a list of texts"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer


tfidf_features, tfidf_vectorizer = get_tfidf_features(texts)
print(f"TF-IDF shape: {tfidf_features.shape}")
with open("models/vectorizers/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)


# Word2Vec
def get_word2vec_features(texts, vector_size=100, window=5, min_count=1):
    """Train Word2Vec and return average vectors per document"""
    tokenized_texts = [text.split() for text in texts]
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
    )

    features = []
    for tokens in tqdm(tokenized_texts, desc="Word2Vec vectorizing"):
        vectors = []
        for word in tokens:
            if word in w2v_model.wv:
                vectors.append(w2v_model.wv[word])
        if len(vectors) > 0:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features), w2v_model


w2v_features, w2v_model = get_word2vec_features(texts)
print(f"Word2Vec shape: {w2v_features.shape}")
w2v_model.save("models/vectorizers/word2vec.model")

print("Feature extraction complete!")
