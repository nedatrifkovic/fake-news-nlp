# src/features.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import os

# Load dataset
df = pd.read_csv("data/processed/processed.csv")
texts = df['lemmatized_text'].tolist()
labels = df['label'].values

# TF-IDF
def get_tfidf_features(texts, max_features=5000):
    """Generate TF-IDF features for a list of texts"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer

tfidf_features, tfidf_vectorizer = get_tfidf_features(texts)
print(f"TF-IDF shape: {tfidf_features.shape}")

# Save TF-IDF vectorizer
import pickle
with open("data/processed/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Word2Vec
def get_word2vec_features(texts, vector_size=100, window=5, min_count=1):
    """Train Word2Vec and return average vectors per document"""
    tokenized_texts = [text.split() for text in texts]
    w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)
    
    features = []
    for tokens in tqdm(tokenized_texts, desc="Word2Vec vectorizing"):
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features), w2v_model

w2v_features, w2v_model = get_word2vec_features(texts)
print(f"Word2Vec shape: {w2v_features.shape}")

# Save Word2Vec model
w2v_model.save("data/processed/word2vec.model")

# BERT
def get_bert_features(texts, model_name='bert-base-uncased', max_length=128):
    """Generate BERT embeddings for a list of texts"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    features = []
    for text in tqdm(texts, desc="BERT vectorizing"):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token
        features.append(cls_embedding)
    return np.array(features), tokenizer, model

# Warning: BERT can be slow on CPU
# Uncomment to run if you have GPU or small dataset
# bert_features, bert_tokenizer, bert_model = get_bert_features(texts)
# print(f"BERT shape: {bert_features.shape}")

# Optional: Save BERT tokenizer and model
# bert_tokenizer.save_pretrained("data/processed/bert_tokenizer")
# bert_model.save_pretrained("data/processed/bert_model")

print("Feature extraction complete!")
