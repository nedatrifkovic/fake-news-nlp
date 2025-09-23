# src/models.py

import pickle
import os
from gensim.models import Word2Vec
import numpy as np


# Paths
VECTORIZERS_DIR = "models/vectorizers"
ML_MODELS_DIR = "models/ml_models"

# Vectorizer loaders
def load_tfidf_vectorizer(filename="tfidf_vectorizer.pkl"):
    path = os.path.join(VECTORIZERS_DIR, filename)
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def load_word2vec_model(filename="word2vec.model"):
    path = os.path.join(VECTORIZERS_DIR, filename)
    model = Word2Vec.load(path)
    return model


# ML model loader
def load_ml_model(filename):
    path = os.path.join(ML_MODELS_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# Helper: average Word2Vec vectors
def get_w2v_features(texts, model, vector_size=100):
    """
    Compute averaged Word2Vec vectors for a list of texts.
    """
    features = []
    for text in texts:
        tokens = text.split()
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features)
