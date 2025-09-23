# src/predict.py

import argparse
import pickle
import numpy as np
from gensim.models import Word2Vec
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Predict fake/real news")
parser.add_argument("--text", type=str, help="Single news text for prediction")
parser.add_argument(
    "--input_file", type=str, help="Path to a text file with news articles"
)
parser.add_argument(
    "--model", 
    type=str, 
    required=True, 
    help="Model filename from models/ml_models/"
)
parser.add_argument(
    "--feature",
    type=str,
    choices=["tfidf", "w2v"],
    required=True,
    help="Feature type used by the model",
)
args = parser.parse_args()


# Load models and vectorizers
ML_MODEL_PATH = os.path.join("models/ml_models", args.model)

if args.feature == "tfidf":
    with open("models/vectorizers/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
elif args.feature == "w2v":
    w2v_model = Word2Vec.load("models/vectorizers/word2vec.model")

with open(ML_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Prepare input texts
if args.text:
    texts = [args.text]
elif args.input_file:
    with open(args.input_file) as f:
        texts = [line.strip() for line in f if line.strip()]
else:
    text = input("Enter news text for prediction: ")
    texts = [text]


# Feature extraction
def get_w2v_features(texts, model, vector_size=100):
    features = []
    for text in texts:
        tokens = text.split()
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features)


if args.feature == "tfidf":
    X = vectorizer.transform(texts)
elif args.feature == "w2v":
    X = get_w2v_features(texts, w2v_model)


# Predict
preds = model.predict(X)
for text, pred in zip(texts, preds):
    label = "FAKE" if pred == 0 else "REAL"
    print(f"\nText: {text}\nPrediction: {label}")
