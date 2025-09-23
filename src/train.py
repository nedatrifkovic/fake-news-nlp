# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from src.models import (
    get_w2v_features,
    load_tfidf_vectorizer,
    load_word2vec_model,
    save_ml_model,
)

# Load processed dataset
df = pd.read_csv("data/processed/processed.csv")
texts = df["lemmatized_text"].tolist()
labels = df["label"].values


# Load vectorizers
tfidf_vectorizer = load_tfidf_vectorizer()
w2v_model = load_word2vec_model()

# Transform features
X_tfidf = tfidf_vectorizer.transform(texts)
X_w2v = get_w2v_features(texts, w2v_model)

# Split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_tfidf, labels, test_size=0.2, random_state=42
)
X_train_w2v, X_test_w2v, _, _ = train_test_split(
    X_w2v, labels, test_size=0.2, random_state=42
)

# 1. Logistic Regression (TF-IDF)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
save_ml_model(lr_model, "logreg_tfidf.pkl")

# 2. Random Forest (Word2Vec)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_w2v, y_train)
save_ml_model(rf_model, "rf_w2v.pkl")

# 3. XGBoost (TF-IDF)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_tfidf, y_train)
save_ml_model(xgb_model, "xgb_tfidf.pkl")


# Evaluation
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:   ", round(recall_score(y_test, y_pred), 4))
    print("F1:       ", round(f1_score(y_test, y_pred), 4))
    print()

evaluate(lr_model, X_test_tfidf, y_test, "Logistic Regression TF-IDF")
evaluate(rf_model, X_test_w2v, y_test, "Random Forest Word2Vec")
evaluate(xgb_model, X_test_tfidf, y_test, "XGBoost TF-IDF")

print("Training complete! Models saved in models/ml_models/")
