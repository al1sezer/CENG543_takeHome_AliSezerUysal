"""
CENG 467 - Q1: TF-IDF + Logistic Regression Model Strategy
Handles the classical ML approach (Sparse Representation).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
import numpy as np

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF,
    TFIDF_SUBLINEAR_TF, LR_MAX_ITER, LR_C
)

def train_tfidf_lr(train_texts: List[str], train_labels: List[int]) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Fits a TF-IDF vectorizer and trains a Logistic Regression model.
    """
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF
    )
    
    # Transform train data
    X_train = vectorizer.fit_transform(train_texts)
    y_train = np.array(train_labels)
    
    print(f"TF-IDF sparse matrix shape: {X_train.shape}")
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Training complete.")
    
    return vectorizer, model

def predict_tfidf(vectorizer: TfidfVectorizer, model: LogisticRegression, texts: List[str]) -> np.ndarray:
    """
    Transforms texts and generates predictions.
    """
    X = vectorizer.transform(texts)
    return model.predict(X)

def get_top_features(vectorizer: TfidfVectorizer, model: LogisticRegression, n: int = 10):
    """
    Extracts the top n most positive and top n most negative words from the Logistic Regression coefficients.
    Used for Interpretability analysis.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    
    # Sort indices by coefficient value
    sorted_idx = np.argsort(coefs)
    
    # Lowest coefficients (most negative)
    top_negative = [(feature_names[i], coefs[i]) for i in sorted_idx[:n]]
    
    # Highest coefficients (most positive)
    top_positive = [(feature_names[i], coefs[i]) for i in sorted_idx[-n:]]
    top_positive.reverse() # Sort descending
    
    return top_positive, top_negative
