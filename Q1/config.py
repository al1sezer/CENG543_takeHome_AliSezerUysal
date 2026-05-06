"""
CENG 467 - Q1: Text Classification with Representation Learning
Configuration module — all hyperparameters and constants in one place.
"""

import torch
import random
import numpy as np

# ============================================================
# Reproducibility
# ============================================================
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Data
# ============================================================
VAL_RATIO = 0.1          # 10% of train → validation
DATASET_NAME = "imdb"

# ============================================================
# Shared Training
# ============================================================
EPOCHS = 5

# ============================================================
# TF-IDF + Logistic Regression
# ============================================================
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)   # unigram + bigram
TFIDF_MIN_DF = 5
TFIDF_SUBLINEAR_TF = True
LR_MAX_ITER = 1000
LR_C = 1.0

# ============================================================
# BiLSTM + GloVe
# ============================================================
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.5        # increased from 0.3 to reduce overfitting
LSTM_FC_DROPOUT = 0.5     # before final linear layer
LSTM_MAX_SEQ_LEN = 256
LSTM_VOCAB_SIZE = 25_000
LSTM_EMBEDDING_DIM = 300  # GloVe 300d
LSTM_LR = 1e-3
LSTM_BATCH_SIZE = 64

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILENAME = "glove.6B.300d.txt"

# ============================================================
# DistilBERT
# ============================================================
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256     # default, ablation tests 128/256/512
BERT_LR = 2e-5
BERT_BATCH_SIZE = 16
BERT_WARMUP_RATIO = 0.1
BERT_WEIGHT_DECAY = 0.01
