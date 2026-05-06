"""
Q3 - Text Summarization: Configuration
All constants and hyperparameters managed in one place.
"""

# ── Reproducibility ──────────────────────────────────────────
SEED = 42

# ── Dataset ──────────────────────────────────────────────────
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
SPLIT = "test"
SUBSET_SIZE = 1000

# ── Extractive (TextRank) ────────────────────────────────────
TOP_K_SENTENCES = 3

# ── Abstractive (BART) ──────────────────────────────────────
MODEL_NAME = "facebook/bart-large-cnn"
BATCH_SIZE = 8
MAX_LENGTH = 130
MIN_LENGTH = 30
NUM_BEAMS = 4

# ── BERTScore ────────────────────────────────────────────────
BERTSCORE_MODEL = "roberta-large"
