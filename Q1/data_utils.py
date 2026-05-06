import os
import re
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple, Dict, Optional
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config import (
    SEED, VAL_RATIO, DATASET_NAME, GLOVE_URL, GLOVE_FILENAME,
    LSTM_MAX_SEQ_LEN
)

# Initialize NLTK components for preprocessing
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    # Safe fallback if not properly downloaded
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

# ============================================================
# Loading and Splitting
# ============================================================
def load_and_split_data():
    """
    Loads IMDb dataset and splits train into train/val using VAL_RATIO.
    Leaves the original test set untouched for fair evaluation.
    """
    print(f"Loading {DATASET_NAME} dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    # Original split: 25k train, 25k test
    # We split 'train' into 'train' and 'validation' with fixed seed
    train_val_split = dataset['train'].train_test_split(
        test_size=VAL_RATIO, 
        stratify_by_column='label', 
        seed=SEED
    )
    
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    test_data = dataset['test']
    
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    return train_data, val_data, test_data


# ============================================================
# Preprocessing Strategies
# ============================================================
def preprocess_minimal(text: str) -> str:
    """
    Minimal processing mainly dropping HTML tags.
    Used for contextual models (BERT) that handle their own tokenization.
    """
    # Remove HTML tags like <br />
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_classical(text: str, remove_stopwords: bool = True, apply_lemmatization: bool = True) -> str:
    """
    Heavier processing for classical ML (TF-IDF) & standard neural models.
    """
    # 1. HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Lowercasing
    text = text.lower()
    # 3. Punctuation & numbers (keep only letters)
    text = re.sub(r'[^a-z]+', ' ', text)
    
    # 4. Tokenization (word-level)
    tokens = word_tokenize(text)
    
    # 5. Stopword removal (optional for ablation)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    
    # 6. Lemmatization (optional for ablation)
    if apply_lemmatization:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        
    return ' '.join(tokens)

def preprocess_subword(text: str, tokenizer) -> str:
    """
    Subword tokenization (WordPiece) for classical ML.
    Tokenizes text using DistilBERT's tokenizer and joins subwords with spaces.
    e.g. "unhappy" -> "un ##happy"
    """
    # Just remove HTML tags first
    text = re.sub(r'<[^>]+>', ' ', text)
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)


# ============================================================
# GloVe and Vocabulary (For BiLSTM)
# ============================================================
def download_glove_if_needed(data_dir: str = "."):
    """Downloads and extracts GloVe 300d if not present."""
    os.makedirs(data_dir, exist_ok=True)
    glove_txt_path = os.path.join(data_dir, GLOVE_FILENAME)
    
    if os.path.exists(glove_txt_path):
        return glove_txt_path
        
    print("GloVe embeddings not found. Downloading (this may take a while, ~822MB)...")
    glove_zip_path = os.path.join(data_dir, "glove.6B.zip")
    
    urllib.request.urlretrieve(GLOVE_URL, glove_zip_path)
    print("Download complete. Extracting 300d embeddings...")
    
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        # We only need the 300d version to save disk space
        zip_ref.extract(GLOVE_FILENAME, data_dir)
        
    os.remove(glove_zip_path)
    print("Extraction complete.")
    
    return glove_txt_path

def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    """Builds a word-to-index mapping from the most common words."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
        
    # <PAD> at 0, <UNK> at 1
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    # Keep top (vocab_size - 2) words
    for word, _ in counter.most_common(vocab_size - 2):
        word2idx[word] = len(word2idx)
        
    return word2idx

def load_glove_embeddings(word2idx: Dict[str, int], dim: int = 300) -> np.ndarray:
    """Loads GloVe embeddings into a numpy matrix aligned with word2idx."""
    glove_path = download_glove_if_needed(data_dir=".")  # Current working directory
    
    # Initialize with random normal for words not in GloVe
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(scale=0.1, size=(vocab_size, dim))
    # Explicitly static <PAD> embedding (zeros)
    embedding_matrix[0] = np.zeros(dim)
    
    print("Parsing GloVe embeddings...")
    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                idx = word2idx[word]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_matrix[idx] = coefs
                found += 1
                
    print(f"Found {found}/{vocab_size} words in GloVe.")
    return embedding_matrix


# ============================================================
# PyTorch Dataset for BiLSTM
# ============================================================
class IMDbDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], word2idx: Dict[str, int], max_len: int = LSTM_MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
        # Precompute integer sequences
        self.sequences = []
        for text in texts:
            seq = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.split()]
            self.sequences.append(seq)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Truncate
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
            
        seq_len = len(seq)
        if seq_len == 0:
            # Fallback for empty strings
            seq = [self.word2idx['<UNK>']]
            seq_len = 1
            
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32), seq_len

def collate_fn(batch):
    """
    Collate function to pad variable length sequences to the max length in the current batch.
    """
    sequences, labels, lengths = zip(*batch)
    
    # Determine max length in this batch
    batch_max_len = max(lengths)
    
    # Pad sequences
    padded_seqs = torch.zeros((len(batch), batch_max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
        
    # Attention mask (1 for real tokens, 0 for PAD)
    mask = (padded_seqs != 0).float()
    
    labels = torch.stack(labels)
    
    return padded_seqs, labels, mask, torch.tensor(lengths, dtype=torch.long)
