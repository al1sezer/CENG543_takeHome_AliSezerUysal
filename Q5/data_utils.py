import torch
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Vocabulary:
    def __init__(self, min_freq=3):
        self.itos = ['<unk>', '<eos>']
        self.stoi = {'<unk>': 0, '<eos>': 1}
        self.min_freq = min_freq
        
    def build_vocab(self, tokens):
        counter = Counter(tokens)
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                
    def __len__(self):
        return len(self.itos)

def get_data_wikitext2(min_freq=3):
    logger.info("Loading/downloading WikiText-2 dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    except ImportError:
        logger.error("HuggingFace 'datasets' library not found. Please install with 'pip install datasets'.")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    def tokenize(text):
        # Simple whitespace and lowercase tokenization for WikiText
        return [word.lower() for word in text.split() if word.strip()]

    def process_split(split_data):
        tokens = []
        for example in split_data:
            text = example['text']
            if text.strip():  # Skip empty lines
                tokens.extend(tokenize(text))
                tokens.append('<eos>')
        return tokens

    train_tokens = process_split(dataset['train'])
    valid_tokens = process_split(dataset['validation'])
    test_tokens = process_split(dataset['test'])

    logger.info("Building vocabulary...")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(train_tokens)
    logger.info(f"Vocabulary size: {len(vocab)} (min_freq={min_freq})")

    def numericalize(tokens):
        # Converts tokens to IDs. Unknown words become <unk>.
        unk_idx = vocab.stoi['<unk>']
        return torch.tensor([vocab.stoi.get(token, unk_idx) for token in tokens], dtype=torch.long)

    train_data = numericalize(train_tokens)
    valid_data = numericalize(valid_tokens)
    test_data = numericalize(test_tokens)

    return train_data, valid_data, test_data, vocab, train_tokens, valid_tokens, test_tokens

def batchify(data, batch_size, device):
    """
    Splits data into batch_size columns. Trims remainders.
    Required format for LSTM sliding window.
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt):
    """
    Returns input and target blocks of bptt length starting from index i.
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target
