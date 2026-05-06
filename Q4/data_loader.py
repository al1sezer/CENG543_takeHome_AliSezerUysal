import spacy
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from config import (
    DATASET_NAME, SPACY_DE, SPACY_EN, MIN_FREQ, MAX_LEN,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, BATCH_SIZE
)

# Global spaCy loaders to avoid redundant loading
_nlp_de = None
_nlp_en = None

def tokenize_de(text):
    global _nlp_de
    if _nlp_de is None:
        try:
            _nlp_de = spacy.load(SPACY_DE)
        except OSError:
            import os
            os.system(f"python -m spacy download {SPACY_DE}")
            _nlp_de = spacy.load(SPACY_DE)
    return [token.text.lower() for token in _nlp_de.tokenizer(text)]

def tokenize_en(text):
    global _nlp_en
    if _nlp_en is None:
        try:
            _nlp_en = spacy.load(SPACY_EN)
        except OSError:
            import os
            os.system(f"python -m spacy download {SPACY_EN}")
            _nlp_en = spacy.load(SPACY_EN)
    return [token.text.lower() for token in _nlp_en.tokenizer(text)]

class Vocabulary:
    def __init__(self, min_freq):
        self.min_freq = min_freq
        self.itos = {PAD_IDX: "<pad>", SOS_IDX: "<sos>", EOS_IDX: "<eos>", UNK_IDX: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in sentence:
                frequencies[word] += 1
        
        for word, freq in frequencies.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text_list):
        return [self.stoi.get(token, UNK_IDX) for token in text_list]

    def __len__(self):
        return len(self.itos)

def process_dataset():
    print("Loading Multi30k dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]

    print("Tokenizing and building vocabs (this may take a while)...")
    train_src_tokens = [tokenize_de(ex["de"]) for ex in train_data]
    train_tgt_tokens = [tokenize_en(ex["en"]) for ex in train_data]

    src_vocab = Vocabulary(MIN_FREQ)
    tgt_vocab = Vocabulary(MIN_FREQ)

    src_vocab.build_vocabulary(train_src_tokens)
    tgt_vocab.build_vocabulary(train_tgt_tokens)

    return dataset, src_vocab, tgt_vocab

def collate_fn(batch, src_vocab, tgt_vocab, device):
    src_batch, tgt_batch = [], []
    for example in batch:
        # Source (DE)
        src_tokens = tokenize_de(example["de"])
        src_indices = [SOS_IDX] + src_vocab.numericalize(src_tokens)[:MAX_LEN-2] + [EOS_IDX]
        src_batch.append(torch.tensor(src_indices))
        
        # Target (EN)
        tgt_tokens = tokenize_en(example["en"])
        tgt_indices = [SOS_IDX] + tgt_vocab.numericalize(tgt_tokens)[:MAX_LEN-2] + [EOS_IDX]
        tgt_batch.append(torch.tensor(tgt_indices))

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX).to(device)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX).to(device)
    
    return src_batch, tgt_batch

def get_dataloaders(dataset, src_vocab, tgt_vocab, device):
    def partial_collate(batch):
        return collate_fn(batch, src_vocab, tgt_vocab, device)

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=partial_collate)
    valid_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=partial_collate)
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=partial_collate)
    
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    # Quick test
    from config import DEVICE
    ds, sv, tv = process_dataset()
    print(f"Vocab sizes: DE={len(sv)}, EN={len(tv)}")
    tl, _, _ = get_dataloaders(ds, sv, tv, DEVICE)
    src, tgt = next(iter(tl))
    print(f"Batch shapes: src={src.shape}, tgt={tgt.shape}")
