import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import os
import tqdm # import kalsa da kullanmayacagiz
from seqeval.metrics import f1_score, classification_report

# Mute redundant library logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from data_utils import get_conll_dataset
from models import BiLSTM_CRF

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_LEN = 128
BATCH_SIZE = 32
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 3
GLOVE_PATH = "../glove.6B.300d.txt"

# Tag Mapping (CoNLL-2003 Standard)
TAG_TO_IX = {
    'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
}
IX_TO_TAG = {v: k for k, v in TAG_TO_IX.items()}

class NERDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['ner_tags']

        # Map tokens to indices (1 for <UNK>)
        seq = [self.word_to_ix.get(t.lower(), 1) for t in tokens]
        
        # Padding and Masking
        padding_len = MAX_LEN - len(seq)
        mask = [1] * len(seq) + [0] * padding_len
        seq = seq + [0] * padding_len # 0 is <PAD>
        tags = tags + [0] * padding_len 

        return {
            'input_ids': torch.tensor(seq[:MAX_LEN], dtype=torch.long),
            'labels': torch.tensor(tags[:MAX_LEN], dtype=torch.long),
            'mask': torch.tensor(mask[:MAX_LEN], dtype=torch.bool)
        }

def build_vocab(dataset):
    """Build vocabulary from unique words in dataset."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for split in ['train', 'validation']:
        for example in dataset[split]:
            for token in example['tokens']:
                token = token.lower()
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab

def load_glove_embeddings(vocab, glove_path):
    """Load GloVe embeddings for words in vocabulary."""
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), EMBEDDING_DIM))
    embeddings[0] = 0 # Pad zero initialization
    
    found_count = 0
    logger.info(f"Loading GloVe weights: {glove_path}")
    
    if not os.path.exists(glove_path):
        logger.error(f"GloVe file not found: {glove_path}!")
        return torch.from_numpy(embeddings).float()

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(values[1:], dtype='float32')
                found_count += 1
    
    logger.info(f"Found {found_count}/{len(vocab)} words in GloVe.")
    return torch.from_numpy(embeddings).float()

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)
            
            # Viterbi decoding
            outputs = model(ids, mask)
            
            # Clean based on mask
            for i in range(len(outputs)):
                true_len = mask[i].sum().item()
                # Map predictions to tag strings
                pred_tags = [IX_TO_TAG[p] for p in outputs[i]]
                true_tags = [IX_TO_TAG[l.item()] for l in labels[i][:true_len]]
                
                all_preds.append(pred_tags)
                all_labels.append(true_tags)
                
    return f1_score(all_labels, all_preds), classification_report(all_labels, all_preds)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    dataset = get_conll_dataset()
    vocab = build_vocab(dataset)
    pretrained_weights = load_glove_embeddings(vocab, GLOVE_PATH)

    train_data = NERDataset(dataset['train'], vocab)
    val_data = NERDataset(dataset['validation'], vocab)
    test_data = NERDataset(dataset['test'], vocab)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = BiLSTM_CRF(len(vocab), TAG_TO_IX, EMBEDDING_DIM, HIDDEN_DIM).to(device)
    model.word_embeds.weight.data.copy_(pretrained_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0
    patience_counter = 0

    logger.info(f"Training started (Epochs: {EPOCHS}, Batch: {BATCH_SIZE})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Quiet training (no per-iteration progress bar)
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            model.zero_grad()
            loss = model.neg_log_likelihood(ids, labels, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Log only at the end of each epoch
        val_f1, _ = evaluate(model, val_loader, device)
        logger.info(f">> [Epoch {epoch+1:02d}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_bilstm_crf.pt")
            logger.info("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered.")
                break

    # Test Evaluation
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load("best_bilstm_crf.pt"))
    test_f1, test_report = evaluate(model, test_loader, device)
    
    with open("bilstm_crf_results.txt", "w") as f:
        f.write("BiLSTM-CRF Results\n")
        f.write("==================\n")
        f.write(test_report)
    
    logger.info(f"Final Test F1: {test_f1:.4f}. Results saved to bilstm_crf_results.txt")

if __name__ == "__main__":
    train()
