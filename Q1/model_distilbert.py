import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from copy import deepcopy
import numpy as np
from typing import List, Dict, Any

from config import (
    DEVICE, BERT_MODEL_NAME, BERT_MAX_LENGTH, BERT_LR,
    BERT_WEIGHT_DECAY, BERT_WARMUP_RATIO
)

# ============================================================
# Dataset Wrapper for BERT
# ============================================================
class BertDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = BERT_MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================
# Model and Training
# ============================================================
def init_distilbert():
    """Initializes the tokenizer and model."""
    print(f"Loading {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2
    )
    return tokenizer, model

def train_distilbert(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict[str, Any]:
    """
    Trains DistilBERT and evaluates on validation set at each epoch.
    Returns the best model state dict AND training history.
    """
    model = model.to(DEVICE)
    
    # Optimizer with weight decay for regularization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': BERT_WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=BERT_LR)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_steps * BERT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    best_val_f1 = -1.0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        # Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Deep copy to memory to prevent overwriting
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f"Training complete. Best Val Macro-F1: {best_val_f1:.4f}")
    
    # Reload best model locally
    model.load_state_dict(best_model_state)
    return best_model_state, history

def predict_distilbert(model: nn.Module, data_loader: DataLoader) -> np.ndarray:
    """ Generates predictions using DistilBERT. """
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    return np.array(all_preds)
