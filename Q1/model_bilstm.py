import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from copy import deepcopy
from typing import Optional, Dict, Any, List
import numpy as np

from config import (
    DEVICE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_FC_DROPOUT, LSTM_LR
)

class AttentionPooling(nn.Module):
    """
    Learns to assign importance scores to each sequence step.
    Produces a weighted sum of the LSTM hidden states.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Score scalar for each hidden state
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output, mask):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        # mask shape: (batch_size, seq_len) -- 1 for valid, 0 for PAD
        
        # Calculate raw scores
        scores = self.attention(lstm_output).squeeze(-1) # shape: (batch, seq_len)
        
        # Apply mask: set score of PAD tokens to very large negative number
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get probabilities (weights) over the sequence
        weights = torch.softmax(scores, dim=1) # shape: (batch, seq_len)
        
        # Weighted sum: multiply each hidden state by its weight and sum over sequence
        # Expand weights to match hidden_size for multiplication
        context = (weights.unsqueeze(-1) * lstm_output).sum(dim=1) # shape: (batch, hidden_size)
        
        return context

class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pretrained_embeddings: Optional[np.ndarray] = None):
        super().__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # If GloVe provided, load it
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Freeze embeddings to prevent overfitting on small dataset
            self.embedding.weight.requires_grad = False
            
        # 2. BiLSTM Layer
        # hidden output will be 2 * hidden_size because bidirectional
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 3. Attention Pooling
        self.attention = AttentionPooling(hidden_size=LSTM_HIDDEN_SIZE * 2)
        
        # 4. Classifier Head
        self.dropout = nn.Dropout(LSTM_FC_DROPOUT)
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths, mask):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        
        # Pack sequence (optimization for variable lengths)
        # Move lengths to cpu as pack_padded_sequence requires CPU tensors in some PyTorch versions
        lengths_cpu = lengths.cpu() 
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch, seq_len, hidden_size * 2)
        
        # Apply attention pooling
        context = self.attention(output, mask) # (batch, hidden_size * 2)
        
        # Classifier
        out = self.dropout(context)
        out = self.fc(out) # (batch, 1)
        out = self.sigmoid(out).squeeze(1) # (batch)
        
        return out

def train_bilstm(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict[str, Any]:
    """
    Trains the BiLSTM and evaluates on validation set at each epoch.
    Returns the best model state dict AND the training history.
    """
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LSTM_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_val_f1 = -1.0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (seqs, labels, mask, lengths) in enumerate(train_loader):
            seqs, labels, mask, lengths = seqs.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE), lengths.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(seqs, lengths, mask)
            
            loss = criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for seqs, labels, mask, lengths in val_loader:
                seqs, labels, mask, lengths = seqs.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE), lengths.to(DEVICE)
                predictions = model(seqs, lengths, mask)
                
                # Convert prob -> binary class
                preds_binary = (predictions > 0.5).long().cpu().numpy()
                val_preds.extend(preds_binary)
                val_targets.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        # Checkpoint if best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = deepcopy(model.state_dict())
            
        scheduler.step(val_f1)
            
    print(f"Training complete. Best Val Macro-F1: {best_val_f1:.4f}")
    
    # Load best model locally before returning
    model.load_state_dict(best_model_state)
    return best_model_state, history

def predict_bilstm(model: nn.Module, data_loader: DataLoader) -> np.ndarray:
    """ Generates predictions using the BiLSTM model. """
    model = model.to(DEVICE)
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for seqs, labels, mask, lengths in data_loader:
            seqs, mask, lengths = seqs.to(DEVICE), mask.to(DEVICE), lengths.to(DEVICE)
            predictions = model(seqs, lengths, mask)
            preds_binary = (predictions > 0.5).long().cpu().numpy()
            all_preds.extend(preds_binary)
            
    return np.array(all_preds)
