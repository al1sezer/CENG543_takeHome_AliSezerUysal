import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
from tqdm import tqdm
from config import CLIP_GRAD, PAD_IDX, TF_WARMUP_STEPS, TF_D_MODEL

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, dataloader, optimizer, criterion, clip, is_transformer=False, scheduler=None):
    model.train()
    epoch_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        if is_transformer:
            # Transformer: tgt_input is [SOS, ..., EOS-1], tgt_output is [SOS+1, ..., EOS]
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            # output: [batch, tgt_len-1, vocab_size]
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
        else:
            # Seq2Seq
            output, _ = model(src, tgt)
            # output: [batch, tgt_len, vocab_size]
            # tgt: [batch, tgt_len]
            # Ignore the first token (<sos>) for loss
            output = output[:, 1:].reshape(-1, output.shape[-1])
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, is_transformer=False):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            if is_transformer:
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                output = model(src, tgt_input)
                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt_output.reshape(-1)
                loss = criterion(output, tgt_output)
            else:
                output, _ = model(src, tgt, teacher_forcing_ratio=0)
                output = output[:, 1:].reshape(-1, output.shape[-1])
                tgt = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt)
                
            epoch_loss += loss.item()
            
    return epoch_loss / len(dataloader)

def transformer_lr_lambda(step):
    step = max(step, 1)
    return (TF_D_MODEL ** -0.5) * min(step ** -0.5, step * (TF_WARMUP_STEPS ** -1.5))

def train_model_orchestrator(model, train_loader, val_loader, optimizer, criterion, epochs, clip, is_transformer=False, scheduler=None):
    from config import EARLY_STOP_PATIENCE
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, is_transformer, scheduler)
        valid_loss = evaluate_epoch(model, val_loader, criterion, is_transformer)
        
        end_time = time.time()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        
        print(f"Epoch: {epoch+1:02} | Time: {end_time - start_time:.2f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f'best_model_{"transformer" if is_transformer else "seq2seq"}.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
    return history
