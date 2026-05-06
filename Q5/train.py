import time
import math
import torch
import torch.nn as nn
import logging
import os
from data_utils import get_data_wikitext2, batchify, get_batch
from ngram_model import NgramLanguageModel
from lstm_model import LSTMLanguageModel

# Zero-Console Rule
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hyperparameters (TUNED ADAMW) ---
BATCH_SIZE = 32      
EVAL_BATCH_SIZE = 32
BPTT = 40            # Increased for long-range dependencies
CLIP = 1.0           # Loosened for larger steps
LR = 1e-3            # Stable starting LR for AdamW
EPOCHS = 15          
PATIENCE = 5         # Early stopping patience
SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def repackage_hidden(h):
    """
    Detaches hidden state from its history (gradient graph).
    Prevents unnecessary backprop to past batches (Truncated BPTT).
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def train_lstm_epoch(model, data_source, criterion, optimizer, epoch, bptt, lr):
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(BATCH_SIZE)
    
    for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
        data, targets = get_batch(data_source, i, bptt)
        
        # State detachment
        hidden = repackage_hidden(hidden)
        
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        
        # Calculate Loss: CrossEntropy loss expects a flat vector
        loss = criterion(output.view(-1, model.vocab_size), targets)
        loss.backward()
        
        # 🔥 Clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            logger.info(f'| epoch {epoch:3d} | {batch:5d}/{len(data_source) // bptt:5d} batches | '
                        f'lr {lr:.4f} | ms/batch {elapsed * 1000 / 200:5.2f} | '
                        f'loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate_lstm(model, data_source, criterion, bptt):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(EVAL_BATCH_SIZE)
    
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            
            output_flat = output.view(-1, model.vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
            
    return total_loss / (len(data_source) - 1)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    train_data, valid_data, test_data, vocab, train_tokens, valid_tokens, test_tokens = get_data_wikitext2()
    
    # ----------------------------------------------------
    #  N-GRAM MODEL EĞİTİMİ (Baseline)
    # ----------------------------------------------------
    logger.info("=== Starting N-gram Model Training ===")
    ngram_model = NgramLanguageModel(n=3)
    ngram_model.train(train_tokens)
    
    # Validation
    ngram_valid_ppl = ngram_model.perplexity(valid_tokens)
    logger.info(f"🔥 N-gram Validation PPL: {ngram_valid_ppl:.2f}")
    
    # N-gram training stays in memory; no need to save weights separately.
    
    # ----------------------------------------------------
    #  LSTM MODEL EĞİTİMİ
    # ----------------------------------------------------
    logger.info("=== Starting LSTM Model Training ===")
    
    lstm_train_data = batchify(train_data, BATCH_SIZE, device)
    lstm_valid_data = batchify(valid_data, EVAL_BATCH_SIZE, device)
    
    vocab_size = len(vocab)
    model = LSTMLanguageModel(vocab_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # 🚀 TUNED ADAMW APPROACH
    # Stable convergence with AdamW supporting Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = None
    patience_counter = 0
    lr = LR
    
    best_model_path = os.path.join(os.path.dirname(__file__), "best_lstm_model_tuned.pt")
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Train & Eval
        train_lstm_epoch(model, lstm_train_data, criterion, optimizer, epoch, BPTT, lr)
        val_loss = evaluate_lstm(model, lstm_valid_data, criterion, BPTT)
        val_ppl = math.exp(val_loss)
        
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | '
                    f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        logger.info('-' * 89)
        
        # LR Step
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        
        # Checkpointing and Early Stopping
        if not best_val_loss or val_loss < best_val_loss:
            with open(best_model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(f"✨ New best model saved! (PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.warning(f"🚨 Early stopping triggered! Validation loss did not improve for {PATIENCE} epochs.")
                break

if __name__ == '__main__':
    main()
