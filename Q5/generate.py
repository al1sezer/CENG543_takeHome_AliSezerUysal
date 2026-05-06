import torch
import torch.nn.functional as F
import logging
import os
from data_utils import get_data_wikitext2
from ngram_model import NgramLanguageModel
from lstm_model import LSTMLanguageModel

# Zero-Console Rule
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_lstm_text(model, vocab, prompt_tokens, max_len=50, temperature=1.0, top_k=0, top_p=1.0, device='cpu'):
    """
    Text generation (inference) function for the LSTM model.
    Generates the continuation of the given prompt using the specified strategy.
    """
    model.eval()
    
    # Initialize hidden state with a batch_size of 1
    hidden = model.init_hidden(1)
    
    current_tokens = list(prompt_tokens)
    
    # Convert prompt words to Tensor IDs
    input_seq = torch.tensor([[vocab.stoi.get(t, vocab.stoi['<unk>'])] for t in current_tokens], dtype=torch.long).to(device)
    
    # Feed the prompt to the model to "warm up" the hidden state (build up context)
    with torch.no_grad():
        for i in range(input_seq.size(0) - 1):
            _, hidden = model(input_seq[i].unsqueeze(0), hidden)
            
        # Start actual prediction with the last word of the prompt
        last_input = input_seq[-1].unsqueeze(0)
        
        for _ in range(max_len):
            output, hidden = model(last_input, hidden)
            
            # Logits (vocab_size number of elements)
            logits = output.squeeze()
            
            if temperature == 0:
                # Greedy
                next_word_idx = torch.argmax(logits).item()
            else:
                # Temperature
                if temperature != 1.0:
                    logits = logits / temperature
                    
                probs = F.softmax(logits, dim=-1)
                
                # Top-k
                if top_k > 0:
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask[top_indices] = True
                    probs[~mask] = 0.0
                    probs = probs / probs.sum()
                    
                # Top-p (Nucleus)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    if probs.sum() > 0:
                        probs = probs / probs.sum()
                        
                # Sample
                if probs.sum() <= 0 or torch.isnan(probs).any():
                    next_word_idx = torch.randint(0, len(vocab), (1,)).item()
                else:
                    next_word_idx = torch.multinomial(probs, 1).item()
            
            next_word = vocab.itos[next_word_idx]
            current_tokens.append(next_word)
            
            if next_word == '<eos>':
                break
                
            # Use the selected word as input for the next step (Autoregressive)
            last_input = torch.tensor([[next_word_idx]], dtype=torch.long).to(device)
            
    return current_tokens

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data and Vocab
    _, _, _, vocab, train_tokens, _, _ = get_data_wikitext2()
    
    # 2. N-Gram Preparation (Trains quickly since it's lightweight)
    logger.info("Preparing N-gram model for text generation...")
    ngram_model = NgramLanguageModel(n=3)
    ngram_model.train(train_tokens)
    
    # 3. Load LSTM
    vocab_size = len(vocab)
    lstm_model = LSTMLanguageModel(vocab_size, embed_dim=400, hidden_dim=400, num_layers=2).to(device)
    
    model_path = os.path.join(os.path.dirname(__file__), "best_lstm_model_steplr.pt")
    if os.path.exists(model_path):
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info("Trained LSTM weights loaded successfully!")
    else:
        logger.warning("Trained weights not found! (Please run train.py first). Generating with random weights.")
        
    # --- Experiment Setup ---
    prompts = [
        ["the", "history", "of"],
        ["it", "is", "a"],
        ["she", "went", "to", "the"]
    ]
    
    strategies = [
        {"name": "Greedy", "temperature": 0.0, "top_k": 0, "top_p": 1.0},
        {"name": "Temp 0.8", "temperature": 0.8, "top_k": 0, "top_p": 1.0},
        {"name": "Top-k (k=40)", "temperature": 1.0, "top_k": 40, "top_p": 1.0},
        {"name": "Top-p (p=0.9)", "temperature": 1.0, "top_k": 0, "top_p": 0.9},
    ]
    
    output_path = os.path.join(os.path.dirname(__file__), "generation_results.txt")
    logger.info(f"Generated texts will be saved to {output_path}.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            header = f"\n{'='*50}\nPROMPT: {' '.join(prompt)}\n{'='*50}\n"
            f.write(header)
            
            for strategy in strategies:
                strat_name = strategy["name"]
                
                # N-Gram Generation
                ngram_gen = ngram_model.generate(prompt, max_len=40, **{k: v for k, v in strategy.items() if k != "name"})
                ngram_text = " ".join(ngram_gen).replace('<eos>', '.')
                
                # LSTM Generation
                lstm_gen = generate_lstm_text(lstm_model, vocab, prompt, max_len=40, device=device, **{k: v for k, v in strategy.items() if k != "name"})
                lstm_text = " ".join(lstm_gen).replace('<eos>', '.')
                
                res = f"Strategy: {strat_name}\nN-GRAM: {ngram_text}\nLSTM  : {lstm_text}\n"
                f.write(res + "\n")
                
    logger.info("Text generation completed.")

if __name__ == '__main__':
    main()
