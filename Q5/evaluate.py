import torch
import torch.nn as nn
import logging
import os
import math
from data_utils import get_data_wikitext2, batchify
from ngram_model import NgramLanguageModel
from lstm_model import LSTMLanguageModel
from train import evaluate_lstm, EVAL_BATCH_SIZE

# Zero-Console Rule
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_specific_model(model_path, model, test_data, criterion, bptt, device):
    """Calculates Test PPL using the specified model and BPTT"""
    if not os.path.exists(model_path):
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    loss = evaluate_lstm(model, test_data, criterion, bptt)
    return math.exp(loss)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation device: {device}")
    
    # 1. Data Loading
    logger.info("Loading test data...")
    _, _, test_data, vocab, train_tokens, _, test_tokens = get_data_wikitext2()
    vocab_size = len(vocab)
    criterion = nn.CrossEntropyLoss()
    lstm_test_data = batchify(test_data, EVAL_BATCH_SIZE, device)
    
    # 2. N-gram Evaluation
    logger.info("Preparing N-gram model for test set (approx. 5-10 sec)...")
    ngram_model = NgramLanguageModel(n=3)
    ngram_model.train(train_tokens)
    ngram_test_ppl = ngram_model.perplexity(test_tokens)
    
    # 3. Test All LSTM Variations
    logger.info("Evaluating LSTM Ablation Models...")
    results = {"Trigram + KN (Baseline)": ngram_test_ppl}
    
    # Ablation 1: Champion Model (400-400-2, BPTT=35)
    model_steplr = LSTMLanguageModel(vocab_size, embed_dim=400, hidden_dim=400, num_layers=2).to(device)
    ppl = evaluate_specific_model(os.path.join(os.path.dirname(__file__), "best_lstm_model_steplr.pt"), model_steplr, lstm_test_data, criterion, 35, device)
    if ppl: results["LSTM (SGD+StepLR) [Champion]"] = ppl

    # Ablation 2: AdamW Modern (400-400-2, BPTT=70)
    model_modern = LSTMLanguageModel(vocab_size, embed_dim=400, hidden_dim=400, num_layers=2).to(device)
    ppl = evaluate_specific_model(os.path.join(os.path.dirname(__file__), "best_lstm_model_modern.pt"), model_modern, lstm_test_data, criterion, 70, device)
    if ppl: results["LSTM (AdamW+CosineLR)"] = ppl

    # Ablation 3: SGD+Momentum (400-400-2, BPTT=70)
    model_sgdm = LSTMLanguageModel(vocab_size, embed_dim=400, hidden_dim=400, num_layers=2).to(device)
    ppl = evaluate_specific_model(os.path.join(os.path.dirname(__file__), "best_lstm_model_sgdm.pt"), model_sgdm, lstm_test_data, criterion, 70, device)
    if ppl: results["LSTM (SGD+Momentum+Cosine)"] = ppl

    # Ablation 4: Batch 64 Testi (400-400-2, BPTT=35)
    model_b64 = LSTMLanguageModel(vocab_size, embed_dim=400, hidden_dim=400, num_layers=2).to(device)
    ppl = evaluate_specific_model(os.path.join(os.path.dirname(__file__), "best_lstm_model_batch64.pt"), model_b64, lstm_test_data, criterion, 35, device)
    if ppl: results["LSTM (SGD B64 Test)"] = ppl

    # Ablation 5: Tuned Model (512-512-4, BPTT=40)
    model_tuned = LSTMLanguageModel(vocab_size, embed_dim=512, hidden_dim=512, num_layers=4).to(device)
    ppl = evaluate_specific_model(os.path.join(os.path.dirname(__file__), "best_lstm_model_tuned.pt"), model_tuned, lstm_test_data, criterion, 40, device)
    if ppl: results["LSTM (Tuned 512x4 AdamW)"] = ppl

    # 4. Table Output
    result_str = "\n╔══════════════════════════════════════════════════╗\n"
    result_str += "║         ABLATION STUDY & TEST RESULTS            ║\n"
    result_str += "╠══════════════════════════════════╦═══════════════╣\n"
    result_str += "║ Model                            ║ Test PPL      ║\n"
    result_str += "╠══════════════════════════════════╬═══════════════╣\n"
    
    for name, score in results.items():
        result_str += f"║ {name:<32} ║    {score:8.2f}   ║\n"
        
    result_str += "╚══════════════════════════════════╩═══════════════╝\n"
    
    logger.info("Test Results:" + result_str)
    
    eval_file = os.path.join(os.path.dirname(__file__), "evaluation_results.txt")
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write("CENG 467 - Q5: Language Modeling Ablation Results\n")
        f.write("="*58 + "\n")
        f.write(result_str)
        f.write("\nNote: Lower PPL value indicates a better model.\n")
        
    logger.info(f"✅ Results also saved to '{eval_file}'.")

if __name__ == '__main__':
    main()
