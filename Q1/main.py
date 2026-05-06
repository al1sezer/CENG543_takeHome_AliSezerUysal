import os
import gc
import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader

from config import set_seed, EPOCHS, LSTM_VOCAB_SIZE, LSTM_BATCH_SIZE, BERT_BATCH_SIZE
from data_utils import (
    load_and_split_data, preprocess_classical, preprocess_minimal, preprocess_subword,
    build_vocab, load_glove_embeddings, IMDbDataset, collate_fn
)
from model_tfidf import train_tfidf_lr, predict_tfidf, get_top_features
from model_bilstm import SentimentBiLSTM, train_bilstm, predict_bilstm
from model_distilbert import init_distilbert, train_distilbert, predict_distilbert, BertDataset
from evaluate import compute_metrics, compare_models, error_analysis, ablation_report

# Setup global dictionary to collect final metrics and predictions
final_metrics = {}
final_predictions = {}
y_true_test = None
test_texts_raw = None

def run_tfidf_pipeline(train_data, val_data, test_data):
    """ Orchestrates TF-IDF + Logistic Regression training and evaluation. """
    print("\n" + "="*50)
    print("--- 1. TF-IDF + LOGISTIC REGRESSION ---")
    print("="*50)
    
    # Needs classical preprocessing with stopwords & lemmatization
    print("Preprocessing text...")
    X_train_text = [preprocess_classical(t) for t in train_data['text']]
    y_train = list(train_data['label'])
    
    X_test_text = [preprocess_classical(t) for t in test_data['text']]
    y_test = np.array(test_data['label'])
    
    vectorizer, model = train_tfidf_lr(X_train_text, y_train)
    
    print("Testing TF-IDF model...")
    preds = predict_tfidf(vectorizer, model, X_test_text)
    
    metrics = compute_metrics(y_test, preds)
    final_metrics["TF-IDF"] = metrics
    final_predictions["TF-IDF"] = preds
    
    print("Top features extracted by TF-IDF (Interpretability):")
    top_pos, top_neg = get_top_features(vectorizer, model, n=5)
    print(f"Most Positive: {[w for w, _ in top_pos]}")
    print(f"Most Negative: {[w for w, _ in top_neg]}")
    
    # Store globally
    global y_true_test
    if y_true_test is None:
        y_true_test = y_test
    
    # Run Ablation: Without Stopwords/Lemmatization
    print("\n--- Ablation Study: Classical Processing ---")
    X_train_abl = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in train_data['text']]
    X_test_abl = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in test_data['text']]
    
    vec_abl, mod_abl = train_tfidf_lr(X_train_abl, y_train)
    preds_abl = predict_tfidf(vec_abl, mod_abl, X_test_abl)
    abl_metrics = compute_metrics(y_test, preds_abl)
    
    # Run Tokenization Ablation: Subword (WordPiece)
    print("\n--- Ablation Study: Subword Tokenization on TF-IDF ---")
    tokenizer_for_ablation, _ = init_distilbert() # We just need the tokenizer
    X_train_subword = [preprocess_subword(t, tokenizer_for_ablation) for t in train_data['text']]
    X_test_subword = [preprocess_subword(t, tokenizer_for_ablation) for t in test_data['text']]
    
    vec_subword, mod_subword = train_tfidf_lr(X_train_subword, y_train)
    preds_subword = predict_tfidf(vec_subword, mod_subword, X_test_subword)
    subword_metrics = compute_metrics(y_test, preds_subword)
    
    del vectorizer, model, X_train_text, X_test_text, X_train_abl, X_test_abl, tokenizer_for_ablation
    gc.collect()
    
    return abl_metrics, subword_metrics


def run_bilstm_pipeline(train_data, val_data, test_data):
    """ Orchestrates BiLSTM + GloVe + Attention Pooling training and evaluation. """
    print("\n" + "="*50)
    print("--- 2. BiLSTM + GLOVE + ATTENTION POOLING ---")
    print("="*50)
    
    # Minimal cleaning, keeping sequence order, no stopword removal for LSTM
    X_train_text = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in train_data['text']]
    X_val_text = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in val_data['text']]
    X_test_text = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in test_data['text']]
    
    print("Building vocabulary...")
    word2idx = build_vocab(X_train_text, LSTM_VOCAB_SIZE)
    embeddings = load_glove_embeddings(word2idx)
    
    # PyTorch Datasets
    train_ds = IMDbDataset(X_train_text, list(train_data['label']), word2idx)
    val_ds = IMDbDataset(X_val_text, list(val_data['label']), word2idx)
    test_ds = IMDbDataset(X_test_text, list(test_data['label']), word2idx)
    
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = SentimentBiLSTM(len(word2idx), 300, embeddings)
    
    print(f"Training for {EPOCHS} epochs...")
    train_bilstm(model, train_loader, val_loader, EPOCHS)
    
    print("Testing BiLSTM model...")
    preds = predict_bilstm(model, test_loader)
    
    metrics = compute_metrics(np.array(test_data['label']), preds)
    final_metrics["BiLSTM_GloVe"] = metrics
    final_predictions["BiLSTM_GloVe"] = preds
    
    del model, train_loader, val_loader, test_loader, embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_distilbert_pipeline(train_data, val_data, test_data):
    """ Orchestrates DistilBERT training and evaluation. """
    print("\n" + "="*50)
    print("--- 3. DISTILBERT ---")
    print("="*50)
    
    # Highly context dependent, minimal manual cleaning
    X_train_text = [preprocess_minimal(t) for t in train_data['text']]
    X_val_text = [preprocess_minimal(t) for t in val_data['text']]
    X_test_text = [preprocess_minimal(t) for t in test_data['text']]
    
    global test_texts_raw
    test_texts_raw = X_test_text
    
    tokenizer, model = init_distilbert()
    
    train_ds = BertDataset(X_train_text, list(train_data['label']), tokenizer)
    val_ds = BertDataset(X_val_text, list(val_data['label']), tokenizer)
    test_ds = BertDataset(X_test_text, list(test_data['label']), tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BERT_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BERT_BATCH_SIZE, shuffle=False)
    
    print(f"Training for {EPOCHS} epochs...")
    train_distilbert(model, train_loader, val_loader, EPOCHS)
    
    print("Testing DistilBERT model...")
    preds = predict_distilbert(model, test_loader)
    
    metrics = compute_metrics(np.array(test_data['label']), preds)
    final_metrics["DistilBERT"] = metrics
    final_predictions["DistilBERT"] = preds
    
    del model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Run Sentiment Analysis Models")
    parser.add_argument('--model', type=str, choices=['all', 'tfidf', 'bilstm', 'distilbert'], default='all', help='Which model to run')
    args = parser.parse_args()

    # Enforce Reproducibility
    set_seed()
    
    train_data, val_data, test_data = load_and_split_data()
    
    global test_texts_raw, y_true_test
    test_texts_raw = [preprocess_minimal(t) for t in test_data['text']]
    y_true_test = np.array(test_data['label'])
    
    abl_tfidf, subword_metrics = None, None
    metrics_128 = None
    bilstm_history, bert_history = None, None

    if args.model in ['all', 'tfidf']:
        abl_tfidf, subword_metrics = run_tfidf_pipeline(train_data, val_data, test_data)
        
    if args.model in ['all', 'bilstm']:
        print("\n" + "="*50)
        print("--- 2. BiLSTM + GLOVE + ATTENTION POOLING ---")
        print("="*50)
        X_train_lstm = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in train_data['text']]
        X_val_lstm = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in val_data['text']]
        X_test_lstm = [preprocess_classical(t, remove_stopwords=False, apply_lemmatization=False) for t in test_data['text']]
        
        word2idx = build_vocab(X_train_lstm, LSTM_VOCAB_SIZE)
        embeddings = load_glove_embeddings(word2idx)
        
        train_ds = IMDbDataset(X_train_lstm, list(train_data['label']), word2idx)
        val_ds = IMDbDataset(X_val_lstm, list(val_data['label']), word2idx)
        test_ds = IMDbDataset(X_test_lstm, list(test_data['label']), word2idx)
        
        train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model_lstm = SentimentBiLSTM(len(word2idx), 300, embeddings)
        _, bilstm_history = train_bilstm(model_lstm, train_loader, val_loader, EPOCHS)
        
        preds_lstm = predict_bilstm(model_lstm, test_loader)
        final_metrics["BiLSTM_GloVe"] = compute_metrics(np.array(test_data['label']), preds_lstm)
        final_predictions["BiLSTM_GloVe"] = preds_lstm
        
        # --- Truncation Ablation: length 128 ---
        print("\n[BiLSTM Ablation: Training with max_seq_len=128]")
        train_ds_128 = IMDbDataset(X_train_lstm, list(train_data['label']), word2idx, max_len=128)
        val_ds_128 = IMDbDataset(X_val_lstm, list(val_data['label']), word2idx, max_len=128)
        test_ds_128 = IMDbDataset(X_test_lstm, list(test_data['label']), word2idx, max_len=128)
        
        train_loader_128 = DataLoader(train_ds_128, batch_size=LSTM_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader_128 = DataLoader(val_ds_128, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader_128 = DataLoader(test_ds_128, batch_size=LSTM_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model_lstm_128 = SentimentBiLSTM(len(word2idx), 300, embeddings)
        _, _ = train_bilstm(model_lstm_128, train_loader_128, val_loader_128, EPOCHS)
        
        preds_lstm_128 = predict_bilstm(model_lstm_128, test_loader_128)
        metrics_128 = compute_metrics(np.array(test_data['label']), preds_lstm_128)
        print(f"BiLSTM (len=128) Test Macro-F1: {metrics_128['macro_f1']:.4f}")
        
        del model_lstm_128, train_loader_128, val_loader_128, test_loader_128
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    if args.model in ['all', 'distilbert']:
        print("\n" + "="*50)
        print("--- 3. DISTILBERT ---")
        print("="*50)
        X_train_bert = [preprocess_minimal(t) for t in train_data['text']]
        X_val_bert = [preprocess_minimal(t) for t in val_data['text']]
        X_test_bert = [preprocess_minimal(t) for t in test_data['text']]
        
        tokenizer, model_bert = init_distilbert()
        train_ds_bert = BertDataset(X_train_bert, list(train_data['label']), tokenizer)
        val_ds_bert = BertDataset(X_val_bert, list(val_data['label']), tokenizer)
        test_ds_bert = BertDataset(X_test_bert, list(test_data['label']), tokenizer)
        
        train_loader_bert = DataLoader(train_ds_bert, batch_size=BERT_BATCH_SIZE, shuffle=True)
        val_loader_bert = DataLoader(val_ds_bert, batch_size=BERT_BATCH_SIZE, shuffle=False)
        test_loader_bert = DataLoader(test_ds_bert, batch_size=BERT_BATCH_SIZE, shuffle=False)
        
        _, bert_history = train_distilbert(model_bert, train_loader_bert, val_loader_bert, EPOCHS)
        
        preds_bert = predict_distilbert(model_bert, test_loader_bert)
        final_metrics["DistilBERT"] = compute_metrics(np.array(test_data['label']), preds_bert)
        final_predictions["DistilBERT"] = preds_bert
        
    # If running all, generate complete reports
    if args.model == 'all':
        compare_models(final_metrics)
        from visualize import generate_report_plots, plot_learning_curves
        
        ablation_rows = [
            {'Model Variant': 'TF-IDF (Full preproc / Word-lvl)', 'Accuracy': final_metrics["TF-IDF"]['accuracy'], 'Macro-F1': final_metrics["TF-IDF"]['macro_f1']},
            {'Model Variant': 'TF-IDF (Minimal preproc)', 'Accuracy': abl_tfidf['accuracy'], 'Macro-F1': abl_tfidf['macro_f1']},
            {'Model Variant': 'TF-IDF (Subword/WordPiece)', 'Accuracy': subword_metrics['accuracy'], 'Macro-F1': subword_metrics['macro_f1']},
            {'Model Variant': 'BiLSTM (max_len=256)', 'Accuracy': final_metrics["BiLSTM_GloVe"]['accuracy'], 'Macro-F1': final_metrics["BiLSTM_GloVe"]['macro_f1']},
            {'Model Variant': 'BiLSTM (max_len=128)', 'Accuracy': metrics_128['accuracy'], 'Macro-F1': metrics_128['macro_f1']}
        ]
        
        ablation_df = pd.DataFrame(ablation_rows)
        print("\n" + "="*50)
        print("             ABLATION STUDIES RESULTS              ")
        print("="*50)
        print(ablation_df.to_string(index=False))
        
        generate_report_plots(final_metrics, ablation_df)
        plot_learning_curves(bilstm_history, bert_history)
        
    print(f"\n--- Pipeline Completed Successfully (Mode: {args.model}) ---")

if __name__ == "__main__":
    main()
