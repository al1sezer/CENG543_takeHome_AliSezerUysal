import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from tabulate import tabulate

from config import (
    DEVICE, SEED, NUM_EPOCHS_SEQ2SEQ, NUM_EPOCHS_TRANSFORMER,
    SEQ2SEQ_EMB_DIM, SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_ENC_LAYERS, SEQ2SEQ_DEC_LAYERS,
    SEQ2SEQ_ENC_DROPOUT, SEQ2SEQ_DEC_DROPOUT, SEQ2SEQ_LR,
    TF_D_MODEL, TF_NHEAD, TF_NUM_ENC_LAYERS, TF_NUM_DEC_LAYERS,
    TF_D_FF, TF_DROPOUT, PAD_IDX, CLIP_GRAD
)
from data_loader import process_dataset, get_dataloaders, tokenize_de, tokenize_en
from seq2seq_model import Encoder, Decoder, BahdanauAttention, Seq2Seq
from transformer_model import MiniTransformer
from train import set_seed, train_model_orchestrator, transformer_lr_lambda
from inference import translate_sentence, translate_batch
from evaluation import compute_all_metrics, compute_per_sample_metrics
from visualization import plot_attention, plot_training_curves, plot_metrics_comparison

def main():
    set_seed(SEED)
    
    # ── 1. Data Preparation ────────────────────────────────────
    print("\n[1/7] Data Preparation...")
    dataset, src_vocab, tgt_vocab = process_dataset()
    train_loader, val_loader, test_loader = get_dataloaders(dataset, src_vocab, tgt_vocab, DEVICE)
    print(f"Vocab Sizes: DE={len(src_vocab)}, EN={len(tgt_vocab)}")
    
    # ── 2. Seq2Seq + Attention Training ──────────────────────────
    print("\n[2/7] Seq2Seq + Bahdanau Attention Training...")
    attn = BahdanauAttention(SEQ2SEQ_HIDDEN_DIM)
    enc = Encoder(len(src_vocab), SEQ2SEQ_EMB_DIM, SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_ENC_DROPOUT)
    dec = Decoder(len(tgt_vocab), SEQ2SEQ_EMB_DIM, SEQ2SEQ_HIDDEN_DIM, SEQ2SEQ_DEC_DROPOUT, attn)
    s2s_model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer_s2s = optim.Adam(s2s_model.parameters(), lr=SEQ2SEQ_LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    s2s_history = train_model_orchestrator(
        s2s_model, train_loader, val_loader, optimizer_s2s, criterion, 
        NUM_EPOCHS_SEQ2SEQ, CLIP_GRAD, is_transformer=False
    )
    plot_training_curves(s2s_history, "Seq2Seq_Attention", "seq2seq_training.png")
    
    # Load best model for evaluation
    s2s_model.load_state_dict(torch.load('best_model_seq2seq.pt', weights_only=True))
    
    # ── 3. Mini-Transformer Training ───────────────────────────
    print("\n[3/7] Mini-Transformer Training...")
    tf_model = MiniTransformer(
        len(src_vocab), len(tgt_vocab), TF_D_MODEL, TF_NHEAD,
        TF_NUM_ENC_LAYERS, TF_NUM_DEC_LAYERS, TF_D_FF, TF_DROPOUT, PAD_IDX
    ).to(DEVICE)
    
    optimizer_tf = optim.Adam(tf_model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # Note: lr=1.0 because lambda scheduler scales it
    scheduler_tf = optim.lr_scheduler.LambdaLR(optimizer_tf, transformer_lr_lambda)
    
    tf_history = train_model_orchestrator(
        tf_model, train_loader, val_loader, optimizer_tf, criterion,
        NUM_EPOCHS_TRANSFORMER, CLIP_GRAD, is_transformer=True, scheduler=scheduler_tf
    )
    plot_training_curves(tf_history, "Mini_Transformer", "transformer_training.png")
    
    # Load best model for evaluation
    tf_model.load_state_dict(torch.load('best_model_transformer.pt'))

    # ── 4. Quantitative Evaluation ─────────────────────────────
    print("\n[4/7] Batch Translation and Metric Computation...")
    test_src = [ex["de"] for ex in dataset["test"]]
    test_ref = [ex["en"] for ex in dataset["test"]]
    
    print("Inference: Seq2Seq...")
    s2s_preds = []
    for sent in test_src:
        tokens, _ = translate_sentence(sent, s2s_model, src_vocab, tgt_vocab, tokenize_de, False)
        s2s_preds.append(" ".join(tokens))
        
    print("Inference: Transformer...")
    tf_preds = []
    for sent in test_src:
        tokens, _ = translate_sentence(sent, tf_model, src_vocab, tgt_vocab, tokenize_de, True)
        tf_preds.append(" ".join(tokens))
    
    print("Computing metrics...")
    s2s_metrics = compute_all_metrics(s2s_preds, test_ref)
    tf_metrics = compute_all_metrics(tf_preds, test_ref)
    
    plot_metrics_comparison(s2s_metrics, tf_metrics, "metrics_comparison.png")
    
    # ── 5. Qualitative Analysis (Attention Maps) ───────────────
    print("\n[5/7] Qualitative Analysis & Attention Mapping...")
    
    # Categorilla: 1. Simple, 2. Rare word, 3. Long dependency
    categories = [
        {"name": "Simple", "idx": 0},
        {"name": "Long", "idx": 13}, # Adjust indices manually for demonstration after seeing data if needed
        {"name": "Rare", "idx": 42}
    ]
    
    qual_results = []
    for cat in categories:
        src_sent = test_src[cat["idx"]]
        ref_sent = test_ref[cat["idx"]]
        
        s2s_toks, s2s_attn = translate_sentence(src_sent, s2s_model, src_vocab, tgt_vocab, tokenize_de, False)
        tf_toks, tf_attn = translate_sentence(src_sent, tf_model, src_vocab, tgt_vocab, tokenize_de, True)
        
        src_toks_viz = tokenize_de(src_sent)
        
        plot_attention(s2s_attn, src_toks_viz, s2s_toks, f"Seq2Seq Attention: {cat['name']}", f"attn_s2s_{cat['name'].lower()}.png")
        if tf_attn is not None:
             plot_attention(tf_attn, src_toks_viz, tf_toks, f"Transformer Attention: {cat['name']}", f"attn_tf_{cat['name'].lower()}.png")
        
        qual_results.append({
            "category": cat["name"],
            "source": src_sent,
            "reference": ref_sent,
            "seq2seq": " ".join(s2s_toks),
            "transformer": " ".join(tf_toks)
        })

    # ── 6. Final Reporting ─────────────────────────────────────
    print("\n" + "="*50)
    print("            FINAL COMPARISON RESULTS")
    print("="*50)
    
    headers = ["Metric", "Seq2Seq + Attention", "Mini-Transformer"]
    rows = []
    for k in s2s_metrics.keys():
        rows.append([k, f"{s2s_metrics[k]:.4f}", f"{tf_metrics[k]:.4f}"])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # ── 7. Save Results ────────────────────────────────────────
    final_output = {
        "metrics": {"seq2seq": s2s_metrics, "transformer": tf_metrics},
        "qualitative": qual_results
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print("\nResults saved to results.json and plots generated.")

if __name__ == "__main__":
    main()
