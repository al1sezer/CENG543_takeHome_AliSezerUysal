"""
Q3 - Text Summarization: Main Orchestrator
Orchestrates all modules, ensures sequential VRAM management, and reports results.
ASCII used for Windows CP1254 compatibility.
"""

import time
import torch
import gc
import json
from tabulate import tabulate

# Import modules
from config import SUBSET_SIZE
from data_loader import load_subset
from extractive import textrank_summarize
from abstractive import load_bart, bart_summarize, cleanup_bart
from evaluation import compute_metrics, compute_bertscore
from qualitative import select_examples, print_analysis

def main():
    print("Starting Q3 - Text Summarization...")
    print(f"Configuration: {SUBSET_SIZE} samples (subset), TextRank (Extractive) vs BART (Abstractive)")
    
    # ── 1. DATA LOADING ───────────────────────────────────────────────────
    print("\n[1/6] Loading dataset (CNN/DailyMail)...")
    subset = load_subset()
    articles = subset["article"]
    references = subset["highlights"]
    print(f"OK. {len(subset)} samples successfully loaded.")

    # ── 2. EXTRACTIVE SUMMARIZATION (TextRank) ──────────────────────────────────
    print("\n[2/6] Running Extractive (TextRank) summarization (CPU)...")
    start_time = time.time()
    ext_summaries = [textrank_summarize(article) for article in articles]
    ext_duration = time.time() - start_time
    print(f"OK. Completed. Time: {ext_duration:.2f} seconds")

    # ── 3. ABSTRACTIVE SUMMARIZATION (BART) ────────────────────────────────────
    print("\n[3/6] Running Abstractive (BART) summarization (GPU)...")
    start_time = time.time()
    
    # Load model
    summarizer = load_bart()
    # Summarize
    abs_summaries = bart_summarize(articles, summarizer)
    # Cleanup (clear VRAM!)
    cleanup_bart(summarizer)
    
    abs_duration = time.time() - start_time
    print(f"OK. Completed. Time: {abs_duration:.2f} seconds")

    # ── 4. EVALUATION (CPU Metrics) ──────────────────────────────────
    print("\n[4/6] Computing CPU metrics (ROUGE, BLEU, METEOR)...")
    ext_metrics = compute_metrics(ext_summaries, references)
    abs_metrics = compute_metrics(abs_summaries, references)
    print("OK. Completed.")

    # ── 5. EVALUATION (GPU Metric: BERTScore) ──────────────────────────
    # Space available on GPU as BART is cleaned up.
    print("\n[5/6] Computing GPU metric (BERTScore)...")
    ext_bert = compute_bertscore(ext_summaries, references)
    abs_bert = compute_bertscore(abs_summaries, references)
    print("OK. Completed.")

    # ── 6. QUALITATIVE ANALYSIS ───────────────────────────────────────────────────
    print("\n[6/6] Selecting qualitative analysis samples...")
    examples = select_examples(articles, references, ext_summaries, abs_summaries)
    print_analysis(examples)

    # ── FINAL RESULTS ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("                      FINAL COMPARATIVE RESULTS")
    print("="*80)
    
    headers = ["Metric", "TextRank (Extractive)", "BART (Abstractive)"]
    table_data = [
        ["ROUGE-1 (F1)", f"{ext_metrics['rouge1']:.4f}", f"{abs_metrics['rouge1']:.4f}"],
        ["ROUGE-2 (F1)", f"{ext_metrics['rouge2']:.4f}", f"{abs_metrics['rouge2']:.4f}"],
        ["ROUGE-L (F1)", f"{ext_metrics['rougeL']:.4f}", f"{abs_metrics['rougeL']:.4f}"],
        ["BLEU", f"{ext_metrics['bleu']:.4f}", f"{abs_metrics['bleu']:.4f}"],
        ["METEOR", f"{ext_metrics['meteor']:.4f}", f"{abs_metrics['meteor']:.4f}"],
        ["BERTScore (F1)", f"{ext_bert['bertscore_f1']:.4f}", f"{abs_bert['bertscore_f1']:.4f}"],
        ["-"*15, "-"*20, "-"*20],
        ["Total Time (s)", f"{ext_duration:.2f}", f"{abs_duration:.2f}"],
        ["Time per Sample (s)", f"{ext_duration/len(subset):.4f}", f"{abs_duration/len(subset):.4f}"],
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # SAVE RESULTS (for reporting and visualization)
    final_output = {
        "extractive": {**ext_metrics, **ext_bert, "duration": ext_duration},
        "abstractive": {**abs_metrics, **abs_bert, "duration": abs_duration}
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    
    print("\nResults saved to results.json.")
    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
