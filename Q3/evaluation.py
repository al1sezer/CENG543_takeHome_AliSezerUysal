"""
Q3 - Text Summarization: Evaluation
Calculates ROUGE-1/2/L, BLEU, METEOR (CPU), and BERTScore (GPU) metrics.
BERTScore should be called after BART cleanup (VRAM management).
"""

import gc
import torch
import evaluate

from config import BERTSCORE_MODEL


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Calculates CPU-based metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, METEOR.
    """
    # ── ROUGE ────────────────────────────────────────────────
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references,
    )

    # ── BLEU ─────────────────────────────────────────────────
    # BLEU expects references as list of lists (multiple references per prediction possible)
    bleu = evaluate.load("bleu")
    # predictions: list[str], references: list[list[str]]
    bleu_result = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )

    # ── METEOR ───────────────────────────────────────────────
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(
        predictions=predictions,
        references=references,
    )

    return {
        "rouge1": rouge_results.get("rouge1", 0.0),
        "rouge2": rouge_results.get("rouge2", 0.0),
        "rougeL": rouge_results.get("rougeL", 0.0),
        "bleu": bleu_result.get("bleu", 0.0),
        "meteor": meteor_result.get("meteor", 0.0),
    }


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """
    Calculates BERTScore (GPU-based) using roberta-large.

    ⚠️ VRAM: Run this function after cleanup_bart().
    GPU is automatically cleared after computation.

    Args:
        predictions: Model-generated summaries
        references: Reference (gold) summaries

    Returns:
        dict: {"bertscore_precision": float, "bertscore_recall": float, "bertscore_f1": float}
    """
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL,
        batch_size=32,
    )

    # Calculate average scores (evaluate returns per-sample)
    avg_precision = sum(results["precision"]) / len(results["precision"])
    avg_recall = sum(results["recall"]) / len(results["recall"])
    avg_f1 = sum(results["f1"]) / len(results["f1"])

    # GPU cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "bertscore_precision": avg_precision,
        "bertscore_recall": avg_recall,
        "bertscore_f1": avg_f1,
    }


def compute_per_sample_rougeL(
    predictions: list[str], references: list[str]
) -> list[float]:
    """
    Calculates ROUGE-L F1 score for each sample.
    Used for qualitative analysis (qualitative.py).

    Args:
        predictions: Model-generated summaries
        references: Reference (gold) summaries

    Returns:
        list[float]: ROUGE-L F1 score for each sample
    """
    rouge = evaluate.load("rouge")
    scores = []
    for pred, ref in zip(predictions, references):
        result = rouge.compute(predictions=[pred], references=[ref])
        scores.append(result.get("rougeL", 0.0))
    return scores
