"""
Q3 - Text Summarization: Qualitative Analysis
Programmatic sample selection and analysis output based on per-sample ROUGE-L scores.

Standalone usage:
    python qualitative.py
    -> Loads dataset, performs TextRank and BART inference,
       selects top 3 samples and saves to qualitative_results.txt.

Module usage (called by main.py):
    from qualitative import select_examples, print_analysis
"""

from evaluation import compute_per_sample_rougeL


def select_examples(
    articles: list[str],
    references: list[str],
    ext_summaries: list[str],
    abs_summaries: list[str],
) -> list[dict]:
    """
    Selects 3 qualitative samples based on per-sample ROUGE-L scores.

    Strategy:
        Example 1: Extractive wins (ext >> abs)
        Example 2: Abstractive wins (abs >> ext)
        Example 3: Both models are weak (both low)

    Returns:
        list[dict]: 3 dictionaries, each with the following keys:
            - index, label, article, reference,
              ext_summary, abs_summary, ext_rougeL, abs_rougeL
    """
    ext_scores = compute_per_sample_rougeL(ext_summaries, references)
    abs_scores = compute_per_sample_rougeL(abs_summaries, references)

    # Difference vectors
    diffs = [e - a for e, a in zip(ext_scores, abs_scores)]
    combined = [e + a for e, a in zip(ext_scores, abs_scores)]

    # Example 1: Largest ext - abs (extractive wins)
    idx_ext_wins = max(range(len(diffs)), key=lambda i: diffs[i])

    # Example 2: Largest abs - ext (abstractive wins)
    idx_abs_wins = min(range(len(diffs)), key=lambda i: diffs[i])

    # Example 3: Lowest total ROUGE-L (both weak)
    # Exclude already selected indices
    used = {idx_ext_wins, idx_abs_wins}
    remaining = [i for i in range(len(combined)) if i not in used]
    idx_both_weak = min(remaining, key=lambda i: combined[i])

    examples = []
    for idx, label in [
        (idx_ext_wins, "Extractive Wins"),
        (idx_abs_wins, "Abstractive Wins"),
        (idx_both_weak, "Both Models Weak"),
    ]:
        examples.append(
            {
                "index": idx,
                "label": label,
                "article": articles[idx],
                "reference": references[idx],
                "ext_summary": ext_summaries[idx],
                "abs_summary": abs_summaries[idx],
                "ext_rougeL": ext_scores[idx],
                "abs_rougeL": abs_scores[idx],
            }
        )

    return examples


def format_example(i: int, ex: dict) -> str:
    """
    Formats a single example into a readable text block.
    Used for both console and file output.
    """
    separator = "=" * 80
    lines = []
    lines.append(f"\n{separator}")
    lines.append(f"  EXAMPLE {i}: {ex['label']}")
    lines.append(f"  (Dataset index: {ex['index']})")
    lines.append(separator)

    art_snippet = ex["article"][:800]
    if len(ex["article"]) > 800:
        art_snippet += "..."
    lines.append(f"\n[SOURCE ARTICLE] (first 800 chars):\n  {art_snippet}")
    lines.append(f"\n[REFERENCE (GOLD)]:\n  {ex['reference']}")
    lines.append(f"\n[TEXTRANK (EXTRACTIVE)] (ROUGE-L: {ex['ext_rougeL']:.4f}):\n  {ex['ext_summary']}")
    lines.append(f"\n[BART (ABSTRACTIVE)] (ROUGE-L: {ex['abs_rougeL']:.4f}):\n  {ex['abs_summary']}")

    lines.append(f"\n[COMPARISON TABLE]:")
    lines.append(f"  {'Dimension':<25} {'TextRank':<20} {'BART':<20}")
    lines.append(f"  {'-'*65}")
    lines.append(f"  {'ROUGE-L':<25} {ex['ext_rougeL']:<20.4f} {ex['abs_rougeL']:<20.4f}")
    lines.append(f"  {'Fluency':<25} {'Original sentences':<20} {'Fluent generation':<20}")
    lines.append(f"  {'Factual Consistency':<25} {'No hallucination':<20} {'Risk present':<20}")
    lines.append(f"  {'Information Coverage':<25} {'Limited (K sents)':<20} {'Compressed':<20}")

    return "\n".join(lines)


def print_analysis(examples: list[dict]):
    """
    Prints detailed comparison table for selected samples.
    Called by main.py (backward-compatible).
    """
    for i, ex in enumerate(examples, 1):
        print(format_example(i, ex))
    print(f"\n{'=' * 80}")


def save_analysis(examples: list[dict], filepath: str = "qualitative_results.txt"):
    """
    Saves selected samples to a file (for reproducibility).
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Q3 - Text Summarization: Qualitative Analysis Results\n")
        f.write("Generated from CNN/DailyMail test set (seed=42)\n")
        f.write("Selection: per-sample ROUGE-L score differences\n\n")
        for i, ex in enumerate(examples, 1):
            f.write(format_example(i, ex))
            f.write("\n")
        f.write(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    from data_loader import load_subset
    from extractive import textrank_summarize
    from abstractive import load_bart, bart_summarize, cleanup_bart

    # 30 samples are enough for qualitative analysis.
    # BART inference takes ~12 min for 1000 samples, ~20 sec for 30 samples.
    QUAL_SUBSET = 30

    print("=" * 60)
    print("  Q3 Qualitative Analysis - Standalone Runner")
    print(f"  (Inference on {QUAL_SUBSET} samples for efficiency)")
    print("=" * 60)

    # 1. Load data (same seed, same subset — take first N)
    print("\n[1/4] Loading CNN/DailyMail test subset...")
    subset = load_subset()
    articles = list(subset["article"][:QUAL_SUBSET])
    references = list(subset["highlights"][:QUAL_SUBSET])
    print(f"  OK. Using first {QUAL_SUBSET} samples from the 1000-sample subset.")

    # 2. Extractive inference (CPU, fast)
    print("\n[2/4] Running TextRank (Extractive)...")
    ext_summaries = [textrank_summarize(a) for a in articles]
    print("  OK.")

    # 3. Abstractive inference (GPU)
    print("\n[3/4] Running BART (Abstractive)...")
    model_tok = load_bart()
    abs_summaries = bart_summarize(articles, model_tok)
    cleanup_bart(model_tok)
    print("  OK.")

    # 4. Select, print, and save examples
    print("\n[4/4] Selecting best qualitative examples via ROUGE-L...")
    examples = select_examples(articles, references, ext_summaries, abs_summaries)

    print_analysis(examples)
    save_analysis(examples)
    print("\nResults saved to: qualitative_results.txt")

