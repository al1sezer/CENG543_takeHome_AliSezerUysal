"""
Q3 - Text Summarization Visualization Script (English Version)
Reads results.json and generates 3 different category plots in English for LaTeX integration.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_charts():
    # Load results
    try:
        with open("results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: results.json not found!")
        return

    # Color Palette
    color_ext = '#3498db' # Blue
    color_abs = '#e74c3c' # Red

    def create_bar_chart(metrics, labels, title, filename, ylabel='Scores (0-1)'):
        ext_vals = [data["extractive"][m] for m in metrics]
        abs_vals = [data["abstractive"][m] for m in metrics]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, ext_vals, width, label='TextRank', color=color_ext, edgecolor='black', alpha=0.8)
        rects2 = ax.bar(x + width/2, abs_vals, width, label='BART', color=color_abs, edgecolor='black', alpha=0.8)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Chart saved: {filename}")
        plt.close()

    # 1. ROUGE Comparison
    create_bar_chart(
        ["rouge1", "rouge2", "rougeL"],
        ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
        "ROUGE Metrics Comparison",
        "rouge_comparison.png"
    )

    # 2. Other Metrics (BLEU, METEOR, BERTScore)
    create_bar_chart(
        ["bleu", "meteor", "bertscore_f1"],
        ["BLEU", "METEOR", "BERTScore"],
        "BLEU, METEOR, and BERTScore Comparison",
        "other_metrics_comparison.png"
    )

    # 3. Execution Time (Computational Cost)
    create_bar_chart(
        ["duration"],
        ["Total Time (s)"],
        "Computational Cost (Seconds)",
        "runtime_comparison.png",
        ylabel='Seconds'
    )

if __name__ == "__main__":
    plot_charts()
    print("\nAll charts have been saved as PNG in English.")
