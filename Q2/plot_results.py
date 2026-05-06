import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clean design: high DPI, consistent colors, removed grid clutter.
sns.set_theme(style="whitegrid", context="talk")
MODEL_COLORS = {"CRF (Baseline)": "#E74C3C", "BiLSTM-CRF": "#F39C12", "DistilBERT": "#2E86C1"}

def parse_report(filepath, model_name):
    """
    Parse .txt classification report from seqeval.
    Defensive handling for missing/invalid files.
    """
    data = []
    if not os.path.exists(filepath):
        print(f"[WARNING] {filepath} not found. Skipping.")
        return data

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            # Split lines
            parts = line.split()
            # Valid metric lines (entities or averages)
            if len(parts) >= 5 and parts[0] in ['LOC', 'MISC', 'ORG', 'PER']:
                data.append({
                    "Model": model_name,
                    "Entity": parts[0],
                    "Precision": float(parts[1]),
                    "Recall": float(parts[2]),
                    "F1-Score": float(parts[3])
                })
            elif len(parts) >= 5 and parts[0] == 'weighted' and parts[1] == 'avg':
                data.append({
                    "Model": model_name,
                    "Entity": "Overall (Weighted)",
                    "Precision": float(parts[2]),
                    "Recall": float(parts[3]),
                    "F1-Score": float(parts[4])
                })
    except Exception as e:
        print(f"[ERROR] Error processing {filepath}: {e}")
        
    return data

def main():
    print("Reading and parsing results...")
    
    # Paths and naming
    files = {
        "CRF (Baseline)": "crf_results.txt",
        "BiLSTM-CRF": "bilstm_crf_results.txt",
        "DistilBERT": "transformer_results.txt"
    }
    
    all_data = []
    for model_name, filepath in files.items():
        all_data.extend(parse_report(filepath, model_name))
        
    if not all_data:
        print("[ERROR] No data found. Check txt files.")
        return
        
    df = pd.DataFrame(all_data)

    # ---------------------------------------------------------
    # CHART 1: Overall Performance Comparison
    # ---------------------------------------------------------
    print("Creating Chart 1: Overall Performance...")
    df_overall = df[df['Entity'] == 'Overall (Weighted)'].melt(
        id_vars=['Model'], value_vars=['Precision', 'Recall', 'F1-Score'], 
        var_name='Metric', value_name='Score'
    )
    
    plt.figure(figsize=(10, 6))
    g = sns.barplot(data=df_overall, x='Metric', y='Score', hue='Model', palette=MODEL_COLORS)
    plt.title("Overall Performance Comparison of Models (Weighted Avg)", pad=20, fontweight='bold')
    plt.ylim(0.60, 1.0)
    plt.ylabel("Score")
    plt.xlabel("")
    
    # Annotate bars
    for p in g.patches:
        g.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points', fontsize=10)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("q2_1_overall_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # CHART 2: F1-Scores by Entity Type
    # ---------------------------------------------------------
    print("Creating Chart 2: Entity-Level F1...")
    df_entities = df[df['Entity'] != 'Overall (Weighted)']
    
    plt.figure(figsize=(12, 6))
    g2 = sns.barplot(data=df_entities, x='Entity', y='F1-Score', hue='Model', palette=MODEL_COLORS)
    plt.title("F1-Scores by Entity Type", pad=20, fontweight='bold')
    plt.ylim(0.60, 1.0)
    plt.ylabel("F1-Score")
    plt.xlabel("Entity Type")
    
    # Annotate to highlight performance gaps
    for p in g2.patches:
        g2.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points', fontsize=10)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("q2_2_entity_f1_scores.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # CHART 3: Precision vs Recall Balance
    # ---------------------------------------------------------
    print("Creating Chart 3: Precision vs Recall Scatter Plot...")
    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(data=df_entities, x='Precision', y='Recall', hue='Model', 
                    style='Entity', s=200, palette=MODEL_COLORS, alpha=0.8)
    
    # Perfect score reference line
    plt.plot([0.6, 1.0], [0.6, 1.0], ls="--", c=".3", alpha=0.5, label="Perfect Balance (P=R)")
    
    plt.title("Precision - Recall Balance (per Entity)", pad=20, fontweight='bold')
    plt.xlim(0.65, 1.0)
    plt.ylim(0.65, 1.0)
    
    # Efsaneyi (Legend) saga disari tasi
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Model & Entity")
    plt.tight_layout()
    plt.savefig("q2_3_precision_recall_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Done! 3 PNG charts generated.")

if __name__ == "__main__":
    main()
