import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

def parse_evaluation_results(file_path):
    models = []
    scores = []
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return [], []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Find table rows
    for line in lines:
        if "║" in line and "Test PPL" not in line and "Model" not in line and "ABLATION" not in line:
            parts = line.split("║")
            if len(parts) >= 3:
                model_name = parts[1].strip()
                try:
                    score = float(parts[2].strip())
                    
                    # Shorten names/add newlines for better visualization
                    if "Trigram" in model_name:
                        model_name = "N-Gram\n(Baseline)"
                    elif "Champion" in model_name:
                        model_name = "LSTM\n(Champion)"
                    elif "Tuned" in model_name:
                        model_name = "LSTM\n(Tuned 512x4)"
                    elif "AdamW" in model_name:
                        model_name = "LSTM\n(AdamW)"
                    elif "Momentum" in model_name:
                        model_name = "LSTM\n(SGD+Mom)"
                    elif "B64" in model_name:
                        model_name = "LSTM\n(SGD B64)"
                        
                    models.append(model_name)
                    scores.append(score)
                except ValueError:
                    continue
                    
    # Sort by scores descending for aesthetic descent in the plot
    if models and scores:
        sorted_pairs = sorted(zip(scores, models), reverse=True)
        scores, models = zip(*sorted_pairs)
        
    return list(models), list(scores)

def create_ablation_plot():
    current_dir = os.path.dirname(__file__)
    eval_file = os.path.join(current_dir, 'evaluation_results.txt')
    
    models, ppl_scores = parse_evaluation_results(eval_file)
    
    if not models:
        print("No data parsed from evaluation_results.txt")
        return
        
    # Set a modern academic style
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))
    
    # Set color palette (Grey for Baseline, Dark Green for Champion, Blue for others)
    colors = []
    for model in models:
        if "Baseline" in model:
            colors.append('#95a5a6')  # Gri
        elif "Champion" in model:
            colors.append('#27ae60')  # Yeşil
        else:
            colors.append('#3498db')  # Mavi
            
    # Create bar plot
    bars = plt.bar(models, ppl_scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Print exact value on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Set axes and title
    plt.title('Language Modeling: Test Perplexity Comparison (WikiText-2)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Test Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture & Optimizer Settings', fontsize=12, fontweight='bold')
    plt.xticks(rotation=0, fontsize=11)
    
    # Set Y-axis limit dynamically
    plt.ylim(0, max(ppl_scores) * 1.15)
    
    # Adjust and save
    plt.tight_layout()
    
    output_path = os.path.join(current_dir, 'q5_ablation_ppl.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    create_ablation_plot()
