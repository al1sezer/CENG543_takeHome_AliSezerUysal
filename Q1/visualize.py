import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def generate_report_plots(results, ablation_results, output_dir="Q1/reports"):
    """
    Generates professional plots for model comparison and ablation studies.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-muted') # Modern look
    
    # 1. Model Comparison Plot (Accuracy & Macro-F1)
    models = list(results.keys())
    acc_scores = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['macro_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, acc_scores, width, label='Accuracy', color='#3498db')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro-F1', color='#e74c3c')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (IMDb)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.0) # Zoom in to see differences
    ax.legend()
    
    # Function to add values on top of bars
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
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300)
    print(f"Plot saved: {output_dir}/model_comparison.png")
    
    # 2. Ablation Study Plot
    plt.clf()
    variants = ablation_results['Model Variant'].tolist()
    abl_acc = ablation_results['Accuracy'].tolist()
    
    plt.figure(figsize=(10, 6))
    plt.bar(variants, abl_acc, color='#3498db', width=0.5)
    plt.ylabel('Accuracy')
    plt.title('Ablation Studies: Impact of Tokenization & Preprocessing')
    plt.ylim(min(abl_acc) - 0.02, max(abl_acc) + 0.02)
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(abl_acc):
        plt.text(i, v + 0.001, f'{v:.4f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_study.png", dpi=300)
    print(f"Plot saved: {output_dir}/ablation_study.png")

def plot_learning_curves(bilstm_hist, bert_hist, output_dir="Q1/reports"):
    """
    Plots Loss and Validation F1 curves for BiLSTM and DistilBERT.
    Useful for identifying overfitting.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(bilstm_hist['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Training Loss Curve
    ax1.plot(epochs, bilstm_hist['train_loss'], 'o-', label='BiLSTM Loss', color='#e67e22')
    ax1.plot(epochs, bert_hist['train_loss'], 's-', label='DistilBERT Loss', color='#2ecc71')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Validation F1 Curve
    ax2.plot(epochs, bilstm_hist['val_f1'], 'o-', label='BiLSTM Val F1', color='#e67e22')
    ax2.plot(epochs, bert_hist['val_f1'], 's-', label='DistilBERT Val F1', color='#2ecc71')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title('Validation Performance per Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_curves.png", dpi=300)
    print(f"Plot saved: {output_dir}/learning_curves.png")

if __name__ == "__main__":
    # 1. Final Results
    model_results = {
        "TF-IDF": {"accuracy": 0.8909, "macro_f1": 0.8909},
        "BiLSTM (GloVe)": {"accuracy": 0.8848, "macro_f1": 0.8848},
        "DistilBERT": {"accuracy": 0.9134, "macro_f1": 0.9134}
    }
    
    # 2. Ablation Data
    ablation_df = pd.DataFrame([
        {'Model Variant': 'Full Preproc', 'Accuracy': 0.89092},
        {'Model Variant': 'Minimal Preproc', 'Accuracy': 0.89784}
    ])
    
    # 3. Training History (Extracted from your actual terminal log!)
    bilstm_history = {
        'train_loss': [0.3704, 0.1959, 0.0991, 0.0356, 0.0151],
        'val_f1': [0.8924, 0.8924, 0.8904, 0.8852, 0.8700]
    }
    
    bert_history = {
        'train_loss': [0.3448, 0.1975, 0.1169, 0.0641, 0.0374],
        'val_f1': [0.8840, 0.9072, 0.9088, 0.9104, 0.9128]
    }
    
    generate_report_plots(model_results, ablation_df)
    plot_learning_curves(bilstm_history, bert_history)

