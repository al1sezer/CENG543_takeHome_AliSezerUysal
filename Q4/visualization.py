import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

def plot_attention(attention, sentence, predicted_sentence, title, save_path):
    # attention: [tgt_len, src_len]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    
    sns.heatmap(attention, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=sentence, yticklabels=predicted_sentence, 
                cbar=True, ax=ax, annot_kws={"size": 8})
    
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    plt.xticks(rotation=90)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(history, model_name, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(seq_metrics, tf_metrics, save_path):
    labels = list(seq_metrics.keys())
    seq_values = [seq_metrics[l] for l in labels]
    tf_values = [tf_metrics[l] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, seq_values, width, label='Seq2Seq + Attn')
    ax.bar(x + width/2, tf_values, width, label='Mini-Transformer')
    
    ax.set_ylabel('Score')
    ax.set_title('Metric Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
