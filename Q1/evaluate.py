"""
CENG 467 - Q1: Evaluation & Error Analysis
Calculates accuracy, macro-f1, formats results, and identifies error patterns based on model predictions.
"""

from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates common metrics: Accuracy and Macro-F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {"accuracy": acc, "macro_f1": macro_f1}

def compare_models(results_dict: Dict[str, Dict[str, float]]):
    """
    Prints a formatted table for model comparison based on evaluation metrics.
    results_dict format: {"TF-IDF": {"accuracy": 0.88, "macro_f1": 0.88}, ...}
    """
    print("\n" + "="*50)
    print("             MODEL COMPARISON RESULTS             ")
    print("="*50)
    print(f"{'Model':<20} | {'Accuracy':<12} | {'Macro-F1':<12}")
    print("-" * 50)
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<20} | {metrics['accuracy']:.4f}       | {metrics['macro_f1']:.4f}")
    print("="*50 + "\n")

def error_analysis(texts: List[str], y_true: np.ndarray, preds_dict: Dict[str, np.ndarray], n: int = 5):
    """
    Identifies misclassified examples to find patterns.
    Looks for strategic errors (e.g. all fail, TF-IDF fails but BERT gets it right).
    """
    print("\n" + "="*50)
    print("                 ERROR ANALYSIS                   ")
    print("="*50)
    
    # We assume preds_dict has keys: 'TF-IDF', 'BiLSTM', 'DistilBERT'
    results = pd.DataFrame(preds_dict)
    results['True_Label'] = y_true
    results['Text'] = texts
    
    # Condition 1: All models failed (hard examples / noisy data)
    all_fail_mask = (results['TF-IDF'] != results['True_Label']) & \
                    (results['BiLSTM'] != results['True_Label']) & \
                    (results['DistilBERT'] != results['True_Label'])
    
    all_fail_df = results[all_fail_mask]
    
    # Condition 2: TF-IDF failed, DistilBERT succeeded (Contextual superiority)
    context_win_mask = (results['TF-IDF'] != results['True_Label']) & \
                       (results['DistilBERT'] == results['True_Label'])
    
    context_win_df = results[context_win_mask]
    
    selected_errors = []
    
    # Try to pick 2 hard failures
    if len(all_fail_df) > 0:
        for _, row in all_fail_df.head(min(2, len(all_fail_df))).iterrows():
            selected_errors.append((row['Text'], row['True_Label'], row['TF-IDF'], row['BiLSTM'], row['DistilBERT'], "All Models Failed"))
            
    # Try to pick 3 context wins
    if len(context_win_df) > 0:
        for _, row in context_win_df.head(min(3, len(context_win_df))).iterrows():
            selected_errors.append((row['Text'], row['True_Label'], row['TF-IDF'], row['BiLSTM'], row['DistilBERT'], "Context Win (TF-IDF Failed, BERT Passed)"))

    # Print out
    for i, (text, true_lbl, t_pred, b_pred, d_pred, category) in enumerate(selected_errors[:n]):
        print(f"\nExample {i+1} [{category}]")
        print(f"True Label: {'Positive' if true_lbl == 1 else 'Negative'}")
        print(f"Predictions -> TF-IDF: {t_pred} | BiLSTM: {b_pred} | DistilBERT: {d_pred}")
        
        # Print a snippet of text to avoid overwhelming console
        snippet = text[:500] + "..." if len(text) > 500 else text
        print(f"Text Snippet:\n{snippet}\n")
    
    print("="*50 + "\n")

    
def ablation_report(ablation_results: pd.DataFrame):
     print("\n" + "="*50)
     print("             ABLATION STUDY RESULTS              ")
     print("="*50)
     print(ablation_results.to_string(index=False))
     print("="*50 + "\n")
