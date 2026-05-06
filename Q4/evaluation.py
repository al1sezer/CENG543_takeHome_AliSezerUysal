import evaluate
import torch
import gc
from config import BERTSCORE_MODEL, DEVICE

def compute_all_metrics(predictions, references):
    """
    Computes BLEU, METEOR, ChrF, and BERTScore.
    predictions: list of strings
    references: list of strings
    """
    results = {}
    
    # ── BLEU ──────────────────────────────────────────────────
    bleu = evaluate.load("bleu")
    bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])
    results['bleu'] = bleu_res['bleu']
    
    # ── METEOR ────────────────────────────────────────────────
    meteor = evaluate.load("meteor")
    meteor_res = meteor.compute(predictions=predictions, references=references)
    results['meteor'] = meteor_res['meteor']
    
    # ── ChrF ──────────────────────────────────────────────────
    chrf = evaluate.load("chrf")
    chrf_res = chrf.compute(predictions=predictions, references=[[r] for r in references])
    results['chrf'] = chrf_res['score'] / 100.0 # scale to 0-1
    
    # ── BERTScore ─────────────────────────────────────────────
    # Run BERTScore last as it stays on GPU
    bertscore = evaluate.load("bertscore")
    bs_res = bertscore.compute(
        predictions=predictions, 
        references=references, 
        model_type=BERTSCORE_MODEL,
        device=str(DEVICE)
    )
    results['bertscore_f1'] = sum(bs_res['f1']) / len(bs_res['f1'])
    
    # Cleanup GPU for BERTScore
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

def compute_per_sample_metrics(pred, ref):
    # Useful for qualitative analysis
    bleu = evaluate.load("bleu")
    b = bleu.compute(predictions=[pred], references=[[ref]])['bleu']
    
    meteor = evaluate.load("meteor")
    m = meteor.compute(predictions=[pred], references=[ref])['meteor']
    
    chrf = evaluate.load("chrf")
    c = chrf.compute(predictions=[pred], references=[[ref]])['score'] / 100.0
    
    return {'bleu': b, 'meteor': m, 'chrf': c}
