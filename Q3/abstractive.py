"""
Q3 - Text Summarization: Abstractive Summarization (BART)
Inference pipeline with facebook/bart-large-cnn model.
Model lifecycle (load -> run -> delete) managed in this module.
ASCII used for Windows terminal compatibility.
"""

import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from config import MODEL_NAME, BATCH_SIZE, MAX_LENGTH, MIN_LENGTH, NUM_BEAMS


def load_bart():
    """
    Loads BART model and tokenizer to GPU (FP16).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to("cuda")
    return model, tokenizer


def bart_summarize(articles: list[str], model_tokenizer) -> list[str]:
    """
    Summarizes articles using BART. 
    Includes progress bar, attention_mask, and bos_token settings.
    """
    model, tokenizer = model_tokenizer
    summaries = []

    # Batch processing + Progress Bar
    for i in tqdm(range(0, len(articles), BATCH_SIZE), desc="Abstractive (BART) Processing"):
        batch_articles = articles[i : i + BATCH_SIZE]
        
        # Tokenize (includes padding and truncation)
        inputs = tokenizer(
            batch_articles, 
            max_length=1024, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        ).to("cuda")
        
        # Generate
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # Critical to prevent noise
                num_beams=NUM_BEAMS,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                early_stopping=True,
                forced_bos_token_id=0  # BART-CNN start token
            )
        
        # Decode
        batch_summaries = tokenizer.batch_decode(
            summary_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        summaries.extend(batch_summaries)

    return summaries


def cleanup_bart(model_tokenizer):
    """
    Removes BART model from GPU and clears VRAM.
    """
    model, tokenizer = model_tokenizer
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
