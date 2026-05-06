from transformers import pipeline
import torch

try:
    print("Testing pipeline with task='summarization'...")
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Success with 'summarization'")
except Exception as e:
    print(f"Failed with 'summarization': {e}")

try:
    print("\nTesting pipeline without explicit task...")
    pipe = pipeline(model="facebook/bart-large-cnn")
    print(f"Success! Inferred task: {pipe.task}")
except Exception as e:
    print(f"Failed without task: {e}")

try:
    print("\nTesting pipeline with task='text-generation'...")
    pipe = pipeline("text-generation", model="facebook/bart-large-cnn")
    print(f"Success with 'text-generation'")
except Exception as e:
    print(f"Failed with 'text-generation': {e}")
