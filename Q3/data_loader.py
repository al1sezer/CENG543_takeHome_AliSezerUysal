"""
Q3 - Text Summarization: Data Loader
Loads CNN/DailyMail dataset and creates an in-memory subset.
No physical disk storage used.
"""

from datasets import load_dataset
from config import DATASET_NAME, DATASET_VERSION, SPLIT, SUBSET_SIZE, SEED


def load_subset():
    """
    Loads a random subset from CNN/DailyMail test split.

    Returns:
        Dataset: Hugging Face Dataset object. Each sample includes:
            - article (str): Full news article
            - highlights (str): Human-written reference summary
            - id (str): Unique identifier
    """
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split=SPLIT)
    subset = dataset.shuffle(seed=SEED).select(range(SUBSET_SIZE))
    return subset


if __name__ == "__main__":
    subset = load_subset()
    print(f"Subset size: {len(subset)}")
    print(f"Fields: {subset.column_names}")
    print(f"\n--- Example Article (first 200 chars) ---")
    print(subset[0]["article"][:200])
    print(f"\n--- Example Highlights ---")
    print(subset[0]["highlights"])
