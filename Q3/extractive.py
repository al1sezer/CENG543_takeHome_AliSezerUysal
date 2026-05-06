"""
Q3 - Text Summarization: Extractive Summarization (TextRank)
Sentence ranking via TF-IDF similarity + NetworkX PageRank.
"""

import numpy as np
import networkx as nx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K_SENTENCES

nltk.download("punkt_tab", quiet=True)


def _split_sentences(article: str) -> list[str]:
    """Splits article into sentences and filters empty ones."""
    sentences = nltk.tokenize.sent_tokenize(article)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _build_similarity_matrix(sentences: list[str]) -> np.ndarray:
    """
    Creates cosine similarity matrix between sentences using TF-IDF vectors.

    Args:
        sentences: List of sentences

    Returns:
        np.ndarray: (N, N) similarity matrix, diagonal zeroed
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Remove self-loops — a sentence should not be compared with itself
    np.fill_diagonal(sim_matrix, 0.0)
    return sim_matrix


def _rank_sentences(sim_matrix: np.ndarray) -> list[tuple[int, float]]:
    """
    Builds a weighted graph from similarity matrix and applies PageRank.

    Returns:
        Each sentence's (index, score) pair, in descending order of score
    """
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=200, tol=1e-6)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def textrank_summarize(article: str) -> str:
    """
    Summarizes an article using TextRank.

    Algorithm:
        1. Split article into sentences
        2. Compute TF-IDF vectors
        3. Create similarity matrix via cosine similarity
        4. Run PageRank via NetworkX
        5. Select top K sentences
        6. Join according to original text order

    Args:
        article: Source text

    Returns:
        Extractive summary (union of selected sentences)
    """
    sentences = _split_sentences(article)

    # Safeguard for short articles
    if len(sentences) <= TOP_K_SENTENCES:
        return " ".join(sentences)

    # Handle cases where all sentences produce empty TF-IDF vectors
    # (rare, for defensive programming)
    try:
        sim_matrix = _build_similarity_matrix(sentences)
    except ValueError:
        return " ".join(sentences[:TOP_K_SENTENCES])

    ranked = _rank_sentences(sim_matrix)

    # Get indices of top K sentences
    k = min(TOP_K_SENTENCES, len(sentences))
    top_indices = sorted([idx for idx, _ in ranked[:k]])

    # Join in original order
    summary = " ".join(sentences[i] for i in top_indices)
    return summary


if __name__ == "__main__":
    test_article = (
        "The quick brown fox jumps over the lazy dog. "
        "Natural language processing is a field of artificial intelligence. "
        "Text summarization aims to reduce the length of a document. "
        "There are two main approaches: extractive and abstractive. "
        "Extractive methods select important sentences from the original text. "
        "Abstractive methods generate new sentences to capture the meaning."
    )
    result = textrank_summarize(test_article)
    print(f"Summary:\n{result}")
