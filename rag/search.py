"""Search over the chunks: by meaning (vectors), by words (BM25), and the two combined."""
import re
from collections import defaultdict
from itertools import zip_longest

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking import Chunk
from rag.embed import Embedder


def tokenize(text: str) -> list[str]:
    # Keeps codes like "of-450h" and "e-26" as single tokens.
    return re.findall(r"[\w\-]+", text.lower())


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed([c.text for c in chunks], "passage")
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def vector_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks closest in meaning to the query, best first."""
        q = self.embedder.embed([query], "query")[0]
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:k]]

    def bm25_search(self, query: str, k: int = 50) -> list[int]:
        """Indexes of the k chunks with the best word match, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: sums 1 / (k + rank) over the lists and sorts by the total."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def is_current(chunk: Chunk) -> bool:
    """False for chunks from documents that were replaced by a newer version."""
    return chunk.meta.get("status", "current") != "superseded"


def rank(index: Index, query: str, mode: str = "vector", current_only: bool = False) -> list[int]:
    if mode == "vector":
        ranked = index.vector_search(query)
    elif mode == "hybrid":
        ranked = rrf([index.vector_search(query), index.bm25_search(query)])
    else:
        raise ValueError(f"unknown mode: {mode}")
    if current_only:
        ranked = [i for i in ranked if is_current(index.chunks[i])]
    return ranked


def search(index: Index, query: str, mode: str = "vector", top_k: int = 5,
           current_only: bool = False) -> list[Chunk]:
    return [index.chunks[i] for i in rank(index, query, mode, current_only)[:top_k]]


def search_many(index: Index, queries: list[str], mode: str = "vector", top_k: int = 5,
                current_only: bool = False) -> list[Chunk]:
    """One search per sub-question, each keeps its own top_k, and the lists are interleaved."""
    rankings = [rank(index, q, mode, current_only)[:top_k] for q in queries]
    merged = []
    for group in zip_longest(*rankings):
        for i in group:
            if i is not None and i not in merged:
                merged.append(i)
    return [index.chunks[i] for i in merged]
