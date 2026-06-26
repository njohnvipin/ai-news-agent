"""
embedding_service.py — Encapsulates all embedding logic.

Responsibilities:
  - Load / cache the SentenceTransformer model once.
  - Embed a single query string.
  - Embed a batch of text chunks.
"""
from typing import Optional
from typing import List
from sentence_transformers import SentenceTransformer

# Use a lightweight but capable model.
# Swap to "all-mpnet-base-v2" for higher accuracy at the cost of speed.
MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — loaded once per process.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedding_service] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> List[float]:
    """
    Embed a single query string for retrieval.

    Returns:
        A list of floats (the embedding vector).
    """
    model = _get_model()
    vector = model.encode([text], normalize_embeddings=True)
    return vector[0].tolist()


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks in one batched call.

    Args:
        chunks: A list of text strings to embed.

    Returns:
        A list of embedding vectors (one per chunk).
    """
    if not chunks:
        return []
    model = _get_model()
    vectors = model.encode(chunks, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]
