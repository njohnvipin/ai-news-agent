"""
embedding_service.py — Encapsulates all embedding logic.

Responsibilities:
  - Load / cache the SentenceTransformer model once.
  - Embed a single query string.
  - Embed a batch of text chunks.
"""
import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lightweight but capable model.
# Swap to "all-mpnet-base-v2" for higher accuracy at the cost of speed.
# (If you change it, delete the chroma_db/ folder and re-ingest: vectors from
# different models can't be compared.)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Module-level singleton — loaded once per process.
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(text: str) -> List[float]:
    """Embed a single query string. Returns one L2-normalized vector."""
    vector = _get_model().encode([text], normalize_embeddings=True)
    return vector[0].tolist()


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embed a list of text chunks in batches. Returns one vector per chunk."""
    if not chunks:
        return []
    vectors = _get_model().encode(chunks, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]
