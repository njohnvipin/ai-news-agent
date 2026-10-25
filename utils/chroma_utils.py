"""
chroma_utils.py — ChromaDB helpers for the AI News Agent.

Provides collection initialisation, duplicate checks and basic statistics
so the rest of the codebase doesn't need to talk to chromadb directly.
"""
import logging
from pathlib import Path
from typing import Iterable, Optional, Set

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Anchored to the project folder, so the same database is used no matter
# which directory you start the app from.
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "ai_news"

# Module-level singleton so every import shares one client.
# Note: run the Streamlit app OR the API against this folder at one time,
# not both — two processes writing the same local DB can conflict.
_client: Optional[chromadb.ClientAPI] = None


def get_client() -> chromadb.ClientAPI:
    """Return (or create) the shared ChromaDB persistent client."""
    global _client
    if _client is None:
        logger.info("Opening ChromaDB at %s", CHROMA_PATH)
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection() -> Collection:
    """Return the ai_news collection, creating it if it does not exist."""
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def existing_ids(ids: Iterable[str]) -> Set[str]:
    """Return which of the given IDs are already stored (one batched lookup)."""
    ids = list(ids)
    if not ids:
        return set()
    result = get_collection().get(ids=ids, include=[])
    return set(result["ids"])


def distance_to_similarity(distance: float) -> float:
    """
    Convert a Chroma distance into cosine similarity (1 = identical, 0 = unrelated).

    Our embeddings are L2-normalized, so for Chroma's default "l2" space
    (squared L2 distance) cosine similarity = 1 - distance / 2.
    For a collection created with the "cosine" space it is 1 - distance.
    """
    metadata = get_collection().metadata or {}
    if metadata.get("hnsw:space") == "cosine":
        return 1.0 - distance
    return 1.0 - distance / 2.0


def delete_collection() -> None:
    """Drop and recreate the collection — a full reset."""
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # collection may not exist yet
        pass
    client.get_or_create_collection(name=COLLECTION_NAME)


def get_collection_stats() -> dict:
    """Return basic statistics about the stored collection."""
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": get_collection().count(),
        "chroma_path": CHROMA_PATH,
    }
