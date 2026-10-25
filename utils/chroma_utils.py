"""
chroma_utils.py — ChromaDB helpers for the AI News Agent.

Provides collection access, duplicate checks and statistics so the rest of
the codebase doesn't need to talk to chromadb directly. Every function takes
an optional collection name, so the evaluation script can use its own
collection without touching your real data.
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

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


def get_collection(name: str = COLLECTION_NAME) -> Collection:
    """Return the collection, creating it if it does not exist."""
    return get_client().get_or_create_collection(name=name)


def existing_ids(ids: Iterable[str], collection_name: str = COLLECTION_NAME) -> Set[str]:
    """Return which of the given IDs are already stored (one batched lookup)."""
    ids = list(ids)
    if not ids:
        return set()
    result = get_collection(collection_name).get(ids=ids, include=[])
    return set(result["ids"])


def distance_to_similarity(distance: float, collection: Optional[Collection] = None) -> float:
    """
    Convert a Chroma distance into cosine similarity (1 = identical, 0 = unrelated).

    Our embeddings are L2-normalized, so for Chroma's default "l2" space
    (squared L2 distance) cosine similarity = 1 - distance / 2.
    For a collection created with the "cosine" space it is 1 - distance.
    """
    metadata = (collection.metadata if collection is not None else None) or {}
    if metadata.get("hnsw:space") == "cosine":
        return 1.0 - distance
    return 1.0 - distance / 2.0


def delete_collection(name: str = COLLECTION_NAME, recreate: bool = True) -> None:
    """Drop the collection — a full reset. By default an empty one is created again."""
    client = get_client()
    try:
        client.delete_collection(name=name)
    except Exception:  # collection may not exist yet
        pass
    if recreate:
        client.get_or_create_collection(name=name)


def get_source_counts(name: str = COLLECTION_NAME) -> Dict[str, int]:
    """Number of stored ARTICLES per publisher, largest first (source diversity)."""
    result = get_collection(name).get(where={"chunk_index": 0}, include=["metadatas"])
    counts = Counter((m or {}).get("source") or "Other" for m in result["metadatas"])
    return dict(counts.most_common())


def get_collection_stats(name: str = COLLECTION_NAME) -> dict:
    """Return basic statistics about the stored collection."""
    return {
        "collection_name": name,
        "total_chunks": get_collection(name).count(),
        "chroma_path": CHROMA_PATH,
    }
