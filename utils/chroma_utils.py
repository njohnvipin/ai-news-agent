"""
chroma_utils.py — ChromaDB helpers for the AI News Agent.

Provides collection initialisation, deduplication, and basic
collection statistics so the rest of the codebase doesn't need
to talk to the chromadb client directly.
"""
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ai_news"

# module-level singleton so every import shares one client
_client: Optional[chromadb.PersistentClient] = None


def get_client() -> chromadb.PersistentClient:
    """Return (or create) the shared ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection() -> Collection:
    """Return the ai_news collection, creating it if it does not exist."""
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def collection_count() -> int:
    """Return the total number of documents stored in the collection."""
    return get_collection().count()


def document_exists(doc_id: str) -> bool:
    """
    Check whether a document with the given ID already exists.
    Used to prevent duplicate ingestion.
    """
    try:
        result = get_collection().get(ids=[doc_id])
        return len(result["ids"]) > 0
    except Exception:
        return False


def delete_collection() -> None:
    """
    Drop and recreate the collection — useful for a full reset
    (e.g. nightly cron jobs that want a fresh slate).
    """
    client = get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # collection may not exist yet
    client.get_or_create_collection(name=COLLECTION_NAME)


def get_collection_stats() -> dict:
    """Return basic statistics about the stored collection."""
    col = get_collection()
    count = col.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": count,
        "chroma_path": CHROMA_PATH,
    }
