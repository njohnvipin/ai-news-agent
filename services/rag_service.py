"""
rag_service.py — Ingestion and retrieval pipeline.

Ingest flow:
  article text  →  chunk_text()  →  embed_chunks()  →  ChromaDB

Query flow:
  question  →  embed_query()  →  ChromaDB similarity search  →  Groq LLM  →  answer
"""
import hashlib
from typing import List

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME
from services.news_service import fetch_ai_articles
from services.embedding_service import embed_query, embed_chunks
from utils.text_utils import chunk_text, build_article_text
from utils.chroma_utils import get_collection, document_exists, get_collection_stats

# ── LLM ─────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

# ── Chunking settings ────────────────────────────────────────────────────────

CHUNK_SIZE = 300   # words per chunk
CHUNK_OVERLAP = 50  # overlapping words between consecutive chunks


# ── Helpers ──────────────────────────────────────────────────────────────────

def _article_id(article: dict) -> str:
    """
    Deterministic ID derived from the article URL.
    Re-ingesting the same article will produce the same root ID, so we can
    skip it instead of creating duplicates.
    """
    return hashlib.md5(article["link"].encode()).hexdigest()


def _chunk_id(article_id: str, chunk_index: int) -> str:
    return f"{article_id}_chunk_{chunk_index}"


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_latest_news() -> dict:
    """
    Fetch the latest AI news, chunk each article, embed the chunks, and
    store them in ChromaDB.  Articles that are already stored are skipped.
    """
    articles = fetch_ai_articles()

    if not articles:
        return {"status": "failed", "message": "No articles found", "count": 0}

    collection = get_collection()
    total_chunks_added = 0
    skipped_articles = 0

    for article in articles:
        article_id = _article_id(article)

        # Deduplication: skip if the first chunk of this article already exists
        if document_exists(_chunk_id(article_id, 0)):
            skipped_articles += 1
            continue

        full_text = build_article_text(article)
        chunks: List[str] = chunk_text(full_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        if not chunks:
            continue

        chunk_ids = [_chunk_id(article_id, i) for i in range(len(chunks))]
        metadatas = [
            {
                "title": article["title"],
                "link": article["link"],
                "published": article["published"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        embeddings = embed_chunks(chunks)

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=chunk_ids,
        )

        total_chunks_added += len(chunks)
        print(f"[rag_service] Ingested '{article['title']}' → {len(chunks)} chunk(s)")

    stats = get_collection_stats()
    return {
        "status": "success",
        "message": "Ingestion complete",
        "chunks_added": total_chunks_added,
        "articles_skipped_duplicates": skipped_articles,
        "total_chunks_in_db": stats["total_chunks"],
    }


def retrieve_relevant_news(question: str, n_results: int = 5) -> dict:
    """
    Embed the question and retrieve the most semantically similar chunks.
    """
    collection = get_collection()

    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    query_embedding = embed_query(question)

    # Cap n_results to what's actually stored
    n_results = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results


def ask_news_rag(question: str) -> dict:
    """
    Full RAG pipeline:
      question → embedding → ChromaDB → top chunks → Groq LLM → answer
    """
    results = retrieve_relevant_news(question)

    documents: List[str] = results.get("documents", [[]])[0]
    metadatas: List[dict] = results.get("metadatas", [[]])[0]
    distances: List[float] = results.get("distances", [[]])[0]

    if not documents:
        return {
            "answer": "I could not find relevant news articles. Please ingest news first.",
            "sources": [],
        }

    # De-duplicate by article link so the same article doesn't flood the context
    seen_links = set()
    unique_docs = []
    unique_metas = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        link = meta.get("link", "")
        if link not in seen_links:
            seen_links.add(link)
            unique_docs.append(doc)
            unique_metas.append(meta)

    context = "\n\n---\n\n".join(unique_docs)

    prompt = f"""You are an AI news assistant.

Answer the user's question using ONLY the news context provided below.
If the answer is not in the context, say:
"I don't know based on the available news articles."

Context:
{context}

Question:
{question}

Answer:"""

    response = llm.invoke(prompt)

    sources = [
        {
            "title": m.get("title"),
            "link": m.get("link"),
            "published": m.get("published"),
        }
        for m in unique_metas
    ]

    return {"answer": response.content, "sources": sources}


def get_stats() -> dict:
    """Return collection stats (useful for the Streamlit dashboard)."""
    return get_collection_stats()
