import re
import html
from typing import List


def clean_text(text: str) -> str:
    """Remove HTML tags and decode HTML entities from text."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       The input text to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of words to repeat at the start of the next chunk
                    so context is not lost at boundaries.

    Returns:
        A list of text chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        # move forward by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` words for continuity
        start += chunk_size - overlap

    return chunks


def build_article_text(article: dict) -> str:
    """
    Combine article fields into a single text block ready for chunking.
    """
    parts = []

    title = article.get("title", "").strip()
    if title:
        parts.append(f"Title: {title}")

    description = article.get("description", "").strip()
    if description:
        parts.append(f"Description: {description}")

    published = article.get("published", "").strip()
    if published:
        parts.append(f"Published: {published}")

    link = article.get("link", "").strip()
    if link:
        parts.append(f"Link: {link}")

    return "\n".join(parts)
