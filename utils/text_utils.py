"""
text_utils.py — Text cleaning, chunking and article formatting.
"""
import html
import re
from typing import List


def clean_text(text: str) -> str:
    """Remove HTML tags, decode HTML entities and collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text:       The input text to split.
        chunk_size: Maximum number of words per chunk.
        overlap:    Number of words repeated at the start of the next chunk
                    so context is not lost at boundaries.

    Raises:
        ValueError: if chunk_size < 1, or overlap is negative or not smaller
                    than chunk_size (the loop would never move forward).
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    if not text or not text.strip():
        return []

    words = text.split()

    if len(words) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def build_article_header(article: dict) -> str:
    """
    Short header added to every chunk, e.g. "Title: Big news (TechCrunch)".

    It tells the embedding model which article a chunk belongs to. The link and
    date are kept out of it on purpose: they waste the embedding model's limited
    input length and don't help similarity search (they're stored as metadata).
    """
    title = (article.get("title") or "").strip()
    source = (article.get("source") or "").strip()
    return f"Title: {title} ({source})" if source else f"Title: {title}"


def build_article_documents(article: dict, chunk_size: int, overlap: int) -> List[str]:
    """
    Turn one article into the list of texts that get embedded and stored:
    the article body ("content", falling back to "description") is chunked,
    and each chunk is prefixed with the article header.
    """
    header = build_article_header(article)
    body = article.get("content") or article.get("description") or ""
    chunks = chunk_text(body, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return [header]
    return [f"{header}\n\n{chunk}" for chunk in chunks]


def strip_header(document: str) -> str:
    """Remove the header added by build_article_documents (for showing to the LLM)."""
    parts = document.split("\n\n", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("Title: ") else document


# ── Responsible-AI helpers ───────────────────────────────────────────────────

# Phrases that try to give orders to an AI. News articles almost never contain
# them, so a match is worth flagging (prompt-injection defense).
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"\b(reveal|print|show|repeat)\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"\bnew\s+(system\s+)?instructions\s*:",
    r"</?\s*source\b",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def find_injection_phrases(text: str) -> List[str]:
    """Return any instruction-like phrases found in the text."""
    return [m.group(0) for m in _INJECTION_RE.finditer(text or "")]


def neutralize_source_tags(text: str) -> str:
    """Stop article text from opening or closing the <source> delimiters used in the prompt."""
    return re.sub(r"<(/?)\s*source", r"‹\1source", text or "", flags=re.IGNORECASE)


def extract_citations(answer: str) -> List[int]:
    """All citation numbers in an answer, e.g. "x [1] y [2][3]" -> [1, 2, 3]."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer or "")})


def check_citations(answer: str, n_sources: int) -> List[str]:
    """
    Output check: warn when the answer cites sources that don't exist, or
    cites nothing at all.
    """
    cited = extract_citations(answer)
    if not cited:
        return ["The answer doesn't cite any source, so it can't be checked. Treat it with caution."]
    invalid = [n for n in cited if n < 1 or n > n_sources]
    if invalid:
        listed = ", ".join(f"[{n}]" for n in invalid)
        return [f"The answer cites {listed}, which doesn't match any source shown. That part may be made up."]
    return []
