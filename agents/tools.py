"""
tools.py — The tools the digest agent can call.

  fetch_ai_news          today's headlines (Google News)
  search_knowledge_base  details from full articles stored by the RAG pipeline
  get_previous_digests   long-term memory: what earlier digests said
  save_draft             saves the finished briefing for HUMAN REVIEW

The agent cannot send email. Sending happens in app.py, only after a person
approves the draft (human-in-the-loop).

How tool calling works: the @tool decorator turns each function into a tool.
Its docstring becomes the description the LLM reads, and its typed arguments
become the input schema the LLM must fill in. What a tool returns is sent back
to the LLM as an observation.
"""
import logging
from typing import Optional

from langchain_core.tools import tool

from agents.memory import load_recent_digests
from config import ai_disclosure
from services.email_template import build_email_html
from services.news_service import fetch_ai_articles

logger = logging.getLogger(__name__)

DIGEST_MAX_ARTICLES = 10
SEARCH_MAX_RESULTS = 2      # articles returned per search
EXCERPT_CHARS = 500         # keeps each observation small (fewer tokens, fewer rate limits)
MAX_SEARCHES_PER_RUN = 3    # hard cap enforced in code, not just in the prompt
AGENT_MIN_SIMILARITY = 0.35 # stricter than the chat, so loosely related articles aren't passed off as a match


# ── Run state shared by the tools (kept out of the LLM's context) ────────────
# The article list (with URLs) goes into the email without passing through the
# LLM, so the model can't alter or invent links.
_articles_cache: list = []
_draft: Optional[dict] = None
_searches_done: int = 0


def reset_run_state() -> None:
    global _articles_cache, _draft, _searches_done
    _articles_cache, _draft, _searches_done = [], None, 0


def current_draft() -> Optional[dict]:
    """The latest saved draft: {"insights", "body", "headlines"}, or None."""
    return _draft


# ── Email formatting ─────────────────────────────────────────────────────────

def _format_article_list(articles: list) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']}")
        lines.append(f"   Link      : {a['link']}")
        lines.append(f"   Published : {a['published']}")
        lines.append("")
    return "\n".join(lines)


def _format_short_list(articles: list) -> str:
    """Titles and dates only — no URLs for the LLM."""
    return "\n".join(f"{i}. {a['title']} | {a['published']}" for i, a in enumerate(articles, 1))


def build_email_body(article_list: str, insights: str) -> str:
    """Assemble the email, including the AI-generated disclosure."""
    divider = "=" * 50
    return (
        f"Daily AI Insights\n{divider}\n\n"
        f"ARTICLES\n{divider}\n\n"
        f"{article_list}\n\n"
        f"OVERALL INSIGHTS\n{divider}\n\n"
        f"{insights.strip()}\n\n"
        f"{divider}\n"
        f"{ai_disclosure()}\n\n"
        f"Best regards,\nValence Analytics"
    )


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def fetch_ai_news() -> str:
    """Fetch today's AI news headlines. Returns a numbered list of titles and dates."""
    global _articles_cache
    logger.info("Tool called: fetch_ai_news")

    articles = fetch_ai_articles(max_articles=DIGEST_MAX_ARTICLES)
    if not articles:
        return "No AI news articles found at this time."
    _articles_cache = articles
    return _format_short_list(articles)


@tool
def search_knowledge_base(query: str) -> str:
    """Look up details about a news topic in the full articles stored by the app.

    Args:
        query: What to look up, e.g. the topic of one headline.
    """
    global _searches_done
    logger.info("Tool called: search_knowledge_base(%r)", query)

    # Guardrail in code: an LLM can ignore "search at most 3 times", it can't ignore this
    if _searches_done >= MAX_SEARCHES_PER_RUN:
        return ("Search limit reached. Do not search again. "
                "Write the briefing now from what you have and call save_draft.")
    _searches_done += 1
    remaining = MAX_SEARCHES_PER_RUN - _searches_done

    # Imported here so the embedding model only loads if the agent uses this tool
    from services.rag_service import group_by_article, retrieve_relevant_news
    from utils.text_utils import strip_header

    try:
        hits = [h for h in retrieve_relevant_news(query) if h["similarity"] >= AGENT_MIN_SIMILARITY]
        groups = group_by_article(hits)[:SEARCH_MAX_RESULTS]
    except Exception as e:
        logger.exception("Knowledge base search failed")
        return f"Error: knowledge base search failed ({e}). Continue with the headlines."

    if not groups:
        return (f"No closely matching article is stored for this topic. Don't retry it; "
                f"use the headline as it is. Searches left: {remaining}.")

    blocks = []
    for g in groups:
        meta = g["metadata"]
        excerpt = " [...] ".join(strip_header(h["text"]) for h in g["chunks"])[:EXCERPT_CHARS]
        blocks.append(f"- {meta.get('title')} ({meta.get('source', '')}):\n  {excerpt}")
    return "\n\n".join(blocks) + f"\n\nSearches left: {remaining}."


@tool
def get_previous_digests() -> str:
    """Read the most recent briefings that were sent, to spot continuing trends and avoid repeating them."""
    logger.info("Tool called: get_previous_digests")
    digests = load_recent_digests()
    if not digests:
        return "No previous digests yet. This is the first one."
    return "\n\n".join(f"Sent {d['sent_at']}:\n{d['insights']}" for d in digests)


@tool
def save_draft(insights: str) -> str:
    """Save the finished briefing so a human can review it. You cannot send email yourself.

    Args:
        insights: The briefing: exactly 5 sentences in one paragraph, no company names.
    """
    global _draft
    logger.info("Tool called: save_draft")

    if not _articles_cache:
        return "Error: no headlines loaded. Call fetch_ai_news first."
    if not insights or len(insights.strip()) < 10:
        return "Error: insights is empty or too short."

    _draft = {
        "insights": insights.strip(),
        "body": build_email_body(_format_article_list(_articles_cache), insights),
        "html": build_email_html(
            _articles_cache, insights,
            f"{ai_disclosure()}",
        ),
        "headlines": [a["title"] for a in _articles_cache],
    }
    return "Draft saved. A human will review it before anything is sent. You are done."


TOOLS = [fetch_ai_news, search_knowledge_base, get_previous_digests, save_draft]
