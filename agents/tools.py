"""
tools.py — Tools the news agent can call.
"""
import logging

from langchain_core.tools import tool

from services.email_service import send_email
from services.news_service import fetch_ai_articles

logger = logging.getLogger(__name__)

DIGEST_MAX_ARTICLES = 10


def _format_article_list(articles: list) -> str:
    """Build the formatted article list with full URLs for the email."""
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']}")
        lines.append(f"   Link      : {a['link']}")
        lines.append(f"   Published : {a['published']}")
        lines.append("")
    return "\n".join(lines)


def _format_short_list(articles: list) -> str:
    """Build a short article list for the LLM — titles and dates only, no URLs."""
    return "\n".join(f"{i}. {a['title']} | {a['published']}" for i, a in enumerate(articles, 1))


def _build_email_body(article_list: str, insights: str) -> str:
    """Assemble the complete email body."""
    divider = "=" * 50
    return (
        f"Daily AI Insights\n{divider}\n\n"
        f"ARTICLES\n{divider}\n\n"
        f"{article_list}\n\n"
        f"OVERALL INSIGHTS\n{divider}\n\n"
        f"{insights.strip()}\n\n"
        f"{divider}\n"
        f"Best regards,\nValence Analytics"
    )


# Module-level cache shared between the two tools within one run
_articles_cache: list = []


@tool
def fetch_ai_news() -> str:
    """Fetch the latest AI news headlines. Returns titles and dates for analysis."""
    global _articles_cache
    logger.info("Tool called: fetch_ai_news")

    articles = fetch_ai_articles(max_articles=DIGEST_MAX_ARTICLES)
    if not articles:
        return "No AI news articles found at this time."

    _articles_cache = articles
    return _format_short_list(articles)


@tool
def send_email_tool(insights: str) -> str:
    """Send the AI insights email to all recipients.

    Args:
        insights: A 5 sentence paragraph analysing AI trends. No company names.
    """
    logger.info("Tool called: send_email_tool")

    if not _articles_cache:
        return "Error: No articles cached. Call fetch_ai_news first."
    if not insights or len(insights.strip()) < 10:
        return "Error: insights parameter is empty or too short."

    email_body = _build_email_body(_format_article_list(_articles_cache), insights)

    if not send_email(email_body):
        return "Error: the email failed to send. Do not retry; report this failure."
    return "Email sent successfully"
