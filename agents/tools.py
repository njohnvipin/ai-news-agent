from langchain.tools import tool
from services.news_service import fetch_ai_articles
from services.email_service import send_email


def _format_article_list(articles: list) -> str:
    """Build formatted article list with full URLs for email."""
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']}")
        lines.append(f"   Link      : {a['link']}")
        lines.append(f"   Published : {a['published']}")
        lines.append("")
    return "\n".join(lines)


def _format_short_list(articles: list) -> str:
    """Build short article list for LLM — titles and dates only, no URLs."""
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']} | {a['published']}")
    return "\n".join(lines)


def _build_email_body(article_list: str, insights: str) -> str:
    """Assemble the complete email body."""
    divider = "=" * 50
    return (
        f"Daily AI Insights\n{divider}\n\n"
        f"ARTICLES\n{divider}\n\n"
        f"{article_list}\n\n"
        f"OVERALL INSIGHTS\n{divider}\n\n"
        f"{insights}\n\n"
        f"{divider}\n"
        f"Best regards,\nValence Analytics"
    )


# module-level cache — shared between tools in same run
_articles_cache: list = []


@tool
def fetch_ai_news() -> str:
    """Fetch the latest AI news headlines. Returns titles and dates for analysis."""
    global _articles_cache

    print(" Tool called: fetch_ai_news")

    articles = fetch_ai_articles()

    if not articles:
        return "No AI news articles found at this time."

    _articles_cache = articles
    result = _format_short_list(articles)

    print(f" -> Fetched {len(articles)} articles")
    return result


@tool
def send_email_tool(insights: str) -> str:
    """Send the AI insights email to all recipients.
    Args:
        insights: A 5 sentence paragraph analysing AI trends. No company names.
    """
    global _articles_cache

    print(" Tool called: send_email_tool")

    if not _articles_cache:
        return "Error: No articles cached. Call fetch_ai_news first."

    if not insights or len(insights.strip()) < 10:
        return "Error: insights parameter is empty or too short."

    article_list = _format_article_list(_articles_cache)
    email_body   = _build_email_body(article_list, insights)

    send_email(email_body)
    print(" -> Email dispatched")
    return "Email sent successfully"