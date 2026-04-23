from langchain.tools import tool
from services.news_service import fetch_ai_articles
from services.email_service import send_email

# store articles globally so send_email_tool can access full data
_articles_cache = []

@tool
def fetch_ai_news():
    """Fetch latest AI news articles and return titles and published dates only"""
    global _articles_cache
    print("\n fetch_ai_news tool was called!")
    
    articles = fetch_ai_articles()
    _articles_cache = articles  # save full articles for email

    # return SHORT version to agent — no URLs
    formatted = ""
    for i, a in enumerate(articles, 1):
        formatted += f"{i}. {a['title']} | {a['published']}\n"

    return formatted


@tool
def send_email_tool(insights: str):
    """Send AI insights email with article list and analysis"""
    global _articles_cache
    print("\n send_email_tool was called!")

    # build article list with full URLs here — not passed through agent
    article_list = ""
    for i, a in enumerate(_articles_cache, 1):
        article_list += f"{i}. {a['title']}\n"
        article_list += f"   Link      : {a['link']}\n"
        article_list += f"   Published : {a['published']}\n\n"

    final_output = f"""Daily AI Insights
{'='*50}

ARTICLES
{'='*50}

{article_list}

OVERALL INSIGHTS
{'='*50}

{insights}

{'='*50}
Best regards,
Valence Analytics"""

    send_email(final_output)
    return "Email sent successfully"