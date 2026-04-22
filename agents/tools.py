from langchain.tools import tool
from services.news_service import fetch_ai_articles
from services.email_service import send_email


@tool
def fetch_ai_news():
    """Fetch latest AI news articles with title, summary, link and published date"""
    articles = fetch_ai_articles()

    formatted = ""
    for i, a in enumerate(articles, 1):
        formatted += f"{i}. {a['title']}\n"
        formatted += f"   Summary   : {a['description']}\n"
        formatted += f"   Link      : {a['link']}\n"
        formatted += f"   Published : {a['published']}\n\n"

    return formatted


@tool
def send_email_tool(content: str):
    """Send AI insights email"""
    print("\n🔧 send_email_tool was called!")   # ← add this

    final_output = f"""Daily AI Insights
{'='*50}

{content}

{'='*50}
Best regards,
Valence Analytics
"""
    send_email(final_output)
    return "Email sent successfully"