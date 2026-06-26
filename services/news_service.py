import requests
import xml.etree.ElementTree as ET
from utils.text_utils import clean_text

RSS_URL = (
    "https://news.google.com/rss/search"
    "?q=artificial+intelligence+when:1h"
    "&hl=en-US&gl=US&ceid=US:en"
)
MAX_ARTICLES = 5
REQUEST_TIMEOUT = 10  # seconds


def fetch_ai_articles() -> list:
    """Fetch latest AI news from Google News RSS. Returns up to 5 articles."""
    try:
        response = requests.get(RSS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # raises error if 4xx or 5xx status
    except requests.exceptions.Timeout:
        print(" News fetch timed out")
        return []
    except requests.exceptions.RequestException as e:
        print(f" News fetch failed: {e}")
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f" Failed to parse RSS feed: {e}")
        return []

    articles = []
    seen_titles = set()

    for item in root.findall(".//item"):
        title_el = item.find("title")
        link_el  = item.find("link")
        date_el  = item.find("pubDate")
        desc_el  = item.find("description")

        # skip if any required field is missing
        if any(el is None for el in [title_el, link_el, date_el]):
            continue

        title = clean_text(title_el.text or "")

        # skip duplicates
        if title in seen_titles:
            continue
        seen_titles.add(title)

        articles.append({
            "title":       title,
            "link":        link_el.text or "",
            "published":   date_el.text or "",
            "description": clean_text(desc_el.text) if desc_el is not None else "",
        })

        if len(articles) >= MAX_ARTICLES:
            break

    return articles