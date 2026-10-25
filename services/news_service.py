"""
news_service.py — Fetches AI news.

Two sources:

1. Publisher RSS feeds + full article text  →  used by RAG ingestion
   fetch_feed_articles()  lists recent articles from AI news feeds (cheap)
   add_full_text()        downloads each article and extracts the body text
                          with trafilatura (only call it for NEW articles)

2. Google News RSS (headlines only)          →  used by the email digest agent
   fetch_ai_articles()
"""
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import feedparser
import requests
import trafilatura

from utils.text_utils import clean_text

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15  # seconds
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ai-news-agent"
    )
}

# ═══════════════════════════════════════════════════════════════════════════
# 1. Publisher feeds + full text (RAG ingestion)
# ═══════════════════════════════════════════════════════════════════════════

# AI-specific feeds with direct article links. Add or remove freely:
# any RSS or Atom feed works.
AI_FEEDS = {
    "TechCrunch":            "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge":             "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Technology Review": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "Ars Technica":          "https://arstechnica.com/ai/feed/",
}

FEED_MAX_AGE_DAYS = 3     # ignore feed items older than this
PER_FEED_LIMIT = 15       # max items read from each feed
MIN_FULL_TEXT_WORDS = 120 # extracted text shorter than this is treated as a failure
DOWNLOAD_WORKERS = 8      # parallel downloads


def _entry_datetime(entry) -> Optional[datetime]:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)


def _longest_feed_content(entry) -> str:
    """Some feeds (e.g. Atom feeds) include the full article body in the feed."""
    contents = [clean_text(c.get("value", "")) for c in entry.get("content", [])]
    return max(contents, key=len, default="")


def _read_feed(source: str, url: str, cutoff: datetime) -> List[dict]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning("Feed %s failed: %s", source, e)
        return []

    feed = feedparser.parse(response.content)
    articles = []
    for entry in feed.entries[:PER_FEED_LIMIT]:
        link = (entry.get("link") or "").strip()
        title = clean_text(entry.get("title", ""))
        published = _entry_datetime(entry)
        if not link or not title:
            continue
        if published and published < cutoff:
            continue
        articles.append({
            "title": title,
            "link": link,
            "source": source,
            "published": published.strftime("%a, %d %b %Y %H:%M UTC") if published else "",
            "published_ts": int(published.timestamp()) if published else 0,
            "description": clean_text(entry.get("summary", "")),
            "feed_content": _longest_feed_content(entry),
        })
    logger.info("Feed %s: %d recent article(s)", source, len(articles))
    return articles


def fetch_feed_articles(
    max_articles: int = 25,
    max_age_days: int = FEED_MAX_AGE_DAYS,
    feeds: Optional[dict] = None,
) -> List[dict]:
    """
    List recent articles from the publisher feeds, newest first.
    Does NOT download article bodies; call add_full_text() for that.
    """
    feeds = feeds or AI_FEEDS
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    with ThreadPoolExecutor(max_workers=len(feeds)) as pool:
        results = pool.map(lambda item: _read_feed(item[0], item[1], cutoff), feeds.items())

    seen_links, seen_titles = set(), set()
    articles = []
    for article in (a for feed_articles in results for a in feed_articles):
        if article["link"] in seen_links or article["title"].lower() in seen_titles:
            continue
        seen_links.add(article["link"])
        seen_titles.add(article["title"].lower())
        articles.append(article)

    articles.sort(key=lambda a: a["published_ts"], reverse=True)
    return articles[:max_articles]


def _extract_text(url: str) -> Optional[str]:
    """Download a page and extract its main article text. None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.debug("Download failed for %s: %s", url, e)
        return None
    return trafilatura.extract(
        response.text,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )


def _with_full_text(article: dict) -> dict:
    """
    Add "content" (the best text available) and "full_text" (True if it is the
    real article body). Order of preference:
      1. text extracted from the article page
      2. full content included in the feed
      3. the feed's short summary (full_text = False)
    """
    article = dict(article)
    for candidate in (_extract_text(article["link"]), article.get("feed_content", "")):
        if candidate and len(candidate.split()) >= MIN_FULL_TEXT_WORDS:
            article["content"] = candidate.strip()
            article["full_text"] = True
            return article

    logger.info("Using summary only (full text unavailable): %s", article["title"])
    article["content"] = article.get("description", "")
    article["full_text"] = False
    return article


def add_full_text(articles: List[dict]) -> List[dict]:
    """Download and extract full text for each article, in parallel."""
    if not articles:
        return []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        enriched = list(pool.map(_with_full_text, articles))
    logger.info("Full text extracted for %d of %d article(s)",
                sum(a["full_text"] for a in enriched), len(enriched))
    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# 2. Google News headlines (email digest agent)
# ═══════════════════════════════════════════════════════════════════════════

GOOGLE_NEWS_URL = "https://news.google.com/rss/search"
SEARCH_QUERY = "artificial intelligence"
DEFAULT_MAX_ARTICLES = 10
DEFAULT_WINDOW = "1d"   # Google News time filter: e.g. "1h", "12h", "1d", "7d"


def fetch_ai_articles(
    max_articles: int = DEFAULT_MAX_ARTICLES,
    window: str = DEFAULT_WINDOW,
) -> List[dict]:
    """
    Fetch the latest AI headlines from Google News RSS (title, link, date and
    a very short description; no article body).
    """
    params = {"q": f"{SEARCH_QUERY} when:{window}", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        response = requests.get(GOOGLE_NEWS_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Google News fetch failed: %s", e)
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        logger.error("Failed to parse Google News RSS: %s", e)
        return []

    articles: List[dict] = []
    seen_titles: set = set()

    for item in root.findall(".//item"):
        title_el, link_el, date_el, desc_el = (item.find(t) for t in ("title", "link", "pubDate", "description"))

        if any(el is None or not (el.text or "").strip() for el in (title_el, link_el, date_el)):
            continue

        title = clean_text(title_el.text)
        if title in seen_titles:
            continue
        seen_titles.add(title)

        articles.append({
            "title":       title,
            "link":        link_el.text.strip(),
            "published":   date_el.text.strip(),
            "description": clean_text(desc_el.text) if desc_el is not None else "",
        })
        if len(articles) >= max_articles:
            break

    logger.info("Google News: %d headline(s) from the last %s", len(articles), window)
    return articles
