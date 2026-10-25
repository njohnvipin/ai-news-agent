"""Tests for services/news_service.py, with the network replaced by fake responses."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from services import news_service


def _rss(items):
    body = "".join(
        f"<item><title>{t}</title><link>{l}</link><pubDate>{format_datetime(d)}</pubDate>"
        f"<description>{desc}</description></item>"
        for t, l, d, desc in items
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{body}</channel></rss>'


class FakeResponse:
    def __init__(self, text):
        self.text = text
        self.content = text.encode()

    def raise_for_status(self):
        pass


@pytest.fixture
def fake_web(monkeypatch):
    """Map URL -> response text. Unknown URLs raise like a network error."""
    pages = {}

    def fake_get(url, **kwargs):
        if url not in pages:
            raise news_service.requests.exceptions.ConnectionError(url)
        return FakeResponse(pages[url])

    monkeypatch.setattr(news_service.requests, "get", fake_get)
    return pages


NOW = datetime.now(timezone.utc)


def test_feed_articles_are_recent_deduplicated_and_sorted(fake_web):
    fake_web["http://feed-a"] = _rss([
        ("Older story", "http://a/1", NOW - timedelta(hours=5), "s"),
        ("Too old", "http://a/2", NOW - timedelta(days=10), "s"),
    ])
    fake_web["http://feed-b"] = _rss([
        ("Newest story", "http://b/1", NOW - timedelta(hours=1), "s"),
        ("older story", "http://b/2", NOW - timedelta(hours=2), "s"),  # same title, other site
    ])
    articles = news_service.fetch_feed_articles(feeds={"A": "http://feed-a", "B": "http://feed-b"})

    assert [a["title"] for a in articles] == ["Newest story", "Older story"]
    assert articles[0]["source"] == "B"


def test_a_failing_feed_does_not_break_the_others(fake_web):
    fake_web["http://ok"] = _rss([("Works", "http://ok/1", NOW, "s")])
    articles = news_service.fetch_feed_articles(feeds={"Down": "http://down", "Up": "http://ok"})
    assert [a["title"] for a in articles] == ["Works"]


def test_full_text_is_extracted_from_the_page(fake_web, monkeypatch):
    long_body = " ".join(["word"] * 300)
    monkeypatch.setattr(news_service.trafilatura, "extract", lambda html, **kw: long_body)
    fake_web["http://page"] = "<html>...</html>"

    [article] = news_service.add_full_text([{"title": "T", "link": "http://page", "description": "summary"}])
    assert article["full_text"] is True
    assert article["content"] == long_body


def test_falls_back_to_feed_content_then_summary(fake_web):
    long_feed_body = " ".join(["feed"] * 300)
    with_feed = {"title": "A", "link": "http://gone/1", "description": "sum", "feed_content": long_feed_body}
    summary_only = {"title": "B", "link": "http://gone/2", "description": "just a summary"}

    a, b = news_service.add_full_text([with_feed, summary_only])
    assert a["full_text"] is True and a["content"] == long_feed_body
    assert b["full_text"] is False and b["content"] == "just a summary"
