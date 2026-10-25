"""Tests for utils/text_utils.py — run with: pytest"""
import pytest

from utils.text_utils import (
    build_article_documents,
    build_article_header,
    chunk_text,
    clean_text,
    strip_header,
)


# ── clean_text ───────────────────────────────────────────────────────────────

def test_clean_text_strips_tags_and_entities():
    raw = '<a href="x">OpenAI &amp; friends</a>&nbsp;<font>Reuters</font>'
    assert clean_text(raw) == "OpenAI & friends Reuters"


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello \n\n  world\t ") == "hello world"


@pytest.mark.parametrize("value", ["", None])
def test_clean_text_empty(value):
    assert clean_text(value) == ""


# ── chunk_text ───────────────────────────────────────────────────────────────

def _words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_single_chunk():
    assert chunk_text(_words(10), chunk_size=300, overlap=50) == [_words(10)]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_text(_words(800), chunk_size=300, overlap=50)
    assert len(chunks) == 3
    first, second, third = (c.split() for c in chunks)
    assert first[0] == "w0" and first[-1] == "w299"
    assert second[0] == "w250" and second[-1] == "w549"   # 50-word overlap
    assert third[0] == "w500" and third[-1] == "w799"     # reaches the end


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 20), (10, -1), (0, 0)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text(_words(50), chunk_size=size, overlap=overlap)


# ── build_article_header / build_article_documents ───────────────────────────

def test_header_includes_source_when_present():
    assert build_article_header({"title": "Big news", "source": "TechCrunch"}) == "Title: Big news (TechCrunch)"
    assert build_article_header({"title": "Big news"}) == "Title: Big news"


def test_documents_chunk_the_body_and_prefix_the_header():
    article = {"title": "Big news", "source": "Ars", "content": _words(390)}
    docs = build_article_documents(article, chunk_size=150, overlap=30)
    assert len(docs) == 3
    assert all(d.startswith("Title: Big news (Ars)\n\n") for d in docs)
    assert strip_header(docs[0]).split()[0] == "w0"
    assert strip_header(docs[1]).split()[0] == "w120"


def test_documents_fall_back_to_description_then_header():
    assert build_article_documents({"title": "T", "description": "short summary"}, 150, 30) == [
        "Title: T\n\nshort summary"
    ]
    assert build_article_documents({"title": "T"}, 150, 30) == ["Title: T"]


def test_strip_header_leaves_plain_text_alone():
    assert strip_header("no header here\n\nsecond paragraph") == "no header here\n\nsecond paragraph"
