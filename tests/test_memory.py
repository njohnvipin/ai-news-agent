"""Tests for the agent's long-term memory (agents/memory.py)."""
import pytest

from agents import memory


@pytest.fixture(autouse=True)
def temp_memory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DIGEST_FILE", tmp_path / "digests.json")


def test_empty_memory():
    assert memory.load_recent_digests() == []


def test_saved_digests_come_back_newest_first():
    memory.save_digest("first", ["h1"])
    memory.save_digest("second", ["h2"])
    memory.save_digest("third", ["h3"])
    assert [d["insights"] for d in memory.load_recent_digests(limit=2)] == ["third", "second"]


def test_only_the_newest_are_kept(monkeypatch):
    monkeypatch.setattr(memory, "MAX_STORED_DIGESTS", 2)
    for text in ("a", "b", "c"):
        memory.save_digest(text, [])
    assert [d["insights"] for d in memory.load_recent_digests(limit=10)] == ["c", "b"]


def test_corrupt_file_is_treated_as_empty():
    memory.DIGEST_FILE.write_text("{not json", encoding="utf-8")
    assert memory.load_recent_digests() == []
