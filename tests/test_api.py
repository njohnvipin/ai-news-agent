"""Tests for the REST API's input validation and API-key protection."""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "ingest_latest_news", lambda: {"status": "success"})
    monkeypatch.setattr(api, "reset_knowledge_base", lambda: None)
    return TestClient(api.app)


def test_write_endpoints_are_disabled_without_a_configured_key(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", None)
    assert client.post("/ingest-news").status_code == 503


def test_write_endpoints_reject_a_wrong_or_missing_key(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    assert client.post("/ingest-news").status_code == 401
    assert client.delete("/knowledge-base", headers={"X-API-Key": "nope"}).status_code == 401


def test_write_endpoints_accept_the_right_key(client, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    assert client.post("/ingest-news", headers={"X-API-Key": "secret"}).json() == {"status": "success"}
    assert client.delete("/knowledge-base", headers={"X-API-Key": "secret"}).json() == {"status": "cleared"}


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 501}, {}])
def test_invalid_questions_are_rejected(client, body):
    assert client.post("/ask", json=body).status_code == 422


def test_history_is_passed_to_the_rag_pipeline(client, monkeypatch):
    seen = {}

    def fake_ask(question, history=None):
        seen["history"] = history
        return {"ok": True, "answer": "a [1]", "sources": [], "standalone_question": question, "warnings": []}

    monkeypatch.setattr(api, "ask_news_rag", fake_ask)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    response = client.post("/ask", json={"question": "more?", "history": history})
    assert response.status_code == 200
    assert seen["history"] == history
