"""
api.py — FastAPI REST endpoints for the RAG system.

Run with:
    uvicorn api:app --reload

Endpoints that change data (/ingest-news, /knowledge-base) require the
X-API-Key header to match API_KEY in .env.
"""
import secrets
from typing import List, Literal

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import API_KEY
from services.rag_service import ask_news_rag, get_stats, ingest_latest_news, reset_knowledge_base

app = FastAPI(
    title="AI News Agent RAG API",
    description="RAG-powered AI News Agent using ChromaDB and Groq",
    version="1.3.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(api_key_header)) -> None:
    """Protect endpoints that change data. Secure by default: no API_KEY, no access."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Set API_KEY in .env to enable this endpoint")
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=["What are today's AI headlines?"])
    history: List[ChatTurn] = Field(default_factory=list, max_length=20,
                                    description="Earlier turns, oldest first, for follow-up questions")


@app.get("/")
def home():
    return {"message": "AI News Agent RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_stats()


@app.post("/ingest-news", dependencies=[Depends(require_api_key)])
def ingest_news():
    return ingest_latest_news()


@app.delete("/knowledge-base", dependencies=[Depends(require_api_key)])
def clear_knowledge_base():
    reset_knowledge_base()
    return {"status": "cleared"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank")

    history = [turn.model_dump() for turn in request.history]
    result = ask_news_rag(question, history=history)
    if not result["ok"]:
        # 502 = an upstream service (Groq / the vector DB) failed
        raise HTTPException(status_code=502, detail=result["answer"])
    return result
