"""
api.py — FastAPI REST endpoints for the RAG system.

Run with:
    uvicorn api:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.rag_service import ask_news_rag, get_stats, ingest_latest_news, reset_knowledge_base

app = FastAPI(
    title="AI News Agent RAG API",
    description="RAG-powered AI News Agent using ChromaDB and Groq",
    version="1.2.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=["What are today's AI headlines?"])


@app.get("/")
def home():
    return {"message": "AI News Agent RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return get_stats()


@app.post("/ingest-news")
def ingest_news():
    return ingest_latest_news()


@app.delete("/knowledge-base")
def clear_knowledge_base():
    reset_knowledge_base()
    return {"status": "cleared"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank")

    result = ask_news_rag(question)
    if not result["ok"]:
        # 502 = an upstream service (Groq / the vector DB) failed
        raise HTTPException(status_code=502, detail=result["answer"])
    return result
