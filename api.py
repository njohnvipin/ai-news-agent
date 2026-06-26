from fastapi import FastAPI
from pydantic import BaseModel

from services.rag_service import ingest_latest_news, ask_news_rag


app = FastAPI(
    title="AI News Agent RAG API",
    description="RAG-powered AI News Agent using ChromaDB and Groq",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI News Agent RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/ingest-news")
def ingest_news():
    result = ingest_latest_news()
    return result


@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = ask_news_rag(request.question)
    return result