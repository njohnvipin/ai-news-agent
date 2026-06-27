# 🤖 AI News Agent

An intelligent news pipeline that fetches the latest AI news, chunks and embeds articles into a vector database, and enables semantic Q&A using Retrieval-Augmented Generation (RAG). Also includes an autonomous agent that writes and emails a daily AI news digest.

---

## 📌 What It Does

- Fetches live AI news from Google News RSS every hour
- Chunks articles into overlapping text segments for better retrieval
- Embeds chunks using SentenceTransformers and stores them in ChromaDB
- Answers natural language questions about the news using RAG + Groq LLM
- Runs an autonomous LangGraph agent that emails a 5-sentence AI news digest

---

## 🏗️ Architecture

```
Google News RSS
      ↓
news_service.py       — fetches & cleans raw articles
      ↓
text_utils.py         — chunks text (300 words, 50-word overlap)
      ↓
embedding_service.py  — embeds chunks via SentenceTransformers (all-MiniLM-L6-v2)
      ↓
ChromaDB              — stores vectors + metadata persistently
      ↓
rag_service.py        — embeds query → similarity search → Groq LLM → answer
      ↓
Streamlit / FastAPI   — user interface & REST API
```

```
Agent Mode (separate):
LangGraph ReAct Agent → fetch_ai_news tool → send_email_tool → Gmail digest
```

---

## 🧠 Chunking & Embedding — How It Works

### Chunking (`utils/text_utils.py`)

Long article text is split into overlapping word-level chunks before being stored. This ensures:
- No single chunk is too large for the embedding model
- Context is not lost at chunk boundaries due to overlap

```
chunk_size = 300 words
overlap    = 50 words
```

Example:
```
Full article text:  800 words
→ Chunk 0: words 0–299
→ Chunk 1: words 250–549   (50-word overlap with Chunk 0)
→ Chunk 2: words 500–799   (50-word overlap with Chunk 1)
```

Each article is assembled from its fields before chunking:
```
Title: <title>
Description: <description>
Published: <date>
Link: <url>
```

### Embedding (`services/embedding_service.py`)

Model: **`all-MiniLM-L6-v2`** from SentenceTransformers

- Lightweight (~80MB) but highly capable for semantic similarity
- Embeddings are **L2-normalized** (`normalize_embeddings=True`) for accurate cosine similarity
- Chunks are embedded in **batches of 32** for efficiency
- Model is loaded once as a **singleton** — no redundant reloads

```python
embed_query(text)         # single question → vector
embed_chunks(chunks)      # list of chunks → list of vectors (batched)
```

### Deduplication (`utils/chroma_utils.py`)

Each article gets a **deterministic ID** derived from its URL:
```python
article_id = md5(article["link"])
chunk_id   = f"{article_id}_chunk_{index}"
```
Re-running ingest will skip articles already stored — no duplicates ever created.

---

## 🗂️ Project Structure

```
ai-news-agent/
├── agents/
│   ├── news_agent.py          # LangGraph ReAct agent
│   └── tools.py               # fetch_ai_news, send_email_tool
├── services/
│   ├── news_service.py        # Google News RSS fetcher
│   ├── embedding_service.py   # SentenceTransformer wrapper
│   ├── rag_service.py         # ingest + retrieval + RAG pipeline
│   └── email_service.py       # Gmail SMTP sender
├── utils/
│   ├── text_utils.py          # chunking + text cleaning
│   └── chroma_utils.py        # ChromaDB helpers + deduplication
├── .streamlit/
│   └── config.toml            # silences torchvision warnings
├── streamlit_app.py           # browser UI
├── api.py                     # FastAPI REST endpoints
├── app.py                     # agent mode entry point
├── config.py                  # env var loader
├── requirements.txt
├── README.md
├── .env.example               # safe to commit — no real secrets
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| Agent Framework | LangGraph (ReAct agent) |
| LLM Orchestration | LangChain + LangChain-Groq |
| News Source | Google News RSS |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Email | Gmail SMTP |
| Language | Python 3.11 |

---

## 🚀 Getting Started

### Step 1 — Clone the repository
```bash
git clone https://github.com/njohnvipin/ai-news-agent.git
cd ai-news-agent
```

### Step 2 — Switch to the feature branch
```bash
git fetch origin
git checkout feature/rag-embedding-chunking
```

### Step 3 — Create a virtual environment
```bash
python -m venv agentenv
```

Activate it:
```bash
# Mac/Linux
source agentenv/bin/activate

# Windows
agentenv\Scripts\activate
```

You should see `(agentenv)` at the start of your terminal line.

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ `sentence-transformers` will download the embedding model (~80MB) on first run — this is normal, just wait for it.

### Step 5 — Set up your environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your values:
```
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=llama-3.3-70b-versatile
EMAIL=your_gmail@gmail.com
APP_PASSWORD=your_16_char_app_password
RECIPIENTS=you@gmail.com,other@gmail.com
```

> 💡 **Groq API Key** — free at [console.groq.com](https://console.groq.com)
>
> 💡 **Gmail App Password** — Go to: Google Account → Security → 2-Step Verification → App Passwords → create one → paste the 16-character password here

### Step 6 — Fix Streamlit warnings (optional but recommended)
Create a `.streamlit/` folder in the project root and add `config.toml` inside it:
```toml
[server]
fileWatcherType = "none"
```
This silences harmless `torchvision` warnings from the `transformers` library.

---

## 🖥️ Running the App

### Option A — Streamlit UI (recommended for beginners)
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

1. Click **Ingest Latest News** in the sidebar to fetch and store articles
2. Type a question in the text box
3. Click **Ask** to get an AI-generated answer with sources

### Option B — FastAPI REST API
```bash
uvicorn api:app --reload
```
Open `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/ingest-news` | POST | Fetch & embed latest articles into ChromaDB |
| `/ask` | POST | Ask a question, get an answer + sources |
| `/health` | GET | Check if the API is running |

Example using curl:
```bash
# Ingest news
curl -X POST http://localhost:8000/ingest-news

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest AI trends?"}'
```

### Option C — Agent mode (email digest)
```bash
python app.py
```
The LangGraph agent will:
1. Fetch today's AI news headlines
2. Write a 5-sentence analysis
3. Email it to all addresses in `RECIPIENTS`

---

## 🔄 RAG Pipeline (Step by Step)

```
Ingest:
  fetch_ai_articles()          fetch up to 5 articles from Google News RSS
  → build_article_text()       combine title + description + date + link
  → chunk_text(300, overlap=50) split into overlapping word chunks
  → embed_chunks()             batch embed via SentenceTransformers
  → document_exists()          check for duplicates before storing
  → collection.add()           store chunks + embeddings in ChromaDB

Query:
  embed_query(question)        embed the user's question
  → collection.query()         find top 5 most similar chunks
  → deduplicate by link        remove repeated sources from context
  → llm.invoke(context+question) Groq LLM generates a grounded answer
  → return answer + sources
```

---

## ❓ Troubleshooting

**`ModuleNotFoundError: No module named 'torchvision'`**
These are warnings from Streamlit's file watcher, not real errors. Create `.streamlit/config.toml` as shown in Step 6 above.

**`TypeError: unsupported operand type(s) for |`**
You're on Python 3.11. In `utils/chroma_utils.py` and `services/embedding_service.py`, change `X | None` to `Optional[X]` and add `from typing import Optional` at the top.

**Answer says "I don't know"**
The knowledge base may be empty. Click **Ingest Latest News** first, then ask headline-level questions like:
- "What are today's AI headlines?"
- "Is there any news about AI regulation?"
- "Summarize the AI news today"

**Agent keeps sending emails in a loop**
Add `config={"recursion_limit": 10}` to the `agent.invoke()` call in `agents/news_agent.py`.

---

## 📁 Files Not Committed (`.gitignore`)

```
.env          # your API keys and credentials — never commit this
agentenv/     # virtual environment folder
chroma_db/    # auto-generated vector database (recreated on first ingest)
__pycache__/  # Python bytecode cache
```

---

## 📄 License

MIT
