# 🤖 AI News Agent

An intelligent news pipeline that fetches the latest AI news, chunks and embeds articles into a vector database, and enables semantic Q&A using Retrieval-Augmented Generation (RAG). Also includes an autonomous agent that writes and emails a daily AI news digest.

---

## 📌 What It Does

- Collects recent AI articles from publisher RSS feeds (TechCrunch, The Verge, MIT Technology Review, Ars Technica) on demand, and downloads their **full text**
- Chunks articles into overlapping text segments for better retrieval
- Embeds chunks using SentenceTransformers and stores them in ChromaDB
- Answers natural language questions about the news using RAG + Groq LLM, with numbered source citations
- Runs an autonomous LangGraph agent that emails a 5-sentence AI news digest

---

## 🏗️ Architecture

```
Publisher RSS feeds
      ↓
news_service.py       — lists recent articles, downloads full text (trafilatura)
      ↓
text_utils.py         — chunks text (150 words, 30-word overlap) + title header
      ↓
embedding_service.py  — embeds chunks via SentenceTransformers (all-MiniLM-L6-v2)
      ↓
ChromaDB              — stores vectors + metadata persistently
      ↓
rag_service.py        — embeds query → similarity search → relevance filter → group by article → Groq LLM → cited answer
      ↓
Streamlit / FastAPI   — user interface & REST API
```

```
Agent Mode (separate):
LangGraph ReAct Agent → fetch_ai_news tool (Google News headlines) → send_email_tool → Gmail digest (BCC)
```

---

## 📰 Full Article Text — How It Works

### Sources (`services/news_service.py`)

Ingestion reads these AI-specific RSS feeds, which link directly to the articles:

| Source | Feed |
|---|---|
| TechCrunch | `https://techcrunch.com/category/artificial-intelligence/feed/` |
| The Verge | `https://www.theverge.com/rss/ai-artificial-intelligence/index.xml` |
| MIT Technology Review | `https://www.technologyreview.com/topic/artificial-intelligence/feed` |
| Ars Technica | `https://arstechnica.com/ai/feed/` |

To add or remove a source, edit the `AI_FEEDS` dictionary at the top of `news_service.py`. Any RSS or Atom feed works.

### Steps

1. **List** recent items from all feeds in parallel (items older than 3 days are ignored), drop duplicate links and titles, and keep the 25 newest
2. **Skip** articles already in ChromaDB, *before* downloading anything
3. **Download** each new article page in parallel and extract the main text with [`trafilatura`](https://trafilatura.readthedocs.io/), which strips menus, ads, footers and comments
4. **Fall back** when extraction fails or is too short (paywalls, blocked pages): use the full content embedded in the feed if there is one, otherwise the feed's short summary. These articles are marked **summary only** in the Sources list

A feed that is down or slow doesn't stop the others, and a failed download never fails the whole ingest.

> ⚖️ Article text is stored only in your local `chroma_db/` folder for your own Q&A. Respect each publisher's terms of use and don't republish their articles.

---

## 🧠 Chunking & Embedding — How It Works

### Chunking (`utils/text_utils.py`)

Article text is split into overlapping word-level chunks before being stored. This ensures:
- No chunk is longer than the embedding model can read
- Context is not lost at chunk boundaries due to overlap

```
chunk_size = 150 words
overlap    = 30 words
```

**Why 150 words?** `all-MiniLM-L6-v2` only reads the first **256 tokens** (about 190 words) of any text and silently ignores the rest. Bigger chunks would be stored in full, but their embedding would only reflect the beginning, so later content could never be found.

Example:
```
Article body:  390 words
→ Chunk 0: words 0–149
→ Chunk 1: words 120–269   (30-word overlap with Chunk 0)
→ Chunk 2: words 240–389   (30-word overlap with Chunk 1)
```

Every chunk starts with a short header naming its article, so each piece of text carries its context into the embedding:
```
Title: <title> (<source>)

<chunk text>
```
The link, date and source are stored as metadata rather than embedded.

### Embedding (`services/embedding_service.py`)

Model: **`all-MiniLM-L6-v2`** from SentenceTransformers

- Lightweight (~80MB) but highly capable for semantic similarity
- Embeddings are **L2-normalized** (`normalize_embeddings=True`), so similarity scores are cosine similarity
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
Before storing, ingest checks all incoming IDs in **one batched lookup** and skips articles already stored. Chunks are written with `upsert`, so re-running ingest never creates duplicates of the same URL.

### Retrieval (`services/rag_service.py`)

- The 12 most similar chunks are retrieved
- Chunks with a cosine similarity below `MIN_SIMILARITY` (default `0.2`) are dropped, so unrelated articles don't pollute the answer. Raise it to make answers stricter, lower it to let more context through
- Chunks are **grouped by article**: up to 5 articles, up to 3 chunks each, in reading order. Each article becomes one numbered source the LLM cites as `[1]`, `[2]`, …

---

## 🗂️ Project Structure

```
ai-news-agent/
├── agents/
│   ├── news_agent.py          # LangGraph ReAct agent
│   └── tools.py               # fetch_ai_news, send_email_tool
├── services/
│   ├── news_service.py        # publisher feeds + full text, Google News headlines
│   ├── embedding_service.py   # SentenceTransformer wrapper
│   ├── rag_service.py         # ingest + retrieval + RAG pipeline
│   └── email_service.py       # Gmail SMTP sender
├── utils/
│   ├── text_utils.py          # chunking + text cleaning
│   └── chroma_utils.py        # ChromaDB helpers + deduplication
├── tests/
│   ├── test_text_utils.py     # chunking & text tests
│   └── test_news_service.py   # feed + full-text tests (no network needed)
├── .streamlit/
│   └── config.toml            # silences torchvision warnings
├── streamlit_app.py           # browser UI
├── api.py                     # FastAPI REST endpoints
├── app.py                     # agent mode entry point
├── config.py                  # env var loader + logging setup
├── requirements.txt
├── pytest.ini
├── README.md
├── CHANGES.md                 # what changed and why
├── LICENSE
├── .env.example               # safe to commit — no real secrets
└── .gitignore
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| Agent Framework | LangGraph (ReAct agent) |
| LLM Orchestration | LangChain + LangChain-Groq |
| News Sources | Publisher RSS feeds (RAG), Google News RSS (email digest) |
| Article Extraction | trafilatura + feedparser |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Email | Gmail SMTP |
| Tests | pytest |
| Language | Python 3.11+ |

> 📝 **Why `openai/gpt-oss-120b`?** This project originally used `llama-3.3-70b-versatile`, but Groq deprecated that model and shut it down on **August 16, 2026**. Requests to it now fail with a `404 model_not_found` error. Groq's recommended replacements are `openai/gpt-oss-120b` (Production) and `qwen/qwen3.6-27b` (Preview). We chose `gpt-oss-120b` because Production models are more stable and get a migration path before removal, while Preview models can be pulled at short notice.
>
> Despite the name, this model runs entirely on Groq. **No OpenAI account or API key is needed**; your Groq API key is all you need.

---

## 🚀 Getting Started

### Prerequisites
- Python **3.11 or newer**
- A free Groq API key
- A Gmail account with an App Password (only for agent mode)

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
# Mac/Linux
cp .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Open `.env` and fill in your values:
```
# Required
GROQ_API_KEY=gsk_your_key_here
MODEL_NAME=openai/gpt-oss-120b

# Only needed for agent mode (python app.py)
EMAIL=your_gmail@gmail.com
APP_PASSWORD=your_16_char_app_password
RECIPIENTS=you@gmail.com,other@gmail.com

# Optional: DEBUG, INFO, WARNING or ERROR
LOG_LEVEL=INFO
```

> 💡 **Groq API Key** — free at [console.groq.com](https://console.groq.com)
>
> 💡 **Gmail App Password** — Go to: Google Account → Security → 2-Step Verification → App Passwords → create one → paste the 16-character password here. The Streamlit UI and the API work without any email settings.
>
> 💡 **`MODEL_NAME`** — keep this set to a model Groq currently supports (if left empty, `openai/gpt-oss-120b` is used). If Groq retires a model in the future, just update this one line in `.env` and restart the app; no code changes required. Check current models at [console.groq.com/docs/models](https://console.groq.com/docs/models) and retirements at [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations).

### About `.streamlit/config.toml`
This file is already included. It sets `fileWatcherType = "none"`, which silences harmless `torchvision` warnings from the `transformers` library. The trade-off is that Streamlit no longer auto-reloads when you edit code, so restart it after making changes.

---

## 🖥️ Running the App

### Option A — Streamlit UI (recommended for beginners)
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

1. Click **Ingest Latest News** in the sidebar to fetch and store articles (the first ingest downloads ~25 articles and can take a minute; later ingests only download new ones)
2. Type a question in the text box
3. Press **Enter** or click **Ask** to get an answer with numbered, cited sources

The sidebar's **⚙️ Maintenance** section has a **Clear knowledge base** button for starting fresh.

### Option B — FastAPI REST API
```bash
uvicorn api:app --reload
```
Open `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| `/ingest-news` | POST | Fetch & embed latest articles into ChromaDB |
| `/ask` | POST | Ask a question (1–500 characters), get a cited answer + sources |
| `/stats` | GET | Number of chunks stored in the knowledge base |
| `/knowledge-base` | DELETE | Delete everything stored in ChromaDB |
| `/health` | GET | Check if the API is running |

If the LLM or database fails, `/ask` returns HTTP `502` with the error in `detail`.

Example using curl:
```bash
# Ingest news
curl -X POST http://localhost:8000/ingest-news

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest AI trends?"}'
```

> ⚠️ Run either the Streamlit app or the API at one time, not both: they share the same local ChromaDB folder, and two processes writing to it simultaneously can conflict.

### Option C — Agent mode (email digest)
```bash
python app.py
```
The LangGraph agent will:
1. Fetch the latest AI news headlines (last 24 hours)
2. Write a 5-sentence analysis
3. Email it to all addresses in `RECIPIENTS` as BCC, so recipients can't see each other's addresses

If sending fails (e.g. a wrong App Password), the agent reports the failure instead of claiming success.

> 💡 **Want it daily?** Agent mode runs once per command. To send a digest automatically, schedule `python app.py` with Windows Task Scheduler or cron.

### Running the tests
```bash
pytest
```

---

## 🔄 RAG Pipeline (Step by Step)

```
Ingest:
  fetch_feed_articles(25)        list the 25 newest articles across all feeds
  → existing_ids()               one batched duplicate check (before downloading)
  → add_full_text()              download pages + extract text with trafilatura
  → build_article_documents()    chunk (150 words, 30 overlap) + title header
  → embed_chunks()               batch embed via SentenceTransformers
  → collection.upsert()          store chunks + embeddings + metadata in ChromaDB

Query:
  embed_query(question)         embed the user's question
  → collection.query()          find the top 12 most similar chunks
  → relevance filter            drop chunks below MIN_SIMILARITY
  → group by article            up to 5 articles × 3 chunks, one number each
  → llm.invoke(sources+question) Groq LLM answers with [1], [2] citations
  → return answer + sources
```

---

## ❓ Troubleshooting

**`groq.NotFoundError: Error code: 404 — The model 'llama-3.3-70b-versatile' does not exist`**
Groq retired this model on August 16, 2026. Fix:
1. Open `.env` and set `MODEL_NAME=openai/gpt-oss-120b`
2. Fully stop Streamlit (Ctrl+C) and run `streamlit run streamlit_app.py` again; `.env` is only read at startup

**`EnvironmentError: Missing GROQ_API_KEY in .env`**
Make sure `.env` exists in the project root (not just `.env.example`) and contains your key.

**Agent mode says email settings are missing / email fails to send**
Fill in `EMAIL`, `APP_PASSWORD` and `RECIPIENTS` in `.env`. `APP_PASSWORD` must be a Gmail App Password, not your normal Gmail password.

**Ingest says "No articles found"**
Check your internet connection. If only some feeds fail, the log shows `Feed <name> failed`; the others still work. A feed URL may have changed; update it in `AI_FEEDS`.

**Some sources say "summary only"**
That site blocked the download or the page couldn't be extracted (often paywalls), so only the feed summary was stored. Other articles are unaffected.

**Answers still look headline-only after updating**
Your database still holds the old headline-only chunks. Open **⚙️ Maintenance → Clear knowledge base** in the sidebar, then ingest again.

**Answer says "I don't know"**
The knowledge base may be empty or may not contain anything related. Click **Ingest Latest News** first. With full article text you can ask detailed questions, for example:
- "What did the latest AI regulation news actually say?"
- "Which companies announced new AI models this week, and what can they do?"
- "Summarize the main AI stories from the last few days"

**`chromadb.errors.InternalError: Error creating hnsw segment reader: Nothing found on disk`**
This comes from ChromaDB 1.x on Windows; the project pins ChromaDB 0.6.3 to avoid it. Stop the app, delete the `chroma_db` folder, run `pip install -r requirements.txt`, then start the app and ingest again.

**The chunk count is 0 even though I ingested before**
The database lives in the `chroma_db/` folder inside the project. If you deleted it, just ingest again.

---

## 📁 Files Not Committed (`.gitignore`)

```
.env           # your API keys and credentials — never commit this
agentenv/      # virtual environment folder
chroma_db/     # auto-generated vector database (recreated on first ingest)
__pycache__/   # Python bytecode cache
.pytest_cache/ # pytest cache
```

---

## 🗺️ Roadmap

- Filter or boost retrieval by publish date (`published_ts` is already stored) for "today" questions
- Migrate the agent to LangChain 1.x `create_agent`

---

## 📄 License

MIT — see [LICENSE](LICENSE).
