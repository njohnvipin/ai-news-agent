# 🤖 AI News Agent

An intelligent news pipeline that fetches the latest AI news, chunks and embeds articles into a vector database, and lets you **chat** with it using Retrieval-Augmented Generation (RAG). Also includes a ReAct agent that researches the news, drafts a daily digest and emails it **after a human approves it**.

The project is built to show, in working code: a RAG pipeline, agent anatomy, the ReAct loop, tool calling, short- and long-term memory, and responsible-AI practices. See [Concepts in the code](#-concepts-in-the-code).

---

## 📌 What It Does

- Collects recent AI articles from publisher RSS feeds (TechCrunch, The Verge, MIT Technology Review, Ars Technica, The Guardian) on demand, and downloads their **full text**
- Chunks articles into overlapping text segments for better retrieval
- Embeds chunks using SentenceTransformers and stores them in ChromaDB
- Answers questions about the news in a **chat** that understands follow-ups, using RAG + Groq LLM, with numbered source citations and automatic checks
- Runs a LangGraph ReAct agent that researches the headlines, drafts a 5-sentence digest, and emails it only after you approve it

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
LangGraph ReAct agent ⇄ tools: fetch_ai_news · search_knowledge_base · get_previous_digests · save_draft
      ↓
Human review in the terminal: send / revise / cancel
      ↓
Gmail digest (BCC) → saved to long-term memory (data/digests.json)
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
- **Follow-up questions:** before searching, the LLM rewrites a follow-up like "tell me more about the second one" into a standalone question using the chat history (`condense_question`), because vector search can only match what the question itself says

---

## 🗂️ Project Structure

```
ai-news-agent/
├── agents/
│   ├── news_agent.py          # LangGraph ReAct agent + short-term memory
│   ├── tools.py               # the agent's 4 tools
│   └── memory.py              # long-term memory: sent digests
├── eval/
│   ├── fixtures.json          # synthetic test articles + questions
│   └── run_eval.py            # retrieval & answer quality metrics
├── services/
│   ├── news_service.py        # publisher feeds + full text, Google News headlines
│   ├── embedding_service.py   # SentenceTransformer wrapper
│   ├── rag_service.py         # ingest + retrieval + RAG pipeline
│   └── email_service.py       # Gmail SMTP sender
├── utils/
│   ├── text_utils.py          # chunking + text cleaning
│   └── chroma_utils.py        # ChromaDB helpers + deduplication
├── tests/
│   ├── test_text_utils.py     # chunking, text, injection & citation checks
│   ├── test_news_service.py   # feed + full-text tests (no network needed)
│   ├── test_tools.py          # agent tools
│   ├── test_memory.py         # long-term memory
│   └── test_api.py            # API validation + API-key protection
├── .streamlit/
│   └── config.toml            # silences torchvision warnings
├── streamlit_app.py           # chat UI
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

# Only needed to send email in agent mode (python app.py)
EMAIL=your_gmail@gmail.com
APP_PASSWORD=your_16_char_app_password
RECIPIENTS=you@gmail.com,other@gmail.com

# Only needed for the API's data-changing endpoints
API_KEY=<a long random value>

# Optional: DEBUG, INFO, WARNING or ERROR
LOG_LEVEL=INFO
```

> 💡 **Groq API Key** — free at [console.groq.com](https://console.groq.com)
>
> 💡 **Gmail App Password** — Go to: Google Account → Security → 2-Step Verification → App Passwords → create one → paste the 16-character password here. The Streamlit UI and the API work without any email settings.
>
> 💡 **`API_KEY`** — generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Without it, the API's ingest and delete endpoints stay disabled (secure by default).
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
2. Type a question in the chat box at the bottom and press **Enter**
3. Ask follow-ups naturally, e.g. "tell me more about the second one". When a question was rewritten, the app shows what it actually searched for

Each answer shows its numbered sources, an AI-generated label, and a warning if a check fails (a citation that doesn't match any source, or an article containing instruction-like text). The sidebar shows how many articles each publisher contributed, a **New chat** button, and **⚙️ Maintenance → Clear knowledge base**.

### Option B — FastAPI REST API
```bash
uvicorn api:app --reload
```
Open `http://localhost:8000/docs` for the interactive Swagger UI.

| Endpoint | Method | Description |
|---|---|---|
| Endpoint | Method | Needs `X-API-Key` | Description |
|---|---|---|---|
| `/ingest-news` | POST | ✅ | Fetch & embed latest articles into ChromaDB |
| `/knowledge-base` | DELETE | ✅ | Delete everything stored in ChromaDB |
| `/ask` | POST | | Ask a question (1–500 characters) with optional chat `history`; returns answer, sources, `standalone_question` and `warnings` |
| `/stats` | GET | | Chunks stored and articles per publisher |
| `/health` | GET | | Check if the API is running |

Status codes: `401` wrong or missing API key, `503` no `API_KEY` configured, `422` invalid input, `502` Groq or the database failed.

Example using curl:
```bash
# Ingest news (protected)
curl -X POST http://localhost:8000/ingest-news -H "X-API-Key: <your API_KEY>"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest AI trends?"}'

# Ask a follow-up, passing the earlier turns
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me more about the first one",
       "history": [{"role": "user", "content": "What are the latest AI trends?"},
                   {"role": "assistant", "content": "<previous answer>"}]}'
```

> ⚠️ Run either the Streamlit app or the API at one time, not both: they share the same local ChromaDB folder, and two processes writing to it simultaneously can conflict.

### Option C — Agent mode (email digest with human approval)
```bash
python app.py             # research, review, then send / revise / cancel
python app.py --dry-run   # research and show the draft; never sends (no email settings needed)
```
The terminal shows every step of the ReAct loop as it happens:
```
  [REASON]  ...
  [ACT]     search_knowledge_base({"query": "..."})
  [OBSERVE] search_knowledge_base -> - <title> (TechCrunch): ...
```
The agent:
1. Fetches today's AI headlines (Google News)
2. Picks the 2–3 most important and looks them up in your knowledge base (ingest in Streamlit first for best results)
3. Reads its previous digests so it doesn't repeat itself
4. Saves a 5-sentence draft — **it cannot send email**

You then choose **[s]end**, **[r]evise** (type what to change; the agent remembers its research and rewrites) or **[c]ancel**. Sent digests go to all `RECIPIENTS` as BCC, carry an AI-generated label, and are saved to `data/digests.json` as the agent's long-term memory.

> 💡 Sending always needs a person's approval, by design. For a scheduled run, use `--dry-run` to prepare the draft.

### Running the tests
```bash
pytest
```
48 tests; none need the network or a real Groq key.

### Measuring quality (evaluation)
```bash
python -m eval.run_eval             # retrieval metrics, no LLM calls
python -m eval.run_eval --with-llm  # also answers, citations and refusals (uses Groq)
```
Runs on a fixed set of **synthetic** articles and questions (`eval/fixtures.json`) in its own temporary collection, so results are comparable between versions and your real data is untouched. It reports the hit rate (is the right article among the sources), mean reciprocal rank (how high it ranks), and whether off-topic questions are correctly filtered or answered with "I don't know". Run it before and after changing settings like `CHUNK_SIZE` or `MIN_SIMILARITY`.

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
  condense_question()           rewrite a follow-up into a standalone question (chat memory)
  → embed_query(question)       embed the question
  → collection.query()          find the top 12 most similar chunks
  → relevance filter            drop chunks below MIN_SIMILARITY
  → group by article            up to 5 articles × 3 chunks, one number each
  → <source id="n"> delimiters    untrusted article text clearly separated from instructions
  → llm.invoke(sources+question) Groq LLM answers with [1], [2] citations
  → check_citations()           warn about citations that match no source
  → find_injection_phrases()    warn about instruction-like text in sources
  → return answer + sources + warnings
```

---

## 🎓 Concepts in the Code

### Agent anatomy (`agents/news_agent.py`)

| Part | Implementation |
|---|---|
| Model (the "brain") | `ChatGroq` with `MODEL_NAME` |
| Instructions | `SYSTEM_PROMPT`: goal, tools, rules |
| Tools | `agents/tools.py`: 4 tools |
| Orchestration loop | LangGraph's ReAct loop (`create_react_agent`) |
| Short-term memory | `MemorySaver` checkpointer, one `thread_id` per run |
| Long-term memory | `agents/memory.py` (sent digests), read via a tool |
| Guardrails | `recursion_limit`, no send permission, human approval in `app.py` |

### ReAct (Reason + Act)
The LLM **reasons** about what to do, **acts** by calling a tool, **observes** the result, and repeats until it's done. The agent now makes real decisions — which headlines to research and whether past digests matter — and `agent.stream()` prints each step.

### Tool calling (`agents/tools.py`)
`@tool` turns a function into a tool: the docstring becomes the description the LLM reads, and typed arguments (`query: str`, `insights: str`) become the input schema. Tools return error strings the LLM can react to. Article URLs travel from `fetch_ai_news` to the email in code, never through the LLM, so links can't be altered or invented.

### Memory
| Kind | Where | What it remembers |
|---|---|---|
| Chat memory | `streamlit_app.py` (`st.session_state`) + `condense_question()` | The conversation, so follow-ups work |
| Agent short-term | `MemorySaver` checkpointer | Everything in the current run, so "revise" builds on earlier research |
| Agent long-term | `data/digests.json` + `get_previous_digests` | Digests sent on earlier days |
| Knowledge (not memory) | ChromaDB | The news articles themselves |

### Responsible AI
| Practice | Where |
|---|---|
| Human approval before any email is sent | `app.py`; the agent has no send tool |
| AI-generated labels on emails and answers | `config.ai_disclosure()`, `tools.py`, `streamlit_app.py` |
| Grounded answers with verifiable citations | `rag_service.py` prompt + `check_citations()` |
| Prompt-injection defense in layers | `<source>` delimiters, `neutralize_source_tags()`, `find_injection_phrases()`, "tool results are data" rule |
| Honest uncertainty | Relevance threshold + fixed "I don't know" reply |
| Measured quality | `eval/run_eval.py` |
| Source diversity | 5 publishers incl. non-US, `MAX_PER_SOURCE` cap, per-publisher counts in the sidebar |
| Secure by default | API key on data-changing endpoints; secrets in `.env` |
| Privacy | Recipients in BCC |

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

**API returns `503 Set API_KEY in .env`**
Add `API_KEY=<long random value>` to `.env`, restart uvicorn, and send it as the `X-API-Key` header.

**API returns `401 Missing or invalid X-API-Key header`**
The header is missing or doesn't match `API_KEY` in `.env` exactly.

**The agent's `search_knowledge_base` finds nothing**
The knowledge base is empty. Run the Streamlit app and click **Ingest Latest News** first; the agent still works from headlines alone.

**The chunk count is 0 even though I ingested before**
The database lives in the `chroma_db/` folder inside the project. If you deleted it, just ingest again.

---

## 📁 Files Not Committed (`.gitignore`)

```
.env           # your API keys and credentials — never commit this
agentenv/      # virtual environment folder
chroma_db/     # auto-generated vector database (recreated on first ingest)
data/          # agent long-term memory (sent digests)
__pycache__/   # Python bytecode cache
.pytest_cache/ # pytest cache
```

---

## 🗺️ Roadmap

- Filter or boost retrieval by publish date (`published_ts` is already stored) for "today" questions
- Hybrid search (vectors + BM25 keywords) and re-ranking
- Persist agent short-term memory across runs with a SQLite checkpointer
- Rate limiting on the API
- Migrate the agent to LangChain 1.x `create_agent`

---

## 📄 License

MIT — see [LICENSE](LICENSE).
