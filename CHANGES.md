# Changes

## v1.3.1 — Agent kept searching and never drafted

**Symptom:** `python app.py` ended with `Sorry, need more steps to process this request.` and "The agent did not produce a draft". Nothing was sent.

**Cause:** the knowledge base comes from publisher feeds while the headlines come from Google News, so searches often returned loosely related articles. The model kept re-searching the same topics, hit Groq's rate limit (429, slow retries) and ran out of steps (`RECURSION_LIMIT`) before calling `save_draft`.

| File | Fix |
|---|---|
| `agents/tools.py` | At most 3 searches per run, **enforced in code** (the tool answers "Search limit reached… call save_draft"). Agent searches only return articles with similarity ≥ 0.35, and say clearly when nothing matches. Smaller results (2 articles, 500 characters) mean fewer tokens and fewer rate-limit pauses. |
| `agents/news_agent.py` | Step-by-step prompt: one search per chosen headline, never the same topic twice, results may be related stories rather than the exact headline. |
| `config.py` | Hides the per-request INFO logs from httpx, Hugging Face and sentence-transformers; their warnings still show. |
| `tests/test_tools.py` | New test for the search cap (49 tests). |

Lesson: a prompt asks, code enforces. Limits that matter belong in the tool.

---

## v1.3 — Agents, memory and responsible AI

### What changed, file by file

| File | Change | Concept |
|---|---|---|
| `agents/news_agent.py` | Rewritten: 4 tools, a looser prompt so the agent decides what to research, `MemorySaver` checkpointer with a `thread_id`, `agent.stream()` printing each Reason → Act → Observe step, `revise()` that reuses the same thread. Recursion limit 20. | Agent anatomy, ReAct, short-term memory |
| `agents/tools.py` | New tools `search_knowledge_base` (uses the RAG retrieval) and `get_previous_digests`; `send_email_tool` replaced by `save_draft` — the agent can no longer send email. Email body carries an AI-generated label. | Tool calling, human-in-the-loop, disclosure |
| `agents/memory.py` | **New.** Saves sent digests to `data/digests.json` and reads recent ones. | Long-term memory |
| `app.py` | Shows the draft and asks **[s]end / [r]evise / [c]ancel**; only a person can send. `--dry-run` never sends. Saves sent digests to long-term memory. Email settings only checked when sending. | Human-in-the-loop |
| `services/rag_service.py` | `condense_question()` rewrites follow-ups using chat history; `ask_news_rag(question, history)` returns `standalone_question` and `warnings`; sources wrapped in `<source id="n">` tags; citation and injection checks on every answer. `ingest_articles()` split out of `ingest_latest_news()`; every function takes an optional collection name (used by the evaluation). `group_by_article()` is now public. `get_stats()` adds articles per publisher. | Memory, prompt-injection defense, output checks |
| `utils/text_utils.py` | New `find_injection_phrases()`, `neutralize_source_tags()`, `extract_citations()`, `check_citations()`. | Responsible AI |
| `utils/chroma_utils.py` | Functions take a collection name; new `get_source_counts()`; `delete_collection(recreate=...)`; `distance_to_similarity()` takes the collection instead of looking it up. | Evaluation, source diversity |
| `services/news_service.py` | Added The Guardian's AI feed (non-US view); `MAX_PER_SOURCE = 8` stops one publisher from dominating an ingest. | Source diversity |
| `streamlit_app.py` | Rewritten as a chat (`st.chat_input`, `st.chat_message`) with history in `session_state`; shows "Searched for" when a question was rewritten, warnings, AI-generated label, per-publisher counts, **New chat** button. | Memory, transparency |
| `api.py` | `X-API-Key` required for `/ingest-news` and `/knowledge-base` (503 if no `API_KEY` set, 401 if wrong); `/ask` accepts `history`. Version 1.3.0. | Security |
| `config.py` | New `API_KEY`, `DATA_DIR`, `ai_disclosure()`. | |
| `eval/fixtures.json`, `eval/run_eval.py` | **New.** Synthetic test set and a script reporting hit rate, MRR, off-topic filtering, and (with `--with-llm`) citation validity and honest refusals. | Evaluation |
| `tests/` | New `conftest.py`, `test_tools.py`, `test_memory.py`, `test_api.py`, and injection/citation tests in `test_text_utils.py`. 48 tests total. | |
| `requirements.txt` | Added `httpx` (for API tests). | |
| `.env.example` | Added `API_KEY`. | |
| `.gitignore` | Added `data/`. | |
| `README.md` | New "Concepts in the code" section; chat, agent, API and evaluation docs updated. | |

### Behaviour changes to know about

- **Agent mode never sends without you.** `python app.py` now waits for **s**/**r**/**c**. For scheduled runs use `--dry-run`.
- **The API's ingest and delete endpoints need `API_KEY`** in `.env` and the `X-API-Key` header. `/ask`, `/stats` and `/health` are unchanged.
- **Follow-up questions cost one extra, small LLM call** (the rewrite). First questions in a chat don't.
- **Existing `chroma_db/` data keeps working.** No re-ingest needed.

---

## v1.2.1 — ChromaDB fix for Windows

- `requirements.txt`: pinned `chromadb==0.6.3` and `posthog<6`. ChromaDB 1.x (the new Rust engine) can fail on Windows with `Error creating hnsw segment reader: Nothing found on disk`, especially when the database is reopened by a new process. `posthog<6` stops harmless but noisy telemetry errors from chromadb 0.6.3.
- `utils/chroma_utils.py`: ChromaDB telemetry turned off (`anonymized_telemetry=False`).
- `services/rag_service.py`: if searching the database fails, the error message explains how to recover (delete `chroma_db/`, restart, re-ingest).

**After updating:** stop the app, delete the `chroma_db` folder, run `pip install -r requirements.txt`, start the app and ingest again. A `chroma_db` folder created by ChromaDB 1.x can't be read by 0.6.3.

---

## v1.2 — Full article text for RAG

| Area | What changed | Why |
|---|---|---|
| `services/news_service.py` | New `fetch_feed_articles()` reads AI feeds from TechCrunch, The Verge, MIT Technology Review and Ars Technica; new `add_full_text()` downloads each article and extracts the body with `trafilatura`, falling back to feed content, then the summary. | Google News RSS only provides headlines, so the RAG had almost nothing to search and chunking never did anything. |
| `services/rag_service.py` | Ingest checks for duplicates *before* downloading; stores `source`, `full_text` and `published_ts` metadata. Retrieval groups chunks by article (up to 5 articles × 3 chunks). New `reset_knowledge_base()`. | Avoids re-downloading stored articles; multiple relevant passages from one article now reach the LLM. |
| Chunking | 150 words with 30 overlap (was 300/50); each chunk starts with `Title: … (Source)`. | `all-MiniLM-L6-v2` only reads the first 256 tokens (~190 words); text past that was invisible to search. The header keeps each chunk tied to its article. |
| `streamlit_app.py` / `api.py` | Ingest shows how many articles got full text; sources show publisher and a "summary only" tag; Clear knowledge base button and `DELETE /knowledge-base`. | Visibility and an easy way to start fresh. |
| Tests | `tests/test_news_service.py` (feeds, dedup, failing feeds, full-text fallbacks; no network needed). 19 tests total. | |
| `requirements.txt` | Added `feedparser`, `trafilatura`. | |

The email digest agent still uses Google News headlines, which suits a headline digest.

### Smaller v1.2 changes

- `utils/text_utils.py`: `build_article_text()` was replaced by `build_article_header()` and `build_article_documents()`; new `strip_header()` removes the chunk header before text is shown to the LLM. `chunk_text()` defaults changed to 150/30.
- `services/news_service.py`: the Google News fetch (still used by the email agent) moved to the bottom of the file, unchanged in behaviour.
- `services/rag_service.py`: retrieves 12 chunks per question (was 8); the prompt tells the model the sources are excerpts.
- Response shapes: ingest results now include `articles_full_text`; each source in `/ask` results includes `source` (publisher) and `full_text`.

**After updating:** run `pip install -r requirements.txt`, then in the app open **⚙️ Maintenance → Clear knowledge base** and ingest again, so old headline-only chunks (made with the old chunk size) are replaced.

---

## v1.1 — Review fixes

### Bug fixes

| File | What changed | Why |
|---|---|---|
| `config.py` | Only `GROQ_API_KEY` is required at startup; email settings are checked only when sending. `MODEL_NAME` defaults to `openai/gpt-oss-120b`. `load_dotenv(override=True)`. | Streamlit and the API refused to start without Gmail settings, while a missing `MODEL_NAME` caused a confusing crash later. `override=True` stops old system environment variables from beating `.env`. |
| `agents/tools.py` | `send_email_tool` checks the result of `send_email()`. | It always reported "Email sent successfully", even when sending failed. |
| `services/rag_service.py` | LLM and retrieval calls wrapped in `try/except`; results include `ok: True/False`. | Any Groq error (like the retired-model 404) crashed the UI with a raw traceback. |
| `utils/chroma_utils.py` | `CHROMA_PATH` anchored to the project folder. | Starting the app from another directory silently created a new, empty database. |
| `services/email_service.py` | Recipients sent as BCC; date added to the subject; SMTP timeout. | Every recipient could see everyone else's address. |

### RAG quality

- Ingest fetches up to **25 articles from the last day** (was 5 from the last hour, which often returned nothing). *(Superseded in v1.2: ingest now uses publisher feeds.)*
- Retrieval drops chunks below `MIN_SIMILARITY` (0.2), so unrelated articles aren't sent to the LLM.
- Sources are numbered and the model cites them as `[1]`, `[2]`; today's date is in the prompt so "today" questions work.
- Prompt split into system + user messages, and tells the model to ignore instructions inside articles.
- The LLM is created on first use instead of at import time.

### Robustness and code quality

- `chunk_text` raises `ValueError` when `overlap >= chunk_size` (previously an infinite loop).
- Duplicate check is one batched lookup; chunks are stored with `upsert` (idempotent).
- Streamlit keeps the last answer across reruns (`st.session_state`), Enter submits the question, LLM errors show as a red error box, and sources show relevance scores.
- API: question must be 1–500 characters; failures return HTTP 502 instead of a 200 with an error; new `/stats` endpoint.
- `print` replaced with `logging` in services (set `LOG_LEVEL` in `.env`).
- `requirements.txt` has upper bounds so a fresh install can't jump to a breaking major version.
- Outdated docstring in `news_agent.py` corrected (LangChain 1.0 does have `create_agent`).

### Smaller v1.1 changes

- **Email digest agent:** fetches 10 headlines from the last day (was 5 from the last hour). The system prompt now says to stop whether the email succeeded or failed, and to reply with one line. `app.py` checks the email settings before starting and shows the error type on failure.
- **Imports:** `agents/tools.py` imports `tool` from `langchain_core.tools` (stable across LangChain versions) instead of `langchain.tools`.
- **Renames and removals:** in `embedding_service.py`, `MODEL_NAME` was renamed `EMBEDDING_MODEL` so it isn't confused with the Groq model in `config.py`. In `chroma_utils.py`, `document_exists()` and `collection_count()` were replaced by `existing_ids()`, and `distance_to_similarity()` was added.
- **Email:** the message is built with `MIMEText` instead of `MIMEMultipart` (plain text only, so the wrapper wasn't needed).
- **Text cleaning:** `clean_text()` replaces HTML tags with a space instead of nothing, so words on either side of a tag no longer get glued together.
- **Google News request:** built with `params=` instead of a hand-written URL, sends a User-Agent header, and skips items whose title, link or date is empty.
- **Response shapes:** `/ask` results include `ok`, and each source includes `number` and `similarity`. Ingest results include `articles_added`; a failed ingest returns `chunks_added: 0` instead of `count: 0`.
- **Streamlit:** removed `use_container_width` (deprecated in newer Streamlit versions); new placeholder question.
- **API:** version `1.1.0`, example question shown in the Swagger UI.
- **`.env.example`:** grouped into required, agent-only and optional settings, with comments.
- **`pytest.ini`:** added so `pytest` finds the `tests/` folder.

### Project hygiene

- Added `tests/test_text_utils.py` (run with `pytest`), `LICENSE` (MIT, copyright `njohnvipin`; change to your full name if you prefer), `.pytest_cache/` in `.gitignore`.
- README updated to match: Windows `copy` command, "on demand" instead of "every hour", email optional for UI/API, outdated troubleshooting entries removed.

### After updating

1. `pip install -r requirements.txt` (adds pytest).
2. Optionally add `LOG_LEVEL=INFO` to your `.env` (see `.env.example`).
3. Your existing `chroma_db/` keeps working, but if you started the app from a different folder before, your data may be in that folder's `chroma_db/`. Simplest: click **Ingest Latest News** again.
