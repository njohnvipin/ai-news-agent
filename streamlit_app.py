"""
streamlit_app.py — Chat UI for the AI News Agent RAG system.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st

from services.rag_service import ask_news_rag, get_stats, ingest_latest_news, reset_knowledge_base

st.set_page_config(page_title="AI News Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI News Agent")
st.caption("Chat with the latest AI news · ChromaDB · SentenceTransformers · Groq")

# Conversation memory: the chat lives in session_state, which survives reruns
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role", "content", + for assistant: sources, warnings, standalone, ok}
if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None

# ── Sidebar: knowledge base ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Knowledge Base")

    stats = get_stats()
    st.metric("Total chunks stored", stats["total_chunks"])

    per_source = stats.get("articles_per_source") or {}
    if per_source:
        st.caption("Articles per publisher (source balance)")
        for name, count in per_source.items():
            st.markdown(f"- {name}: **{count}**")

    if st.button("🔄 Ingest Latest News"):
        with st.spinner("Downloading and embedding articles… (first run can take a minute)"):
            result = ingest_latest_news()
        if result["status"] == "success":
            st.session_state.ingest_message = (
                "success",
                f"✅ Done! {result['articles_added']} new article(s) "
                f"({result['articles_full_text']} with full text), "
                f"+{result['chunks_added']} chunk(s). "
                f"{result['articles_skipped_duplicates']} already stored.",
            )
        else:
            st.session_state.ingest_message = ("error", f"❌ {result['message']}")
        st.rerun()

    if st.session_state.ingest_message:
        kind, text = st.session_state.ingest_message
        (st.success if kind == "success" else st.error)(text)

    st.divider()
    if st.button("🧹 New chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("⚙️ Maintenance"):
        st.caption("Deletes every stored article. Use this after changing chunk settings or the embedding model.")
        confirm = st.checkbox("Yes, delete everything")
        if st.button("🗑️ Clear knowledge base", disabled=not confirm):
            reset_knowledge_base()
            st.session_state.ingest_message = ("success", "🗑️ Knowledge base cleared.")
            st.session_state.messages = []
            st.rerun()

    st.caption(f"Collection `{stats['collection_name']}` · `{stats['chroma_path']}`")


# ── Chat ─────────────────────────────────────────────────────────────────────
def show_assistant_extras(msg: dict) -> None:
    if not msg.get("ok", True):
        return
    if msg.get("standalone") and msg["standalone"] != msg.get("question"):
        st.caption(f"🔎 Searched for: *{msg['standalone']}*")
    for warning in msg.get("warnings", []):
        st.warning(warning, icon="⚠️")
    if msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for src in msg["sources"]:
                details = " · ".join(x for x in (
                    src.get("source"),
                    src.get("published"),
                    f"relevance {src['similarity']:.2f}",
                    None if src.get("full_text") else "summary only",
                ) if x)
                st.markdown(f"**[{src['number']}] {src['title']}**  \n🔗 [{src['link']}]({src['link']})  \n🕒 {details}")
        st.caption("AI-generated from the sources above and may contain errors.")


if not st.session_state.messages:
    st.info("Ask about recent AI news. Follow-up questions work too, e.g. *“tell me more about the second one”*.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and not msg.get("ok", True):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_assistant_extras(msg)

question = st.chat_input("Ask about recent AI news…")
if question:
    if stats["total_chunks"] == 0:
        st.warning("The knowledge base is empty. Click **Ingest Latest News** first.")
    else:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Searching the news and writing an answer…"):
            response = ask_news_rag(question, history=history)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "ok": response["ok"],
            "question": question,
            "standalone": response.get("standalone_question"),
            "sources": response.get("sources", []),
            "warnings": response.get("warnings", []),
        })
        st.rerun()
