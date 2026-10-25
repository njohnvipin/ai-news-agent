"""
streamlit_app.py — Browser UI for the AI News Agent RAG system.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st

from services.rag_service import ask_news_rag, get_stats, ingest_latest_news, reset_knowledge_base

st.set_page_config(page_title="AI News Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI News Agent — RAG Dashboard")
st.caption("Powered by ChromaDB · SentenceTransformers · Groq")

# Survives reruns, so clicking Ingest doesn't wipe the last answer
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "ingest_message" not in st.session_state:
    st.session_state.ingest_message = None

# ── Sidebar: DB stats + ingest ───────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Knowledge Base")

    stats = get_stats()
    st.metric("Total chunks stored", stats["total_chunks"])
    st.caption(f"Collection: `{stats['collection_name']}`")
    st.caption(f"Path: `{stats['chroma_path']}`")

    st.divider()

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
        st.rerun()  # refresh the chunk count

    if st.session_state.ingest_message:
        kind, text = st.session_state.ingest_message
        (st.success if kind == "success" else st.error)(text)

    with st.expander("⚙️ Maintenance"):
        st.caption("Deletes every stored article. Use this after changing chunk settings or the embedding model.")
        confirm = st.checkbox("Yes, delete everything")
        if st.button("🗑️ Clear knowledge base", disabled=not confirm):
            reset_knowledge_base()
            st.session_state.ingest_message = ("success", "🗑️ Knowledge base cleared.")
            st.session_state.last_response = None
            st.rerun()

# ── Main: Q&A ────────────────────────────────────────────────────────────────
st.subheader("💬 Ask a question about today's AI news")

# A form lets users press Enter to submit
with st.form("ask_form"):
    question = st.text_input(
        "Your question",
        placeholder="e.g. What are today's AI headlines?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please type a question first.")
    elif stats["total_chunks"] == 0:
        st.warning("The knowledge base is empty. Click **Ingest Latest News** first.")
    else:
        with st.spinner("Retrieving and generating answer…"):
            st.session_state.last_response = ask_news_rag(question.strip())

response = st.session_state.last_response
if response:
    if not response["ok"]:
        st.error(response["answer"])
    else:
        st.markdown("### Answer")
        st.write(response["answer"])

        if response["sources"]:
            st.markdown("### Sources")
            for src in response["sources"]:
                details = " · ".join(x for x in (
                    src.get("source"),
                    src.get("published"),
                    f"relevance {src['similarity']:.2f}",
                    None if src.get("full_text") else "summary only",
                ) if x)
                st.markdown(
                    f"**[{src['number']}] {src['title']}**  \n"
                    f"🔗 [{src['link']}]({src['link']})  \n"
                    f"🕒 {details}"
                )
