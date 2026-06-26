"""
streamlit_app.py — Browser UI for the AI News Agent RAG system.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st
from services.rag_service import ingest_latest_news, ask_news_rag, get_stats

st.set_page_config(page_title="AI News Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI News Agent — RAG Dashboard")
st.caption("Powered by ChromaDB · SentenceTransformers · Groq")

# ── Sidebar: DB stats + ingest ───────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Knowledge Base")

    stats = get_stats()
    st.metric("Total chunks stored", stats["total_chunks"])
    st.caption(f"Collection: `{stats['collection_name']}`")
    st.caption(f"Path: `{stats['chroma_path']}`")

    st.divider()

    if st.button("🔄 Ingest Latest News", use_container_width=True):
        with st.spinner("Fetching and embedding articles…"):
            result = ingest_latest_news()
        if result["status"] == "success":
            st.success(
                f"✅ Done!  +{result['chunks_added']} chunk(s) added.  "
                f"{result['articles_skipped_duplicates']} duplicate(s) skipped."
            )
            st.rerun()
        else:
            st.error(f"❌ {result['message']}")

# ── Main: Q&A ────────────────────────────────────────────────────────────────
st.subheader("💬 Ask a question about today's AI news")

question = st.text_input(
    "Your question",
    placeholder="e.g. What are the latest trends in large language models?",
)

if st.button("Ask", type="primary") and question.strip():
    if stats["total_chunks"] == 0:
        st.warning("The knowledge base is empty. Click **Ingest Latest News** first.")
    else:
        with st.spinner("Retrieving and generating answer…"):
            response = ask_news_rag(question)

        st.markdown("### Answer")
        st.write(response["answer"])

        if response["sources"]:
            st.markdown("### Sources")
            for src in response["sources"]:
                st.markdown(
                    f"- **{src['title']}**  \n"
                    f"  🔗 [{src['link']}]({src['link']})  \n"
                    f"  🕒 {src['published']}"
                )
