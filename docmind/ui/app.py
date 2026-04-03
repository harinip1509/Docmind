import streamlit as st
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from docmind.ingestion.pdf_extractor import PDFExtractor
from docmind.ingestion.chunk_merger import ChunkMerger
from docmind.ingestion.metadata_store import MetadataStore
from docmind.embeddings.embedder import Embedder
from docmind.index.faiss_index import FaissIndex
from docmind.engine import DocMind
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = None
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def ingest_pdf(uploaded_file) -> str:
    """Full ingestion pipeline: PDF → chunks → embeddings → FAISS."""
    raw_dir = Path(DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    pdf_path = raw_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc_id = pdf_path.stem

    with st.status(f"⚙️ Processing {uploaded_file.name}...", expanded=True) as status:

        st.write("📄 Extracting text, tables and figures...")
        extractor = PDFExtractor(str(pdf_path))
        chunks = extractor.extract()
        extractor.save()

        st.write("🔗 Merging chunks into paragraphs...")
        merger = ChunkMerger(doc_id)
        merged = merger.merge()
        merger.save(merged)

        st.write("🗄️ Loading into metadata store...")
        store = MetadataStore()
        store.ingest_document(
            doc_id, uploaded_file.name,
            str(Path(DATA_PROCESSED_DIR) / doc_id / "chunks_merged.json")
        )

        st.write("🔢 Embedding chunks locally (nomic-embed-text)...")
        embedder = Embedder()
        embeddings, embeddable = embedder.embed_chunks(merged)

        out_dir = Path(DATA_PROCESSED_DIR) / doc_id
        import numpy as np
        np.save(out_dir / "embeddings.npy", embeddings)
        with open(out_dir / "embeddable_chunks.json", "w") as f:
            json.dump(embeddable, f, indent=2)

        st.write("🗂️ Building FAISS index...")
        idx = FaissIndex(doc_id)
        idx.build(embeddings, embeddable)
        idx.save()

        status.update(label="✅ Document ready!", state="complete")

    return doc_id


def load_existing_docs() -> list[str]:
    """Find all already-indexed documents."""
    index_dir = Path("data/indexes")
    if not index_dir.exists():
        return []
    return [p.name for p in index_dir.iterdir() if (p / "index.faiss").exists()]


def render_chunk_card(chunk: dict, rank: int):
    """Render a retrieved chunk as a styled card."""
    type_icons = {"text": "📝", "heading": "📌", "table": "📊", "figure": "🖼️"}
    icon = type_icons.get(chunk["chunk_type"], "📄")
    score = chunk.get("rrf_score", chunk.get("score", 0))

    with st.expander(f"{icon} [{rank}] Page {chunk['page']} · {chunk['chunk_type']} · score={score:.4f}"):
        st.markdown(chunk["content"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/354/brain_1f9e0.png", width=60)
    st.title("DocMind")
    st.caption("Offline Document Intelligence Engine")
    st.divider()

    # Upload new doc
    st.subheader("📂 Load Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded and st.button("⚙️ Ingest Document", use_container_width=True):
        with st.spinner("Processing..."):
            doc_id = ingest_pdf(uploaded)
            st.session_state.doc_id = doc_id
            st.session_state.engine = DocMind(doc_id)
            st.session_state.chat_history = []
            st.session_state.chunks = []
        st.success(f"✅ Ready: {doc_id}")
        st.rerun()

    st.divider()

    # Load existing doc
    existing = load_existing_docs()
    if existing:
        st.subheader("📁 Existing Documents")
        selected = st.selectbox("Switch document", ["— select —"] + existing)
        if selected != "— select —" and st.button("Load", use_container_width=True):
            st.session_state.doc_id = selected
            st.session_state.engine = DocMind(selected)
            st.session_state.chat_history = []
            st.session_state.chunks = []
            st.rerun()

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    mode = st.selectbox("Output mode", ["qa", "summarize", "compare"])
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)
    show_chunks = st.toggle("Show retrieved chunks", value=True)


# ── Main area ─────────────────────────────────────────────────────────────────
if st.session_state.engine is None:
    # Landing screen
    st.markdown("""
    # 🧠 DocMind
    ### Offline Multi-Document Intelligence Engine

    > 100% local · No API keys · Powered by Mistral + nomic-embed-text via Ollama

    ---

    **Get started:**
    1. Upload a PDF in the sidebar
    2. Wait for ingestion to complete
    3. Ask anything about your document

    **Output modes:**
    - `qa` — Direct question answering with citations
    - `summarize` — Structured summary with key points
    - `compare` — Cross-section comparison and contrast
    """)

    st.info("👈 Upload a PDF in the sidebar to get started", icon="🧠")

else:
    doc_id = st.session_state.doc_id
    engine = st.session_state.engine

    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 🧠 Chatting with `{doc_id}`")
    with col2:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.session_state.chunks = []
            st.rerun()

    st.divider()

    # Two column layout — chat | chunks
    chat_col, chunks_col = st.columns([3, 2]) if show_chunks else (st, None)

    with chat_col:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Query input
        query = st.chat_input("Ask anything about your document...")

        if query:
            # Show user message
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Retrieve + generate
            with st.chat_message("assistant"):
                with st.spinner("🔍 Retrieving + generating..."):
                    chunks = engine.retriever.hybrid_search(query, top_k=top_k)
                    result = engine.generator.generate(query, chunks, mode=mode)

                answer = result["answer"]
                st.markdown(answer)

                # Citations
                with st.expander("📚 Citations"):
                    for i, c in enumerate(result["citations"]):
                        st.markdown(f"**[{i+1}]** Page {c['page']} ({c['type']}): _{c['excerpt']}..._")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.chunks = chunks
            st.rerun()

    # Retrieved chunks panel
    if show_chunks and chunks_col and st.session_state.chunks:
        with chunks_col:
            st.markdown("#### 📡 Retrieved Chunks")
            st.caption(f"Top {len(st.session_state.chunks)} chunks via hybrid search (BM25 + FAISS + RRF)")
            for i, chunk in enumerate(st.session_state.chunks):
                render_chunk_card(chunk, i + 1)