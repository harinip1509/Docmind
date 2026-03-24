# DocMind
An offline document intelligence engine built on retrieval-augmented generation.
No API keys. No cloud. Runs entirely on your machine.

---

## Overview
DocMind lets you ingest PDF documents and query them in plain English.
It extracts text, tables, and figures, embeds them locally, indexes them in FAISS,
and generates grounded answers using a local LLM — all without an internet connection.

The goal was to build the RAG stack from scratch rather than wrapping LangChain,
so every layer — chunking, retrieval, reranking, generation — is explicit and controllable.

---

## Features
- Multi-format ingestion — PDF, DOCX, PPTX
- Hybrid retrieval — BM25 + FAISS fused with Reciprocal Rank Fusion
- Fully offline — Mistral and nomic-embed-text run locally via Ollama
- Visual document understanding — tables and figures extracted and reasoned over
- Citation engine — every answer traces back to a source page and chunk
- Streamlit UI — upload, ingest, and chat in the browser

---

## Tech Stack
| Layer | Technology |
|-------|-----------|
| Document Parsing | PyMuPDF, pdfplumber |
| Embeddings | nomic-embed-text via Ollama |
| Vector Index | FAISS |
| Sparse Retrieval | BM25 |
| Local LLM | Mistral via Ollama |
| Metadata Store | SQLite + SQLAlchemy |
| UI | Streamlit |

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/harinip1509/Docmind
cd docmind
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install Ollama and pull models
```bash
brew install ollama
ollama serve &
ollama pull mistral
ollama pull nomic-embed-text
ollama pull llava
```

### 4. Run the UI
```bash
streamlit run docmind/ui/app.py
```

---

## How it works
```
PDF -> Extract -> Merge -> Embed -> Index -> Retrieve -> Generate
       PyMuPDF   chunks   768-dim  FAISS   BM25+RRF   Mistral
```

1. **Ingest** — PyMuPDF extracts text, tables, and figures page by page
2. **Chunk** — raw lines are merged into semantic paragraphs with configurable overlap
3. **Embed** — nomic-embed-text converts every chunk into a 768-dimensional vector
4. **Index** — vectors are stored in FAISS, metadata persisted in SQLite
5. **Retrieve** — BM25 and dense search results are fused using Reciprocal Rank Fusion
6. **Generate** — Mistral produces a cited answer grounded in the retrieved chunks

---

## Project Structure
```
docmind/
├── docmind/
│   ├── ingestion/      # document parsers and chunk merger
│   ├── embeddings/     # local embedding pipeline
│   ├── index/          # FAISS vector index
│   ├── retrieval/      # hybrid retriever
│   ├── generation/     # LLM generation and prompt templates
│   ├── reasoning/      # cross-document reasoning (in progress)
│   └── ui/             # Streamlit app
├── data/
│   ├── raw/            # place input PDFs here
│   ├── processed/      # extracted and merged chunks
│   └── indexes/        # saved FAISS indexes
├── config.py
└── requirements.txt
```

---

## Roadmap
- [x] PDF ingestion and chunking
- [x] Local embeddings with nomic-embed-text
- [x] Hybrid retrieval with BM25 + FAISS + RRF
- [x] Local LLM generation with Mistral
- [x] Streamlit UI with citations
- [ ] Multi-document cross-document reasoning
- [ ] LLaVA vision model for chart and figure understanding
- [ ] FastAPI REST backend
- [ ] Docker setup

---

## License
MIT