import json
import numpy as np
import faiss
from pathlib import Path
from config import INDEX_DIR


class FaissIndex:
    """
    Wraps FAISS with a clean interface.
    Stores vectors + keeps chunk metadata aligned by position.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.index_dir = Path(INDEX_DIR) / doc_id
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index = None
        self.chunks: list[dict] = []   # aligned with index positions

    def build(self, embeddings: np.ndarray, chunks: list[dict]):
        """Build a flat L2 index from embeddings."""
        dim = embeddings.shape[1]

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Flat index — exact search, best for < 100k vectors
        self.index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalization
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"✅ FAISS index built — {self.index.ntotal} vectors, dim={dim}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search index, returns top_k chunks with scores."""
        if self.index is None:
            raise RuntimeError("Index not built or loaded yet.")

        # Normalize query
        query = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self):
        """Persist index + chunks to disk."""
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "chunks.json", "w") as f:
            json.dump(self.chunks, f, indent=2)
        print(f"💾 Index saved → {self.index_dir}/index.faiss")

    def load(self):
        """Load index + chunks from disk."""
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path) as f:
            self.chunks = json.load(f)

        print(f"✅ Index loaded — {self.index.ntotal} vectors")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from docmind.embeddings.embedder import Embedder
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m docmind.index.faiss_index <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1]
    processed_dir = Path("data/processed") / doc_id

    # Load saved embeddings + aligned chunks
    embeddings = np.load(processed_dir / "embeddings.npy")
    with open(processed_dir / "embeddable_chunks.json") as f:
        chunks = json.load(f)

    # Build + save index
    idx = FaissIndex(doc_id)
    idx.build(embeddings, chunks)
    idx.save()

    # Quick search test
    print(f"\n🔍 Quick search test...")
    embedder = Embedder()
    query_vec = embedder.embed_text("What is this document about?")
    results = idx.search(query_vec, top_k=3)

    print(f"\n📄 Top 3 results for 'What is this document about?'")
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] score={r['score']:.4f} | page={r['page']} | type={r['chunk_type']}")
        print(f"       {r['content'][:150]}...")