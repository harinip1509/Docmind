import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from docmind.embeddings.embedder import Embedder
from docmind.index.faiss_index import FaissIndex
from config import TOP_K


class Retriever:
    """
    Hybrid retriever — dense (FAISS) + sparse (BM25)
    fused with Reciprocal Rank Fusion.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.embedder = Embedder()

        # Load FAISS index
        self.faiss_index = FaissIndex(doc_id)
        self.faiss_index.load()

        # Build BM25 on same chunks
        self.chunks = self.faiss_index.chunks
        tokenized = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ Retriever ready — {len(self.chunks)} chunks indexed")

    def dense_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        query_vec = self.embedder.embed_text(query)
        return self.faiss_index.search(query_vec, top_k=top_k)

    def sparse_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results

    def hybrid_search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Reciprocal Rank Fusion of dense + sparse results."""
        dense_results  = self.dense_search(query, top_k=top_k)
        sparse_results = self.sparse_search(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (60 + rank + 1)
            chunk_map[cid] = chunk

        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (60 + rank + 1)
            chunk_map[cid] = chunk

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in ranked[:top_k]:
            chunk = chunk_map[cid].copy()
            chunk["rrf_score"] = round(score, 6)
            results.append(chunk)

        return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: python -m docmind.retrieval.retriever <doc_id> "<query>"')
        sys.exit(1)

    doc_id, query = sys.argv[1], sys.argv[2]
    retriever = Retriever(doc_id)
    results = retriever.hybrid_search(query)

    print(f"\n🔍 Hybrid search: '{query}'")
    for i, r in enumerate(results):
        print(f"\n[{i+1}] rrf={r['rrf_score']} | page={r['page']} | type={r['chunk_type']}")
        print(f"     {r['content'][:200]}...")