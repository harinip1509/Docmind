import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ollama

from config import EMBED_MODEL, DATA_PROCESSED_DIR


class Embedder:
    """
    Embeds chunks locally using nomic-embed-text via Ollama.
    No API keys. No internet. Everything runs on your machine.
    """

    def __init__(self):
        self.model = EMBED_MODEL
        self._verify_ollama()

    def _verify_ollama(self):
        try:
            ollama.list()
            print(f"✅ Ollama running — using model: {self.model}")
        except Exception:
            raise RuntimeError(
                "❌ Ollama is not running. Start it with: ollama serve"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string → numpy vector."""
        response = ollama.embeddings(model=self.model, prompt=text)
        return np.array(response["embedding"], dtype=np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of strings → 2D numpy array (n_chunks, dim)."""
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding"):
            batch = texts[i : i + batch_size]
            batch_embeddings = [
                ollama.embeddings(model=self.model, prompt=t)["embedding"]
                for t in batch
            ]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_chunks(self, chunks: list[dict]) -> tuple[np.ndarray, list[dict]]:
        """
        Takes a list of chunk dicts, returns:
        - embeddings matrix (n, dim)
        - same chunks list (for index alignment)
        Skips figure chunks that have no real text yet.
        """
        embeddable = [
            c for c in chunks
            if c["chunk_type"] != "figure" and len(c["content"].strip()) > 10
        ]
        skipped = len(chunks) - len(embeddable)

        if skipped:
            print(f"⚠️  Skipped {skipped} figure chunks (no text yet)")

        texts = [c["content"] for c in embeddable]
        print(f"🔢 Embedding {len(texts)} chunks with {self.model}...")
        embeddings = self.embed_batch(texts)

        print(f"✅ Embeddings shape: {embeddings.shape}")
        return embeddings, embeddable


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m docmind.embeddings.embedder <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1]
    chunks_path = Path(DATA_PROCESSED_DIR) / doc_id / "chunks_merged.json"

    with open(chunks_path) as f:
        chunks = json.load(f)

    embedder = Embedder()
    embeddings, embeddable_chunks = embedder.embed_chunks(chunks)

    # Save embeddings to disk
    out_dir = Path(DATA_PROCESSED_DIR) / doc_id
    np.save(out_dir / "embeddings.npy", embeddings)

    with open(out_dir / "embeddable_chunks.json", "w") as f:
        json.dump(embeddable_chunks, f, indent=2)

    print(f"💾 Saved embeddings → {out_dir}/embeddings.npy")
    print(f"💾 Saved aligned chunks → {out_dir}/embeddable_chunks.json")
    print(f"\n📐 Embedding dim : {embeddings.shape[1]}")
    print(f"📦 Total vectors : {embeddings.shape[0]}")