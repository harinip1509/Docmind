import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from config import DATA_PROCESSED_DIR, CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    chunk_type: str
    content: str
    page: int
    bbox: Optional[list]
    metadata: dict


class ChunkMerger:
    """
    Merges line-by-line text chunks into proper semantic paragraphs.
    Headings, tables and figures are kept as-is.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.chunks_path = Path(DATA_PROCESSED_DIR) / doc_id / "chunks.json"

    def load(self) -> list[dict]:
        with open(self.chunks_path) as f:
            return json.load(f)

    def merge(self) -> list[dict]:
        raw_chunks = self.load()
        merged = []

        # Separate text chunks from everything else
        text_chunks = [c for c in raw_chunks if c["chunk_type"] == "text"]
        other_chunks = [c for c in raw_chunks if c["chunk_type"] != "text"]

        # Group text chunks by page
        pages: dict[int, list[dict]] = {}
        for chunk in text_chunks:
            pages.setdefault(chunk["page"], []).append(chunk)

        # Merge text chunks within each page into paragraphs
        for page_num, page_chunks in sorted(pages.items()):
            buffer = ""
            buffer_start_idx = 0

            for i, chunk in enumerate(page_chunks):
                content = chunk["content"].strip()
                if not content:
                    continue

                if len(buffer) + len(content) < CHUNK_SIZE:
                    buffer += (" " if buffer else "") + content
                else:
                    if buffer:
                        merged.append(self._make_chunk(
                            buffer, page_num, buffer_start_idx, page_chunks
                        ))
                    words = buffer.split()
                    overlap_text = " ".join(words[-CHUNK_OVERLAP:]) if len(words) > CHUNK_OVERLAP else buffer
                    buffer = overlap_text + " " + content
                    buffer_start_idx = i

            if buffer.strip():
                merged.append(self._make_chunk(
                    buffer, page_num, buffer_start_idx, page_chunks
                ))

        # Add back non-text chunks as dicts
        for c in other_chunks:
            merged.append(c)

        # Sort by page
        merged.sort(key=lambda c: c["page"])

        print(f"✅ Merged {len(text_chunks)} raw text chunks → {len([c for c in merged if c['chunk_type'] == 'text'])} paragraph chunks")
        return merged

    def _make_chunk(self, content: str, page: int, idx: int, page_chunks: list) -> dict:
        chunk_id = f"{self.doc_id}_p{page}_merged_{idx}"
        return {
            "doc_id": self.doc_id,
            "chunk_id": chunk_id,
            "chunk_type": "text",
            "content": content.strip(),
            "page": page,
            "bbox": None,
            "metadata": {
                "source": page_chunks[0]["metadata"].get("source", ""),
                "merged": True
            }
        }

    def save(self, chunks: list[dict]) -> str:
        out_path = Path(DATA_PROCESSED_DIR) / self.doc_id / "chunks_merged.json"
        with open(out_path, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"💾 Saved merged chunks → {out_path}")
        return str(out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m docmind.ingestion.chunk_merger <doc_id>")
        sys.exit(1)

    merger = ChunkMerger(sys.argv[1])
    chunks = merger.merge()
    merger.save(chunks)

    print(f"\n📊 Chunk breakdown:")
    for chunk_type in ["text", "heading", "table", "figure"]:
        count = sum(1 for c in chunks if c["chunk_type"] == chunk_type)
        if count:
            print(f"   {chunk_type}: {count}")