import json
from pathlib import Path
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer,
    Float, Text, Boolean, DateTime, JSON
)
from sqlalchemy.orm import declarative_base, Session
from config import DATA_PROCESSED_DIR, INDEX_DIR

Base = declarative_base()


class ChunkRecord(Base):
    __tablename__ = "chunks"

    chunk_id    = Column(String, primary_key=True)
    doc_id      = Column(String, nullable=False, index=True)
    chunk_type  = Column(String, nullable=False, index=True)  # text/heading/table/figure
    content     = Column(Text,   nullable=False)
    page        = Column(Integer, index=True)
    bbox        = Column(JSON,   nullable=True)
    metadata_   = Column("metadata", JSON, nullable=True)
    embedded    = Column(Boolean, default=False)   # flipped True after embedding
    created_at  = Column(DateTime, default=datetime.utcnow)


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id      = Column(String, primary_key=True)
    filename    = Column(String, nullable=False)
    total_chunks= Column(Integer, default=0)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    indexed     = Column(Boolean, default=False)


class MetadataStore:
    def __init__(self, db_path: str = "data/docmind.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        print(f"📦 MetadataStore ready → {db_path}")

    def ingest_document(self, doc_id: str, filename: str, chunks_path: str):
        with open(chunks_path) as f:
            chunks = json.load(f)

        with Session(self.engine) as session:
            # Upsert document record
            doc = session.get(DocumentRecord, doc_id)
            if not doc:
                doc = DocumentRecord(
                    doc_id=doc_id,
                    filename=filename,
                    total_chunks=len(chunks)
                )
                session.add(doc)
            else:
                doc.total_chunks = len(chunks)

            # Delete old chunks for this doc (re-ingest)
            session.query(ChunkRecord).filter_by(doc_id=doc_id).delete()

            # Insert new chunks
            for c in chunks:
                session.add(ChunkRecord(
                    chunk_id   = c["chunk_id"],
                    doc_id     = c["doc_id"],
                    chunk_type = c["chunk_type"],
                    content    = c["content"],
                    page       = c["page"],
                    bbox       = c.get("bbox"),
                    metadata_  = c.get("metadata", {})
                ))

            session.commit()

        print(f"✅ Ingested {len(chunks)} chunks for '{doc_id}' into DB")

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_chunks(self, doc_id: str = None, chunk_type: str = None, page: int = None) -> list[dict]:
        with Session(self.engine) as session:
            q = session.query(ChunkRecord)
            if doc_id:      q = q.filter_by(doc_id=doc_id)
            if chunk_type:  q = q.filter_by(chunk_type=chunk_type)
            if page is not None: q = q.filter_by(page=page)
            return [self._to_dict(r) for r in q.all()]

    def get_all_documents(self) -> list[dict]:
        with Session(self.engine) as session:
            return [
                {"doc_id": d.doc_id, "filename": d.filename,
                 "total_chunks": d.total_chunks, "indexed": d.indexed}
                for d in session.query(DocumentRecord).all()
            ]

    def get_unembedded_chunks(self, doc_id: str = None) -> list[dict]:
        with Session(self.engine) as session:
            q = session.query(ChunkRecord).filter_by(embedded=False)
            if doc_id:
                q = q.filter_by(doc_id=doc_id)
            return [self._to_dict(r) for r in q.all()]

    def mark_embedded(self, chunk_ids: list[str]):
        with Session(self.engine) as session:
            session.query(ChunkRecord)\
                .filter(ChunkRecord.chunk_id.in_(chunk_ids))\
                .update({"embedded": True}, synchronize_session=False)
            session.commit()

    def stats(self, doc_id: str = None) -> dict:
        with Session(self.engine) as session:
            q = session.query(ChunkRecord)
            if doc_id:
                q = q.filter_by(doc_id=doc_id)
            all_chunks = q.all()
            breakdown = {}
            for c in all_chunks:
                breakdown[c.chunk_type] = breakdown.get(c.chunk_type, 0) + 1
            return {"total": len(all_chunks), "by_type": breakdown}

    def _to_dict(self, r: ChunkRecord) -> dict:
        return {
            "chunk_id":   r.chunk_id,
            "doc_id":     r.doc_id,
            "chunk_type": r.chunk_type,
            "content":    r.content,
            "page":       r.page,
            "bbox":       r.bbox,
            "metadata":   r.metadata_,
            "embedded":   r.embedded
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m docmind.ingestion.metadata_store <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1]
    merged_path = Path(DATA_PROCESSED_DIR) / doc_id / "chunks_merged.json"

    if not merged_path.exists():
        print(f"❌ No merged chunks found at {merged_path}")
        print(f"   Run chunk_merger first: python -m docmind.ingestion.chunk_merger {doc_id}")
        sys.exit(1)

    store = MetadataStore()
    store.ingest_document(doc_id, f"{doc_id}.pdf", str(merged_path))

    print(f"\n📊 DB Stats for '{doc_id}':")
    stats = store.stats(doc_id)
    print(f"   Total chunks : {stats['total']}")
    for t, count in stats["by_type"].items():
        print(f"   {t:10s}: {count}")

    print(f"\n📄 Sample text chunk:")
    samples = store.get_chunks(doc_id=doc_id, chunk_type="text")
    if samples:
        print(f"   {samples[0]['content'][:200]}...")