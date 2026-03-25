import fitz  # PyMuPDF
import pdfplumber
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

from config import DATA_PROCESSED_DIR


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    chunk_type: str          # "text" | "table" | "figure" | "heading"
    content: str             # text content or description
    page: int
    bbox: Optional[list]     # bounding box [x0, y0, x1, y1]
    metadata: dict


class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_id = Path(pdf_path).stem
        self.chunks: list[Chunk] = []

    def extract(self) -> list[Chunk]:
        print(f"\n📄 Extracting: {self.doc_id}")
        self._extract_text_and_headings()
        self._extract_tables()
        self._extract_figures()
        print(f"✅ Extracted {len(self.chunks)} chunks from {self.doc_id}")
        return self.chunks

    def _extract_text_and_headings(self):
        doc = fitz.open(self.pdf_path)

        for page_num, page in enumerate(tqdm(doc, desc="  Text/Headings")):
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] != 0:  # 0 = text block
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text or len(text) < 3:
                            continue

                        font_size = span["size"]
                        is_heading = font_size > 13 or span["flags"] & 2**4  # bold flag

                        chunk_type = "heading" if is_heading else "text"
                        chunk_id = f"{self.doc_id}_p{page_num}_{chunk_type}_{len(self.chunks)}"

                        self.chunks.append(Chunk(
                            doc_id=self.doc_id,
                            chunk_id=chunk_id,
                            chunk_type=chunk_type,
                            content=text,
                            page=page_num,
                            bbox=list(span["bbox"]),
                            metadata={
                                "font_size": font_size,
                                "font_name": span["font"],
                                "source": self.pdf_path
                            }
                        ))
        doc.close()

    def _extract_tables(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="  Tables")):
                tables = page.extract_tables()

                for t_idx, table in enumerate(tables):
                    if not table or not table[0]:
                        continue

                    # Convert table to markdown string
                    headers = table[0]
                    rows = table[1:]
                    md_table = " | ".join(str(h) for h in headers) + "\n"
                    md_table += " | ".join(["---"] * len(headers)) + "\n"
                    for row in rows:
                        md_table += " | ".join(str(c) for c in row) + "\n"

                    chunk_id = f"{self.doc_id}_p{page_num}_table_{t_idx}"
                    self.chunks.append(Chunk(
                        doc_id=self.doc_id,
                        chunk_id=chunk_id,
                        chunk_type="table",
                        content=md_table,
                        page=page_num,
                        bbox=None,
                        metadata={"source": self.pdf_path, "table_index": t_idx}
                    ))

    def _extract_figures(self):
        doc = fitz.open(self.pdf_path)
        figures_dir = Path(DATA_PROCESSED_DIR) / self.doc_id / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        for page_num, page in enumerate(tqdm(doc, desc="  Figures")):
            images = page.get_images(full=True)

            for img_idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]

                img_path = figures_dir / f"p{page_num}_fig{img_idx}.{img_ext}"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                chunk_id = f"{self.doc_id}_p{page_num}_figure_{img_idx}"
                self.chunks.append(Chunk(
                    doc_id=self.doc_id,
                    chunk_id=chunk_id,
                    chunk_type="figure",
                    content=f"[Figure extracted: {img_path}]",
                    page=page_num,
                    bbox=None,
                    metadata={
                        "source": self.pdf_path,
                        "image_path": str(img_path),
                        "described": False   # LLaVA will fill this in later
                    }
                ))
        doc.close()

    def save(self):
        out_dir = Path(DATA_PROCESSED_DIR) / self.doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "chunks.json"

        with open(out_path, "w") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)

        print(f"💾 Saved {len(self.chunks)} chunks → {out_path}")
        return str(out_path)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m docmind.ingestion.pdf_extractor <path_to_pdf>")
        sys.exit(1)

    extractor = PDFExtractor(sys.argv[1])
    chunks = extractor.extract()
    extractor.save()

    # Print a sample of each chunk type
    for chunk_type in ["heading", "text", "table", "figure"]:
        sample = next((c for c in chunks if c.chunk_type == chunk_type), None)
        if sample:
            print(f"\n--- Sample {chunk_type} chunk ---")
            print(f"  ID:      {sample.chunk_id}")
            print(f"  Page:    {sample.page}")
            print(f"  Content: {sample.content[:120]}...")