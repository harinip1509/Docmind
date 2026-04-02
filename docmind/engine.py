from docmind.retrieval.retriever import Retriever
from docmind.generation.generator import Generator
from config import TOP_K


class DocMind:
    """
    Main engine — takes a query, retrieves relevant chunks,
    generates a grounded answer using local LLM.
    100% offline. No API keys.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.retriever  = Retriever(doc_id)
        self.generator  = Generator()

    def ask(self, query: str, mode: str = "qa", stream: bool = False) -> dict:
        # Step 1 — retrieve relevant chunks
        print(f"\n📡 Retrieving chunks for: '{query}'")
        chunks = self.retriever.hybrid_search(query, top_k=TOP_K)
        print(f"   Found {len(chunks)} relevant chunks")

        # Step 2 — generate grounded answer
        if stream:
            self.generator.stream(query, chunks, mode=mode)
            return {}

        result = self.generator.generate(query, chunks, mode=mode)

        # Step 3 — print result
        print(f"\n{'='*60}")
        print(f"📄 ANSWER ({mode.upper()} mode)")
        print(f"{'='*60}")
        print(result["answer"])
        print(f"\n📚 Citations:")
        for i, c in enumerate(result["citations"]):
            print(f"   [{i+1}] Page {c['page']} ({c['type']}): {c['excerpt']}...")
        print(f"{'='*60}\n")

        return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: python -m docmind.engine <doc_id> "<query>" [mode] [--stream]')
        print('Modes: qa | summarize | compare')
        sys.exit(1)

    doc_id  = sys.argv[1]
    query   = sys.argv[2]
    mode    = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "qa"
    stream  = "--stream" in sys.argv

    engine = DocMind(doc_id)
    engine.ask(query, mode=mode, stream=stream)