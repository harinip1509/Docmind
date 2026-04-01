import ollama
from config import LLM_MODEL


PROMPT_TEMPLATES = {
    "qa": """You are DocMind, an offline document intelligence assistant.
Answer the question using ONLY the context below. 
If the answer is not in the context, say "I couldn't find that in the document."
Always cite which page your answer comes from.

Context:
{context}

Question: {query}

Answer:""",

    "summarize": """You are DocMind. Summarize the following document excerpts 
into a clear, structured summary with key points.

Context:
{context}

Provide:
1. One-line overview
2. Key points (bullet list)
3. Important numbers/dates if any

Summary:""",

    "compare": """You are DocMind. Compare and contrast the following excerpts 
from the document. Identify agreements, contradictions, and unique points.

Context:
{context}

Comparison:"""
}


class Generator:
    def __init__(self):
        self.model = LLM_MODEL
        print(f"✅ Generator ready — using model: {self.model}")

    def _build_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Source {i+1} | Page {chunk['page']} | {chunk['chunk_type']}]\n"
                f"{chunk['content']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, chunks: list[dict], mode: str = "qa") -> dict:
        context = self._build_context(chunks)
        template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["qa"])
        prompt = template.format(context=context, query=query)

        print(f"\n🤖 Generating ({mode} mode) with {self.model}...")
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"]
        citations = [
            {"page": c["page"], "type": c["chunk_type"], "excerpt": c["content"][:100]}
            for c in chunks
        ]

        return {
            "query":     query,
            "answer":    answer,
            "mode":      mode,
            "citations": citations,
            "model":     self.model
        }

    def stream(self, query: str, chunks: list[dict], mode: str = "qa"):
        """Stream tokens to terminal as they generate."""
        context = self._build_context(chunks)
        template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["qa"])
        prompt = template.format(context=context, query=query)

        print(f"\n🤖 Streaming ({mode} mode)...\n")
        for part in ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            print(part["message"]["content"], end="", flush=True)
        print("\n")