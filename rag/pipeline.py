from rag.config import TOP_K
from rag.llm import generate_answer, generate_answer_closed_book
from rag.store import PolicyIndex


def retrieve_context(index: PolicyIndex, question: str, k: int | None = None) -> str:
    k = k if k is not None else TOP_K
    result = index.query(question, k=k)
    parts = []
    for i, (doc, meta) in enumerate(zip(result.documents, result.metadatas), start=1):
        src = meta.get("source", "?")
        page = meta.get("page", "?")
        parts.append(f"[{i}] ({src} page {page})\n{doc}")
    return "\n\n".join(parts)


def answer_question(index: PolicyIndex, question: str) -> str:
    ctx = retrieve_context(index, question)
    return generate_answer(ctx, question)


def answer_question_extractive(index: PolicyIndex, question: str) -> str:
    result = index.query(question, k=1)
    doc = result.documents[0]
    snippet = doc[:400].strip()
    return snippet.split(". ")[0][:200]


def answer_question_closed_book(index: PolicyIndex, question: str) -> str:
    # Closed-book baseline: no retrieved context snippets.
    _ = index  # unused; keeps signature consistent
    return generate_answer_closed_book(question)
