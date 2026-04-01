import os
from pathlib import Path

from rag.config import PROJECT_ROOT


def training_fewshot_block() -> str:
    n = int(os.environ.get("RAG_FEW_SHOT", "0"))
    if n <= 0:
        return ""
    qpath = PROJECT_ROOT / "data" / "train" / "questions.txt"
    apath = PROJECT_ROOT / "data" / "train" / "reference_answers.txt"
    if not qpath.is_file() or not apath.is_file():
        return ""
    qs = qpath.read_text(encoding="utf-8").splitlines()
    ans = apath.read_text(encoding="utf-8").splitlines()
    pairs = []
    for q, a in zip(qs[:n], ans[:n]):
        ref0 = a.split(";")[0].strip()
        pairs.append(f"Q: {q.strip()}\nA: {ref0}")
    return "Examples (style only; answer the final question using the context below, not these):\n" + "\n\n".join(pairs) + "\n\n"
