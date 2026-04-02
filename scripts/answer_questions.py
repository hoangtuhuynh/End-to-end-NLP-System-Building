"""Run RAG on a questions.txt file and write one answer per line (system_output.txt)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import DEFAULT_INDEX_DIR
from rag.pipeline import answer_question, answer_question_extractive, answer_question_closed_book
from rag.store import load_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=ROOT / "data" / "test" / "questions.txt")
    ap.add_argument("--output", type=Path, default=ROOT / "system_outputs" / "system_output.txt")
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument(
        "--extractive",
        action="store_true",
        help="Skip LLM; return a short snippet from the top retrieval (debug / no GPU or no LLM).",
    )
    ap.add_argument(
        "--closed-book",
        action="store_true",
        help="Closed-book baseline: run the reader model without retrieved context.",
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    questions = args.questions.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    index = None
    if not args.closed_book and not args.extractive:
        # Full RAG needs the retrieval index.
        index = load_index(args.index_dir)
    elif args.extractive:
        # Extractive baseline also needs the retrieval index.
        index = load_index(args.index_dir)
    for i, q in enumerate(questions, start=1):
        q = q.strip()
        if not q:
            out_lines.append("")
            continue
        print(f"[{i}/{len(questions)}] {q[:80]}...", flush=True)
        if args.extractive:
            ans = answer_question_extractive(index, q)
        elif args.closed_book:
            ans = answer_question_closed_book(index, q)
        else:
            ans = answer_question(index, q)
        out_lines.append(ans.replace("\n", " ").strip())
    args.output.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} lines to {args.output}")


if __name__ == "__main__":
    main()
