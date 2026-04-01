"""Build the dense retrieval index from data/processed_data.jsonl."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import DEFAULT_INDEX_DIR, DEFAULT_JSONL
from rag.store import build_vector_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = ap.parse_args()

    build_vector_store(jsonl_path=args.jsonl, index_dir=args.index_dir, reset=True)
    print(f"Index saved under {args.index_dir}")


if __name__ == "__main__":
    main()
