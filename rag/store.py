"""Dense retrieval index using Sentence-Transformers + NumPy (no ChromaDB)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import DEFAULT_JSONL, EMBED_MODEL


INDEX_DIRNAME = "rag_index"


@dataclass
class SearchResult:
    documents: list[str]
    metadatas: list[dict]


class PolicyIndex:
    def __init__(
        self,
        model: SentenceTransformer,
        texts: list[str],
        metadatas: list[dict],
        embeddings: np.ndarray,
    ):
        self.model = model
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings

    def query(self, question: str, k: int) -> SearchResult:
        q = self.model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = self.embeddings @ q
        top_idx = np.argsort(-sims)[:k]
        docs = [self.texts[i] for i in top_idx]
        meta = [self.metadatas[i] for i in top_idx]
        return SearchResult(documents=docs, metadatas=meta)


def _index_dir(base: Path | None) -> Path:
    root = Path(__file__).resolve().parent.parent
    return Path(base) if base else root / "data" / INDEX_DIRNAME


def build_vector_store(
    jsonl_path: Path | None = None,
    index_dir: Path | None = None,
    reset: bool = True,
) -> PolicyIndex:
    jsonl_path = Path(jsonl_path or DEFAULT_JSONL)
    out_dir = _index_dir(index_dir)
    if reset and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    texts: list[str] = []
    metadatas: list[dict] = []
    for obj in entries:
        texts.append(obj["text"])
        meta = obj["metadata"]
        metadatas.append(
            {
                "source": str(meta.get("source", "")),
                "page": int(meta.get("page", 0)),
                "chunk_id": str(obj.get("chunk_id", "")),
            }
        )

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas, "embed_model": EMBED_MODEL}, f)

    return PolicyIndex(model, texts, metadatas, embeddings)


def load_index(index_dir: Path | None = None) -> PolicyIndex:
    out_dir = _index_dir(index_dir)
    emb_path = out_dir / "embeddings.npy"
    corpus_path = out_dir / "corpus.json"
    if not emb_path.is_file() or not corpus_path.is_file():
        raise FileNotFoundError(
            f"Missing index under {out_dir}. Run: python scripts/build_index.py"
        )
    with open(corpus_path, encoding="utf-8") as f:
        payload = json.load(f)
    model_name = payload.get("embed_model", EMBED_MODEL)
    model = SentenceTransformer(model_name)
    embeddings = np.load(emb_path)
    return PolicyIndex(
        model,
        payload["texts"],
        payload["metadatas"],
        embeddings,
    )
