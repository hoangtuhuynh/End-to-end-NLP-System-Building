import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_JSONL = PROJECT_ROOT / "data" / "processed_data.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "6"))

# Reader LLM (course policy: use open models on Hugging Face; local Ollama counts as hosting an open checkpoint).
# RAG_LLM_BACKEND: "ollama" | "hf" (Hugging Face Serverless Inference / Inference Providers)
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "llama3.2")

# Hugging Face chat model id (must be allowed on your HF account; many need HF_TOKEN)
RAG_HF_MODEL = os.environ.get("RAG_HF_MODEL", "")
