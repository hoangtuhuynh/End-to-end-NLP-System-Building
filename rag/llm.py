import json
import os
import urllib.error
import urllib.request

from rag.config import LLM_BACKEND, OLLAMA_HOST, OLLAMA_MODEL, RAG_HF_MODEL
from rag.fewshot import training_fewshot_block


SYSTEM_PREFIX = """You answer questions about Cal Poly Pomona official policies using ONLY the provided context snippets.
Rules:
- Answer in the fewest words possible (often a name, number, date, or short phrase), matching how references are written.
- If the context does not contain the answer, reply exactly: Cannot answer from provided context.
- Do not cite sources in the answer. No preamble."""

SYSTEM_PREFIX_CLOSED_BOOK = """You answer questions about Cal Poly Pomona official policies.
Rules:
- Answer in the fewest words possible (often a name, number, date, or short phrase), matching how references are written.
- If you do not know the answer with confidence, reply exactly: Cannot answer from provided context.
- Do not cite sources in the answer. No preamble."""


def _ollama_chat(prompt: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama /api/chat HTTP {e.code}: {detail[:800]}"
        ) from e
    return (payload.get("message") or {}).get("content", "").strip()


def _ollama_chat_closed_book(prompt: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PREFIX_CLOSED_BOOK},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama /api/chat (closed-book) HTTP {e.code}: {detail[:800]}"
        ) from e
    return (payload.get("message") or {}).get("content", "").strip()


def _ollama_generate(prompt: str) -> str:
    """Fallback when /api/chat returns 5xx on some setups (CPU/RAM limits)."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    full_prompt = f"{SYSTEM_PREFIX}\n\n{prompt}"
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama /api/generate HTTP {e.code}: {detail[:800]}"
        ) from e
    return (payload.get("response") or "").strip()


def _ollama_generate_answer(context_block: str, question: str) -> str:
    fs = training_fewshot_block()
    prompt = f"""{fs}Context:
{context_block}

Question: {question}
Answer (concise):"""
    try:
        return _ollama_chat(prompt)
    except RuntimeError:
        return _ollama_generate(prompt)


def _ollama_generate_closed_book_answer(question: str) -> str:
    fs = training_fewshot_block()
    prompt = f"""{fs}Question: {question}
Answer (concise):"""
    try:
        return _ollama_chat_closed_book(prompt)
    except RuntimeError:
        # If /api/chat fails, fall back to /api/generate with the closed-book system prefix.
        url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
        full_prompt = f"{SYSTEM_PREFIX_CLOSED_BOOK}\n\n{prompt}"
        body = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return (payload.get("response") or "").strip()


def _hf_inference_chat(prompt: str) -> str:
    """Open models via Hugging Face Inference API (see course model policy)."""
    from huggingface_hub import InferenceClient

    model = (RAG_HF_MODEL or "").strip()
    if not model:
        raise ValueError(
            "Set RAG_HF_MODEL to a Hugging Face model id, e.g. "
            "meta-llama/Llama-3.2-3B-Instruct (with HF_TOKEN if required)."
        )
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(model=model, token=token)
    out = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.1,
    )
    choice = out.choices[0]
    if isinstance(choice, dict):
        msg = choice.get("message") or {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
    else:
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", "") if msg is not None else ""
    return (content or "").strip()


def generate_answer(context_block: str, question: str) -> str:
    backend = LLM_BACKEND.lower()
    if backend == "hf":
        fs = training_fewshot_block()
        prompt = f"""{fs}Context:
{context_block}

Question: {question}
Answer (concise):"""
        return _hf_inference_chat(prompt)
    if backend == "ollama":
        try:
            return _ollama_generate_answer(context_block, question)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed ({e}). Is the Ollama app running? "
                f"`ollama pull {OLLAMA_MODEL}` and try a smaller model if you see HTTP 500 (e.g. llama3.2:1b)."
            ) from e
    raise ValueError(
        f"Unknown RAG_LLM_BACKEND={backend!r}. Use 'ollama' (local open model) or 'hf' (Hugging Face Inference)."
    )


def generate_answer_closed_book(question: str) -> str:
    backend = LLM_BACKEND.lower()
    if backend == "hf":
        from huggingface_hub import InferenceClient

        fs = training_fewshot_block()
        prompt = f"""{fs}Question: {question}
Answer (concise):"""
        model = (RAG_HF_MODEL or "").strip()
        if not model:
            raise ValueError(
                "Set RAG_HF_MODEL to a Hugging Face model id, e.g. meta-llama/Llama-3.2-3B-Instruct."
            )
        token = os.environ.get("HF_TOKEN")
        client = InferenceClient(model=model, token=token)
        out = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PREFIX_CLOSED_BOOK},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        choice = out.choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            content = msg.get("content", "") if isinstance(msg, dict) else ""
        else:
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", "") if msg is not None else ""
        return (content or "").strip()

    if backend == "ollama":
        try:
            return _ollama_generate_closed_book_answer(question)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed ({e}). Is the Ollama app running?"
            ) from e

    raise ValueError(
        f"Unknown RAG_LLM_BACKEND={backend!r}. Use 'ollama' or 'hf'."
    )
