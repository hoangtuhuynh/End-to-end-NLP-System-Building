# CS 5170 Assignment 2 — RAG for Cal Poly Pomona policies

End-to-end retrieval-augmented generation over publicly available campus policy PDFs. This repo matches the course deliverables: **your** knowledge corpus, **your** train/test annotations, preprocessing code, **RAG code**, and scripts to produce `system_output.txt` for the staff test set.

**Course model policy:** use **open models available on Hugging Face** (e.g. local **Ollama** with an open checkpoint, or **Hugging Face Inference** / other HF-compatible hosting). Do **not** use closed APIs such as OpenAI for grading compliance.

## Rubric alignment (what is here)

| Deliverable | In this repo |
|-------------|----------------|
| **Knowledge resource** | `raw_data/*.pdf` → `preprocess.py` → `data/processed_data.jsonl` + `data/rag_index/` |
| **Train / test annotations** | `data/train/questions.txt`, `data/train/reference_answers.txt`, `data/test/…`, IAA line list under `data/test/iaa_question_line_numbers.txt` |
| **Preprocessing code** | `preprocess.py` |
| **Model / RAG code** | `rag/` (embeddings, retrieval, reader), `scripts/` |
| **Run instructions** | This file + commands below |

## Quick start

```text
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python preprocess.py           # if you change raw PDFs
python scripts/build_index.py

# Reader: local Ollama (pull an open model, e.g. llama3.2)
python scripts/answer_questions.py --questions path\to\questions.txt --output system_outputs\system_output.txt

# Reader: Hugging Face Inference (set RAG_HF_MODEL; use HF_TOKEN if required)
set RAG_LLM_BACKEND=hf
set RAG_HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
set HF_TOKEN=...
python scripts/answer_questions.py ...
```

**Optional:** few-shot style hints from the first *N* lines of `data/train/`: `set RAG_FEW_SHOT=3`. **Tuning:** `RAG_TOP_K`, `RAG_EMBED_MODEL`, `RAG_OLLAMA_MODEL`.

**Evaluate** on your dev references:

```text
python scripts/evaluate.py --predictions system_outputs\system_output.txt --references data\test\reference_answers.txt
```

**Sanity check** annotation line counts: `python verify_annotations.py`

**Debug retrieval only (no LLM):** `python scripts/answer_questions.py --extractive ...` (baseline for the report, not for final submission).

## Report checklist (from the syllabus)

Cover **domain & data** (why RAG, corpus construction, extraction tools, annotation protocol, volume, IAA), **models** (≥2 variants / baselines, justification), **results** (numbers on your test split; significance if you compare systems), and **analysis** (per question type, **retrieve+augment vs closed-book**, example outputs from ≥2 systems).

## Credits

- Embeddings: [Sentence-Transformers](https://www.sbert.net/) / [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Reader: your chosen **HF-listed** instruction model (Ollama or HF Inference)
- PDF parsing: `pypdf`; chunking: LangChain text splitters

See `INSTRUCTIONS.txt` for a plain-text copy of run steps.
