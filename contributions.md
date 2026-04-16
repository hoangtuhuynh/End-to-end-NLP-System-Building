## Team members

- Hoang Tu Huynh (`hoangtuhuynh@cpp.edu`)
- Thoa Nguyen (`thoatnguyen@cpp.edu`)

## Data annotation contributions

We created the QA datasets under `data/train/` (42 items) and `data/test/` (28 items).

### Training set (`data/train/`)

- Hoang Tu Huynh: items 1–21  
- Thoa Nguyen: items 22–42

### Test set (`data/test/`)

- Hoang Tu Huynh: items 1–14  
- Thoa Nguyen: items 15–28

### IAA (inter-annotator agreement) subset

Two independent annotations were collected for the IAA subset listed in `data/test/iaa_question_line_numbers.txt` (8 questions).

- Annotator 1: Hoang Tu Huynh (`data/test/iaa_annotator1.txt`)
- Annotator 2: Thoa Nguyen (`data/test/iaa_annotator2.txt`)

## Data collection, processing, and modeling contributions

### Hoang Tu Huynh

- Collected the policy PDFs and organized the knowledge resource under `raw_data/`.
- Implemented preprocessing and chunking from PDFs to JSONL (`preprocess.py`), including metadata fields.
- Implemented the dense retrieval index build/load and retrieval pipeline (`rag/`, `scripts/build_index.py`).
- Implemented the QA runner that maps `questions.txt` → `system_output.txt` (`scripts/answer_questions.py`).
- Implemented evaluation scripts and metrics used in experiments (`scripts/evaluate.py`) and helped add baselines (extractive + closed-book).
- Maintained the GitHub repository structure and run instructions (`README.md`, `INSTRUCTIONS.txt`, `.gitignore`).

### Thoa Nguyen

- Drafted and validated question/answer annotations against the policy documents and extracted text.
- Provided the second independent annotation for the IAA subset and reviewed reference answer formatting (including alternate references with `;`).
- Ran system experiments on the development test set, recorded metrics, and helped interpret error patterns for the report.
- Co-wrote the final report in the ACL template format and ensured the report aligns with the course rubric and model/data policy.

