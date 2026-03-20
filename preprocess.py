import os
import json
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup
input_folder = "raw_data"
output_file = "data/processed_data.jsonl"
os.makedirs("data", exist_ok=True)

# 2. Configure the Chunker
# Chunk size 600 is a "sweet spot" for Llama2 to get enough context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    separators=["\n\n", "\n", " ", ""]
)

processed_chunks = []

print(f"--- Starting Processing ---")

for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(input_folder, filename)
        try:
            reader = PdfReader(path)
            # Extract text page by page to maintain order
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if not page_text.strip():
                    continue
                
                # Split this page into chunks
                chunks = text_splitter.split_text(page_text)
                
                for i, chunk_text in enumerate(chunks):
                    # Clean the text (remove weird PDF artifacts)
                    clean_text = " ".join(chunk_text.split())
                    
                    # Create a structured "Document" object
                    chunk_entry = {
                        "chunk_id": f"{filename}_{page_num}_{i}",
                        "text": clean_text,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "university": "Cal Poly Pomona"
                        }
                    }
                    processed_chunks.append(chunk_entry)
        except Exception as e:
            print(f"Error on {filename}: {e}")

# 3. Save as JSONL (One JSON object per line)
with open(output_file, "w", encoding="utf-8") as f:
    for entry in processed_chunks:
        f.write(json.dumps(entry) + "\n")

print(f"--- Finished! ---")
print(f"Total Chunks Created: {len(processed_chunks)}")
print(f"Saved to: {output_file}")