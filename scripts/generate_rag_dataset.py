#!/usr/bin/env python3
"""
Generate synthetic RAG QA pairs from Oil & Gas domain documents using Gemma 4 MoE.

Optimized for:
- Continuous saving (JSONL) to prevent data loss.
- Smart chunking for short documents.
- Resumability.
"""

import json
import os
import random
import time
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# ── Config ──────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"
OUT_DIR = Path(__file__).parent.parent / "datasets"
RAW_DATA_PATH = OUT_DIR / "raw_dataset.jsonl"
CHUNK_SIZE = 5000  # Based on 3KB-32KB file sizes
CHUNK_OVERLAP = 400
QA_PER_CHUNK = 3  # Increased slightly for larger chunks
TEST_RATIO = 0.15
RATE_LIMIT_DELAY = 4
MODEL_ID = "models/gemma-4-26b-a4b-it"

SYSTEM_PROMPT = (
    "You are a precise Oil & Gas domain extraction assistant. "
    "You ONLY answer based on the provided context. "
    "If the context does not contain the answer, respond with "
    '{"answer": "NOT_FOUND", "source_entity": "N/A"}. '
    "Always respond in valid JSON with keys: answer, source_entity."
)

GENERATION_PROMPT = """You are a dataset generator for training an AI extraction model.

Given the following technical passage from an Oil & Gas document, generate exactly {n} question-answer pairs.

Rules:
1. Questions must be answerable ONLY from the provided passage.
2. Questions should be factual and specific (who, what, how much, what range, etc.).
3. Answers must be extracted or directly derived from the passage — no outside knowledge.
4. Each answer MUST be a JSON object with keys "answer" and "source_entity".
   - "answer": the factual answer (keep it concise, 1-3 sentences max)
   - "source_entity": the key technical concept or entity the answer relates to

Return your output as a JSON array of objects, each with keys "question" and "response".
Example:
[
  {{
    "question": "What is the maximum depth for PCP deployment?",
    "response": {{"answer": "6,000 ft (1,800 m)", "source_entity": "Progressing Cavity Pump"}}
  }}
]

--- PASSAGE ---
{passage}
--- END PASSAGE ---

Return ONLY the JSON array. No markdown fencing, no explanation."""

def chunk_document(text: str) -> list[str]:
    """Split document text into overlapping chunks if it exceeds CHUNK_SIZE."""
    if len(text) <= CHUNK_SIZE:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk) < 500 and chunks:
            break
        chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks

def generate_qa_pairs(client, chunk: str, n: int = QA_PER_CHUNK, max_retries: int = 5) -> list[dict]:
    """Call Gemma 4 via Google API to generate QA pairs from a chunk."""
    prompt = GENERATION_PROMPT.format(n=n, passage=chunk)
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
            )
            text = response.text.strip()
            # Strip markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            pairs = json.loads(text)
            return pairs
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "resource_exhausted" in err_str:
                wait = (attempt + 1) * 30
                print(f"  ⏳ Rate limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"  ⚠ Error: {e}")
                return []
    return []

def get_processed_ids():
    """Read raw_dataset.jsonl to find which (doc, chunk) IDs are already done."""
    processed = set()
    if not RAW_DATA_PATH.exists():
        return processed
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add(data.get("id"))
            except:
                continue
    return processed

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_ids = get_processed_ids()
    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    print(f"📚 Found {len(doc_files)} documents. Already processed {len(processed_ids)} chunks.")

    with open(RAW_DATA_PATH, "a", encoding="utf-8") as raw_f:
        for doc_path in doc_files:
            text = doc_path.read_text(encoding="utf-8")
            chunks = chunk_document(text)
            doc_name = doc_path.name
            
            print(f"\n📄 {doc_name} ({len(chunks)} chunks)")
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_name}_{i}"
                if chunk_id in processed_ids:
                    print(f"  Chunk {i+1} [Skipped]")
                    continue
                
                print(f"  Chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
                pairs = generate_qa_pairs(client, chunk)
                
                if pairs:
                    # Save immediately
                    entry = {
                        "id": chunk_id,
                        "doc": doc_name,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "qa_pairs": pairs
                    }
                    raw_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    print(f"✅ {len(pairs)} pairs")
                else:
                    print("⏭ skipped")
                
                time.sleep(RATE_LIMIT_DELAY)

    # Final Export: Convert JSONL to LLaMA-Factory alpaca format
    print("\n📦 Exporting to final format...")
    all_formatted = []
    if RAW_DATA_PATH.exists():
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunk_text = data["chunk_text"]
                for pair in data["qa_pairs"]:
                    question = pair.get("question", "")
                    response = pair.get("response", {})
                    all_formatted.append({
                        "system": SYSTEM_PROMPT,
                        "instruction": f"Context: {chunk_text}\n\nQuestion: {question}",
                        "input": "",
                        "output": json.dumps(response, ensure_ascii=False),
                    })

    random.seed(42)
    random.shuffle(all_formatted)
    split_idx = int(len(all_formatted) * (1 - TEST_RATIO))
    
    with open(OUT_DIR / "train.json", "w", encoding="utf-8") as f:
        json.dump(all_formatted[:split_idx], f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "test.json", "w", encoding="utf-8") as f:
        json.dump(all_formatted[split_idx:], f, indent=2, ensure_ascii=False)

    print(f"✅ Done! Total samples: {len(all_formatted)}")

if __name__ == "__main__":
    main()
