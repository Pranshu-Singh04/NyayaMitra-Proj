"""
STEP 2 — Chunk NyayaAnumana Cases
===================================
Reads your exact folder structure:
    dataset/
    ├── train/                          ← main training cases
    ├── test/
    │   ├── binary_test/                ← binary LJP test set
    │   ├── ternary_test/               ← ternary LJP test set
    │   └── Temporal Test Data 2020_2024_single/  ← temporal split
    └── dev/
        ├── binary_dev/
        └── ternary_dev/

Outputs → data/processed/chunks/
    cases_train_chunks.jsonl
    cases_binary_test_chunks.jsonl
    cases_ternary_test_chunks.jsonl
    cases_temporal_test_chunks.jsonl
    cases_binary_dev_chunks.jsonl
    cases_ternary_dev_chunks.jsonl
    chunk_stats.json

Usage:
    python 02_chunk_cases.py --root "C:/path/to/dataset" --output "data/processed/chunks"
"""


import csv
csv.field_size_limit(10_000_000)
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — adjust these after running 01_explore_dataset.py
# These are the MOST LIKELY field names for NyayaAnumana (from the paper).
# If your field names differ, update FIELD_MAP below.
# ══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 400    # words per chunk (safe for InLegalBERT 512 token limit)
CHUNK_OVERLAP = 80     # word overlap between consecutive chunks
MIN_CHUNK_LEN = 40     # discard chunks shorter than this (words)

# Mapping: logical name → possible actual column names (tried in order)
FIELD_MAP = {
    "case_id"      : ["filename"],
    "case_name"    : ["filename"],
    "court"        : ["filename"],   # we'll extract court from filename
    "court_level"  : ["filename"],
    "date"         : ["filename"],   # we'll extract year from filename
    "outcome"      : ["label"],
    "statutes"     : [],
    "facts"        : ["text"],
    "issues"       : [],
    "judgment_text": ["text"],
    "headnotes"    : [],
    "held"         : [],
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_field(record: dict, logical_name: str) -> str:
    """Try all possible column names for a logical field; return first match."""
    for candidate in FIELD_MAP.get(logical_name, [logical_name]):
        val = record.get(candidate)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(words[start:end]) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def build_full_text(record: dict) -> str:
    filename = record.get("filename", "")
    text     = record.get("text", "")
    label    = record.get("label", "")
    return f"Case: {filename}\nOutcome: {label}\n\n{text}"

def extract_meta(record: dict, split_name: str, chunk_idx: int) -> dict:
    filename = record.get("filename", "")
    # Extract court and year from filename e.g. "SupremeCourt_1960_50"
    parts      = filename.split("_")
    court      = parts[0] if parts else "Unknown"
    year       = parts[1] if len(parts) > 1 else ""
    return {
        "case_id"    : filename,
        "case_name"  : filename,
        "court"      : court,
        "court_level": court,
        "date"       : year,
        "outcome"    : record.get("label", "Unknown"),
        "statutes"   : "",
        "split"      : split_name,
        "chunk_idx"  : chunk_idx,
        "source"     : "nyayaanumana",
    }

def load_records(folder: Path) -> list[dict]:
    """Load all JSON/JSONL files from a folder into a flat list of records."""
    records = []
    if not folder.exists():
        print(f"  ⚠️  Folder not found: {folder}")
        return records

    files = sorted(list(folder.rglob("*.jsonl")) +
                   list(folder.rglob("*.json"))  +
                   list(folder.rglob("*.csv")))

    if not files:
        print(f"  ⚠️  No data files in: {folder}")
        return records

    for fpath in files:
        try:
            if fpath.suffix == ".jsonl":
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            elif fpath.suffix == ".json":
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    for key in ["data", "cases", "records", "train", "test", "dev"]:
                        if key in data and isinstance(data[key], list):
                            records.extend(data[key])
                            break
                    else:
                        records.append(data)
            elif fpath.suffix == ".csv":
                import csv
                csv.field_size_limit(10_000_000)
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(dict(row))
        except Exception as e:
            print(f"  ⚠️  Error reading {fpath.name}: {e}")

    return records

def process_split(folder: Path, split_name: str, output_dir: Path) -> dict:
    """Load, chunk, and save one split. Returns stats dict."""
    print(f"\n{'─'*55}")
    print(f"Processing split: {split_name}")
    print(f"Folder: {folder}")

    records = load_records(folder)
    if not records:
        print(f"  No records found — skipping.")
        return {"split": split_name, "cases": 0, "chunks": 0}

    print(f"  Loaded {len(records)} cases")

    # Peek at first record to show fields
    if records:
        sample = records[0]
        print(f"  Fields found: {list(sample.keys())[:12]}")
        full_text_sample = build_full_text(sample)
        print(f"  Sample full-text length: {len(full_text_sample)} chars")

    all_chunks = []
    global_chunk_id = 0
    skipped = 0

    for record in tqdm(records, desc=f"  Chunking {split_name}"):
        full_text = build_full_text(record)
        if len(full_text.split()) < MIN_CHUNK_LEN:
            skipped += 1
            continue

        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        meta        = extract_meta(record, split_name, chunk_idx=0)

        for i, chunk_text_content in enumerate(text_chunks):
            entry = {
                "chunk_id"   : f"{split_name}_{global_chunk_id}",
                "text"       : chunk_text_content,
                **meta,
                "chunk_idx"  : i,
                "total_chunks_in_case": len(text_chunks),
            }
            all_chunks.append(entry)
            global_chunk_id += 1

    print(f"  Skipped (too short): {skipped}")
    print(f"  Total chunks: {len(all_chunks)}")

    # Save
    out_path = output_dir / f"cases_{split_name}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  ✅ Saved: {out_path}")

    return {
        "split"       : split_name,
        "cases"       : len(records),
        "chunks"      : len(all_chunks),
        "skipped"     : skipped,
        "output_file" : str(out_path),
        "avg_chunks_per_case": round(len(all_chunks) / max(len(records),1), 2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   type=str, required=True,
                        help="Dataset root (contains train/, test/, dev/)")
    parser.add_argument("--output", type=str, default="data/processed/chunks",
                        help="Output directory for chunk JSONL files")
    args = parser.parse_args()

    ROOT       = Path(args.root)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*55)
    print("NyayaAnumana Chunking Pipeline")
    print("="*55)
    print(f"Root:   {ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")

    # Define all splits to process based on YOUR folder structure
    splits = [
        (ROOT / "train" / "binary_multi_train",                          "train"),
        (ROOT / "test"  / "binary_test",                                 "binary_test"),
        (ROOT / "test"  / "ternary_test",                                "ternary_test"),
        (ROOT / "test"  / "Temporal Test Data 2020_2024_single" / "SCI", "temporal_test_sci"),
        (ROOT / "test"  / "Temporal Test Data 2020_2024_single" / "HCs", "temporal_test_hc"),
        (ROOT / "test"  / "Temporal Test Data 2020_2024_single" / "Dailyorder", "temporal_test_daily"),
        (ROOT / "test"  / "Temporal Test Data 2020_2024_single" / "Tribunals", "temporal_test_tribunal"),
        (ROOT / "dev"   / "binary_dev",                                  "binary_dev"),
        (ROOT / "dev"   / "ternary_dev",                                 "ternary_dev"),
    ]

    all_stats = []
    for folder, split_name in splits:
        stats = process_split(folder, split_name, OUTPUT_DIR)
        all_stats.append(stats)

    # Also create a MERGED index file (train only — used for RAG retrieval)
    print(f"\n{'─'*55}")
    print("Creating merged train chunks for RAG indexing...")
    train_chunks_path = OUTPUT_DIR / "cases_train_chunks.jsonl"
    if train_chunks_path.exists():
        print(f"  Already exists: {train_chunks_path}")

    # Summary
    print(f"\n{'='*55}")
    print("CHUNKING SUMMARY")
    print(f"{'='*55}")
    total_cases  = sum(s.get("cases", 0)  for s in all_stats)
    total_chunks = sum(s.get("chunks", 0) for s in all_stats)
    for s in all_stats:
        print(f"  {s['split']:35s} | cases: {s.get('cases',0):6,} | chunks: {s.get('chunks',0):8,}")
    print(f"  {'TOTAL':35s} | cases: {total_cases:6,} | chunks: {total_chunks:8,}")

    # Save stats
    stats_path = OUTPUT_DIR / "chunk_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✅ Stats saved: {stats_path}")
    print(f"\nNEXT: Run 03_chunk_statutes.py, then 04_embed_and_index.py")

if __name__ == "__main__":
    main()