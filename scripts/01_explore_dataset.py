"""
STEP 1 — Explore NyayaAnumana Dataset Structure
================================================
Run this FIRST. It will print the exact column names, sample rows,
and statistics so you can confirm before chunking.

Usage:
    python 01_explore_dataset.py --root "C:/path/to/your/dataset"
"""
import csv
csv.field_size_limit(10_000_000)
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── Argument ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, required=True,
                    help="Root folder containing train/, test/, dev/")
args = parser.parse_args()
ROOT = Path(args.root)

# ── Find all JSON/JSONL files recursively ────────────────────────────────────
def find_data_files(folder: Path) -> list[Path]:
    files = []
    for ext in ["*.json", "*.jsonl", "*.csv"]:
        files.extend(folder.rglob(ext))
    return sorted(files)

all_files = find_data_files(ROOT)
print(f"\n{'='*60}")
print(f"DATASET ROOT: {ROOT}")
print(f"{'='*60}")
print(f"Total data files found: {len(all_files)}\n")
for f in all_files:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"  {f.relative_to(ROOT)}  ({size_mb:.2f} MB)")

# ── Load and inspect each file ───────────────────────────────────────────────
def load_file(path: Path) -> list[dict]:
    """Load JSON, JSONL, or CSV into a list of dicts."""
    if path.suffix == ".jsonl":
        records = []
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
        return records
    elif path.suffix == ".json":
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # might be {"data": [...]} or {"train": [...]}
            for key in ["data", "train", "cases", "records"]:
                if key in data:
                    return data[key]
            return [data]
        elif path.suffix == ".csv":
            import csv
            csv.field_size_limit(10_000_000)  # allow up to 10MB per field
            rows = []
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 2:   # only read 3 rows for exploration
                        break
            return rows
    return []

print(f"\n{'='*60}")
print("FILE CONTENTS INSPECTION")
print(f"{'='*60}")

all_keys_seen = set()
total_records = 0

for fpath in all_files[:20]:  # inspect first 20 files max
    records = load_file(fpath)
    if not records:
        continue
    
    total_records += len(records)
    sample = records[0]
    keys = list(sample.keys()) if isinstance(sample, dict) else []
    all_keys_seen.update(keys)
    
    print(f"\n── {fpath.relative_to(ROOT)}")
    print(f"   Records : {len(records)}")
    print(f"   Columns : {keys}")
    
    if isinstance(sample, dict):
        print(f"   Sample row (first record):")
        for k, v in sample.items():
            val_str = str(v)
            if len(val_str) > 120:
                val_str = val_str[:120] + "..."
            print(f"     {k:30s} | {val_str}")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total records across all files : {total_records}")
print(f"All unique column names seen   : {sorted(all_keys_seen)}")

# ── Check text length distribution ───────────────────────────────────────────
print(f"\n{'='*60}")
print("TEXT LENGTH ANALYSIS")
print(f"{'='*60}")
print("(checking which field has the main case text...)")

# Try to load train folder
train_folder = ROOT / "train"
train_files  = find_data_files(train_folder) if train_folder.exists() else []

if train_files:
    sample_file = train_files[0]
    records = load_file(sample_file)[:500]  # sample 500
    
    if records and isinstance(records[0], dict):
        for key in records[0].keys():
            lengths = [len(str(r.get(key, "") or "")) for r in records]
            avg_len = sum(lengths) / len(lengths)
            max_len = max(lengths)
            if avg_len > 100:  # only show substantial text fields
                print(f"  Field '{key}': avg={avg_len:.0f} chars, max={max_len} chars")

print("\n✅ Exploration complete.")
print("\nNEXT STEP: Open 02_chunk_cases.py and confirm the field names at the top.")