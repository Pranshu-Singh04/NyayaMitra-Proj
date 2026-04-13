"""
STEP 5 — Test Retrieval Quality
================================
Usage:
    python 05_test_retrieval.py --index_dir indexes/
    python 05_test_retrieval.py --index_dir indexes/ --query "murder Section 302 bail"
"""

import json, time, argparse
import numpy as np
import faiss, torch
from pathlib import Path
from collections import Counter
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--index_dir", default="indexes")
parser.add_argument("--query",     default=None)
parser.add_argument("--top_k",     type=int, default=5)
args = parser.parse_args()

INDEX_DIR = Path(args.index_dir)
config    = json.loads((INDEX_DIR / "index_config.json").read_text())

MODEL_NAME   = config["model_name"]
QUERY_PREFIX = config["query_prefix"]
NORMALIZE    = config["normalize"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, device=device)

case_index    = faiss.read_index(str(INDEX_DIR / "faiss_cases.index"))
statute_index = faiss.read_index(str(INDEX_DIR / "faiss_statutes.index"))

def load_meta(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

case_meta    = load_meta(INDEX_DIR / "faiss_cases_metadata.jsonl")
statute_meta = load_meta(INDEX_DIR / "faiss_statutes_metadata.jsonl")
print(f"Cases: {case_index.ntotal:,} | Statutes: {statute_index.ntotal:,}")

def retrieve(query, index, metadata, top_k=5):
    qtext = f"{QUERY_PREFIX}{query}" if QUERY_PREFIX else query
    with torch.no_grad():
        q = model.encode([qtext], normalize_embeddings=NORMALIZE,
                         convert_to_numpy=True).astype("float32")
    t0 = time.time()
    scores, idxs = index.search(q, top_k)
    lat = (time.time() - t0) * 1000
    results = []
    for s, i in zip(scores[0], idxs[0]):
        if 0 <= i < len(metadata):
            r = metadata[i].copy(); r["score"] = round(float(s), 4)
            results.append(r)
    return results, lat

def show_cases(results, lat):
    print(f"  [{lat:.0f}ms]")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r.get('case_name','?')[:55]}")
        print(f"       {r.get('court_level','?'):20s} | {r.get('outcome','?'):15s} | {r.get('date','?')}")
        print(f"       {r.get('text','')[:110].replace(chr(10),' ')}...")

def show_statutes(results, lat):
    print(f"  [{lat:.0f}ms]")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.3f}] {r.get('source','?')} §{r.get('section_num','?')} — {r.get('section_title','?')[:45]}")
        print(f"       {r.get('text','')[:120].replace(chr(10),' ')}...")

if args.query:
    print(f"\nCASES — '{args.query}'")
    r, l = retrieve(args.query, case_index, case_meta, args.top_k)
    show_cases(r, l)
    print(f"\nSTATUTES — '{args.query}'")
    r, l = retrieve(args.query, statute_index, statute_meta, args.top_k)
    show_statutes(r, l)
else:
    tests = [
        ("murder conviction appeal Supreme Court",   "case"),
        ("cheating IPC 420 dishonest property",      "case"),
        ("bail anticipatory domestic violence",      "case"),
        ("punishment theft stolen property",         "statute"),
        ("murder death penalty life imprisonment",   "statute"),
        ("cruelty husband wife dowry 498A",          "statute"),
    ]
    for query, kind in tests:
        print(f"\n── {kind.upper()}: \"{query}\"")
        if kind == "case":
            r, l = retrieve(query, case_index, case_meta, 3)
            show_cases(r, l)
        else:
            r, l = retrieve(query, statute_index, statute_meta, 3)
            show_statutes(r, l)

    print(f"\nOUTCOME DISTRIBUTION:")
    oc = Counter(m.get("outcome","?") for m in case_meta)
    for label, cnt in oc.most_common(8):
        print(f"  {label:30s}: {cnt:7,}  ({100*cnt/len(case_meta):.1f}%)")

print("\n✅ Done. If scores > 0.75 → retrieval is working well.")
print("If scores < 0.60 → check chunking quality or switch model.")