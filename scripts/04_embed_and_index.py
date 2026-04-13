"""
STEP 4 — Embed Chunks & Build FAISS Indexes
============================================
Reads chunked JSONL files from Step 2 & 3.
Embeds using E5-large-v2 OR InLegalBERT.
Builds two FAISS indexes:
    indexes/faiss_cases.index          ← for retrieving relevant past judgments
    indexes/faiss_statutes.index       ← for retrieving relevant statute sections

IMPORTANT — E5 Prefix Rule:
    Documents get prefix: "passage: "
    Queries   get prefix: "query: "
    (InLegalBERT needs NO prefix — it's BERT-style)

GPU memory guide:
    E5-large-v2  (1024-dim): needs ~6GB VRAM, 15-20 min per 100k chunks
    InLegalBERT  (768-dim) : needs ~3GB VRAM, 8-12 min per 100k chunks
    On CPU only            : multiply times by ~10x, use small batches

Usage:
    # Recommended (E5-large):
    python 04_embed_and_index.py --model e5 --cases data/processed/chunks/cases_train_chunks.jsonl --statutes data/processed/chunks/statute_chunks.jsonl

    # Alternative (InLegalBERT):
    python 04_embed_and_index.py --model inlegalbert --cases data/processed/chunks/cases_train_chunks.jsonl --statutes data/processed/chunks/statute_chunks.jsonl

    # CPU-safe small test run:
    python 04_embed_and_index.py --model e5 --cases data/processed/chunks/cases_train_chunks.jsonl --statutes data/processed/chunks/statute_chunks.jsonl --max_cases 10000
"""

import json
import argparse
import numpy as np
import faiss
import torch
import time
from pathlib import Path
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    "e5": {
        "model_name"  : "intfloat/e5-large-v2",
        "embed_dim"   : 1024,
        "doc_prefix"  : "passage: ",
        "query_prefix": "query: ",
        "batch_size"  : 64,       # reduce to 32 on smaller GPU
        "normalize"   : True,
        "notes"       : "Best quality. Requires ~6GB VRAM.",
    },
    "e5-base": {
        "model_name"  : "intfloat/e5-base-v2",
        "embed_dim"   : 768,
        "doc_prefix"  : "passage: ",
        "query_prefix": "query: ",
        "batch_size"  : 128,
        "normalize"   : True,
        "notes"       : "Good quality, faster. Requires ~3GB VRAM.",
    },
    "inlegalbert": {
        "model_name"  : "law-ai/InLegalBERT",
        "embed_dim"   : 768,
        "doc_prefix"  : "",       # No prefix needed
        "query_prefix": "",
        "batch_size"  : 128,
        "normalize"   : True,
        "notes"       : "Domain-specific for Indian law. Requires ~3GB VRAM.",
    },
    "multilingual-e5": {
        "model_name"  : "intfloat/multilingual-e5-large",
        "embed_dim"   : 1024,
        "doc_prefix"  : "passage: ",
        "query_prefix": "query: ",
        "batch_size"  : 64,
        "normalize"   : True,
        "notes"       : "For future Hindi/regional support.",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_jsonl(path: Path, max_records: int = None) -> list[dict]:
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_records and len(records) >= max_records:
                    break
    return records

def embed_in_batches(
    texts    : list[str],
    model,
    prefix   : str,
    batch_size: int,
    normalize: bool,
    device   : str,
) -> np.ndarray:
    """Embed texts in batches. Returns (N, embed_dim) float32 array."""
    import torch
    from sentence_transformers import SentenceTransformer

    all_embeddings = []
    prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts

    for i in tqdm(range(0, len(prefixed), batch_size), desc="  Embedding"):
        batch = prefixed[i : i + batch_size]
        with torch.no_grad():
            embs = model.encode(
                batch,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=device,
            )
        all_embeddings.append(embs)

        # Log progress every 10 batches
        if (i // batch_size) % 10 == 0 and i > 0:
            elapsed = time.time() - embed_start
            done    = i + len(batch)
            rate    = done / elapsed
            eta     = (len(prefixed) - done) / rate
            print(f"  Progress: {done}/{len(prefixed)} | {rate:.0f} texts/sec | ETA: {eta/60:.1f} min")

    return np.vstack(all_embeddings).astype("float32")

def build_faiss_index(embeddings: np.ndarray, n_total: int) -> faiss.Index:
    """
    Choose index type based on corpus size:
      < 100k  → IndexFlatIP   (exact, fast)
      100k–1M → IndexIVFFlat  (approximate, much faster search)
      > 1M    → IndexIVFPQ    (compressed, fits large datasets in RAM)
    """
    dim = embeddings.shape[1]

    if n_total < 100_000:
        print(f"  Using IndexFlatIP (exact search, n={n_total})")
        index = faiss.IndexFlatIP(dim)

    elif n_total < 1_000_000:
        nlist = min(1024, n_total // 40)
        print(f"  Using IndexIVFFlat (nlist={nlist}, n={n_total})")
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"  Training IVF index on {min(100_000, n_total)} vectors...")
        sample = embeddings[:min(100_000, n_total)]
        index.train(sample)
        index.nprobe = 64   # search 64 Voronoi cells (tune: higher = more accurate but slower)

    else:
        # Product Quantization for very large indexes (memory-efficient)
        nlist  = 2048
        m      = 32    # number of sub-quantizers
        nbits  = 8     # bits per sub-quantizer
        print(f"  Using IndexIVFPQ (nlist={nlist}, m={m}, n={n_total})")
        quantizer = faiss.IndexFlatIP(dim)
        index     = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        print(f"  Training IVFPQ index on {min(200_000, n_total)} vectors...")
        sample = embeddings[:min(200_000, n_total)]
        index.train(sample)
        index.nprobe = 128

    return index

def build_and_save_index(
    chunk_file : Path,
    index_name : str,
    cfg        : dict,
    model,
    index_dir  : Path,
    max_records: int,
    device     : str,
):
    global embed_start

    print(f"\n{'='*60}")
    print(f"Building index: {index_name}")
    print(f"Source: {chunk_file}")

    # Load chunks
    chunks = load_jsonl(chunk_file, max_records)
    print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        print("  ⚠️  No chunks found — skipping")
        return

    # Extract texts
    texts = [str(c.get("text", "")).strip() or "empty" for c in chunks]
    print(f"Embedding {len(texts)} texts with prefix='{cfg['doc_prefix']}'...")

    embed_start = time.time()
    embeddings  = embed_in_batches(
        texts     = texts,
        model     = model,
        prefix    = cfg["doc_prefix"],
        batch_size= cfg["batch_size"],
        normalize = cfg["normalize"],
        device    = device,
    )
    elapsed = time.time() - embed_start
    print(f"  Done in {elapsed/60:.1f} min | Shape: {embeddings.shape}")

    # Build FAISS index
    print(f"Building FAISS index...")
    index = build_faiss_index(embeddings, len(chunks))
    index.add(embeddings)
    print(f"  Index size: {index.ntotal} vectors")

    # Save index
    idx_path = index_dir / f"{index_name}.index"
    faiss.write_index(index, str(idx_path))
    print(f"  ✅ Index saved: {idx_path}")

    # Save metadata (row i in index ↔ row i here)
    meta_path = index_dir / f"{index_name}_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  ✅ Metadata saved: {meta_path}")

    return {"index_name": index_name, "vectors": index.ntotal, "elapsed_min": round(elapsed/60, 1)}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="e5",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Embedding model to use")
    parser.add_argument("--cases",    type=str, required=True,
                        help="Path to cases JSONL chunk file (from Step 2)")
    parser.add_argument("--statutes", type=str, required=True,
                        help="Path to statute JSONL chunk file (from Step 3)")
    parser.add_argument("--output",   type=str, default="indexes",
                        help="Output directory for FAISS indexes")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="Limit number of case chunks (for testing, e.g. 10000)")
    parser.add_argument("--gpu",      action="store_true",
                        help="Force GPU even if not auto-detected")
    args = parser.parse_args()

    cfg       = MODEL_CONFIGS[args.model]
    INDEX_DIR = Path(args.output)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    device = "cuda" if (torch.cuda.is_available() or args.gpu) else "cpu"
    print(f"\n{'='*60}")
    print(f"NyayaMitra Embedding & Indexing Pipeline")
    print(f"{'='*60}")
    print(f"Model  : {cfg['model_name']}")
    print(f"Device : {device}")
    print(f"Notes  : {cfg['notes']}")
    if device == "cpu":
        print(f"⚠️  Running on CPU — this will be slow for large datasets.")
        print(f"   Consider using --max_cases 50000 for initial testing.")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading model: {cfg['model_name']} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["model_name"], device=device)

    actual_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {actual_dim}")
    cfg["embed_dim"] = actual_dim  # update in case model reports differently

    # ── Build case index ─────────────────────────────────────────────────────
    case_stats = build_and_save_index(
        chunk_file  = Path(args.cases),
        index_name  = "faiss_cases",
        cfg         = cfg,
        model       = model,
        index_dir   = INDEX_DIR,
        max_records = args.max_cases,
        device      = device,
    )

    # ── Build statute index ──────────────────────────────────────────────────
    statute_stats = build_and_save_index(
        chunk_file  = Path(args.statutes),
        index_name  = "faiss_statutes",
        cfg         = cfg,
        model       = model,
        index_dir   = INDEX_DIR,
        max_records = None,       # always index ALL statutes (small file)
        device      = device,
    )

    # ── Save config for downstream use ───────────────────────────────────────
    config = {
        "model_name"  : cfg["model_name"],
        "embed_dim"   : cfg["embed_dim"],
        "doc_prefix"  : cfg["doc_prefix"],
        "query_prefix": cfg["query_prefix"],
        "normalize"   : cfg["normalize"],
        "indexes"     : {
            "cases"   : "faiss_cases.index",
            "statutes": "faiss_statutes.index",
        },
        "stats": {
            "cases"   : case_stats,
            "statutes": statute_stats,
        }
    }
    config_path = INDEX_DIR / "index_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\n✅ Config saved: {config_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INDEXING COMPLETE")
    print(f"{'='*60}")
    if case_stats:
        print(f"  Case index   : {case_stats['vectors']:>8,} vectors  ({case_stats['elapsed_min']} min)")
    if statute_stats:
        print(f"  Statute index: {statute_stats['vectors']:>8,} vectors  ({statute_stats['elapsed_min']} min)")
    print(f"\nNEXT: Run 05_test_retrieval.py to verify results")

if __name__ == "__main__":
    main()