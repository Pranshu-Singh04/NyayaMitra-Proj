# NyayaMitra: Grounded RAG for Indian Criminal Law

NyayaMitra is a Retrieval-Augmented Generation (RAG) system for Indian criminal law, built on top of **INLegalLlama** as the base LLM. It combines dense FAISS retrieval (E5-large-v2), BM25 sparse retrieval, cross-encoder reranking, and DeBERTa-based NLI hallucination checking to deliver legally grounded answers — targeting performance above the NyayaAnumana benchmark.

---

## Results (April 2026)

| Metric | Baseline (no RAG) | NyayaAnumana | **NyayaMitra** |
|---|---|---|---|
| Grounding rate | 30.9% | ~60% | **81.6%** |
| Hallucination rate | 6.1% | ~4% | **1.3%** |
| LJP Accuracy | — | ~72% | in progress |

Evaluated on Indian criminal law queries (bail, FIR, arrest, chargesheet, sentencing) using Gemini 2.5 Flash as the generation model.

---

## System Architecture

```
Query
  │
  ▼
Intent Classifier (QueryType: BAIL / FIR / SENTENCING / GENERAL)
  │
  ▼
Hybrid Retriever
  ├── Dense: FAISS + E5-large-v2 (100k cases + 1,040 statutes)
  ├── Sparse: BM25 (statutes only)
  └── Reranker: ms-marco-MiniLM-L-6-v2 cross-encoder
  │
  ▼
Prompt Builder (citation-aware, statute-first prompting)
  │
  ▼
LLM (INLegalLlama / Gemini 2.5 Flash / Groq llama-3.3-70b / GPT-4o-mini)
  │
  ▼
Hallucination Checker (DeBERTa NLI — claim-by-claim verification)
  │
  ▼
Grounded Legal Answer + Source Citations
```

---

## Repository Structure

```
scripts/
├── 01_explore_dataset.py         # Explore NyayaAnumana dataset structure
├── 02_chunk_cases.py             # Chunk case text into retrieval units
├── 03_chunk_statutes.py          # Parse and chunk IPC / BNS / BNSS statutes
├── 04_embed_and_index.py         # Embed chunks → FAISS index (E5-large-v2)
├── 05_test_retrieval.py          # Sanity-check retrieval quality
├── hybrid_retriever.py           # Core: FAISS + BM25 + RRF + cross-encoder
├── prompt_builder.py             # Intent classifier + citation-aware prompts
├── llm_integration.py            # LLM backends: Gemini, GPT, Groq, INLegalLlama
├── hallucination_checker.py      # DeBERTa NLI claim-level hallucination checker
├── rag_pipeline.py               # Full end-to-end pipeline orchestration
├── 10_test_pipeline.py           # Interactive pipeline test (single queries)
├── 12_rag_with_hallucination.py  # Single-query RAG + hallucination check
├── 12_batch_hallucination_eval.py# Batch evaluation (N queries, CSV output)
├── 13_evaluate_hallucination_v2.py # Full grounding/hallucination evaluation suite
├── 14_generate_graphs.py         # Generate paper figures from eval JSON
├── 15_evaluate_ljp_accuracy.py   # Legal Judgment Prediction accuracy vs NyayaAnumana
└── 16_ingest_bnss.py             # Ingest BNSS 2023 sections into FAISS index
instructions.md                   # Detailed usage guide for all scripts
requirements.txt                  # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/Pranshu-Singh04/NyayaMitra-Proj.git
cd NyayaMitra-Proj
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt

# PyTorch with CUDA (recommended):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Get the data and build the index

The NyayaAnumana dataset and FAISS indexes are **not included** in this repo (dataset restrictions + file size). To reproduce:

1. Download the **NyayaAnumana** dataset from the original paper's release
2. Run the pipeline scripts in order (01 → 02 → 03 → 04):

```bash
python -u scripts/01_explore_dataset.py --root "path/to/NyayaAnumana"
python -u scripts/02_chunk_cases.py --root "path/to/NyayaAnumana" --output data/processed/chunks
python -u scripts/03_chunk_statutes.py --output data/processed/chunks
python -u scripts/04_embed_and_index.py --cases data/processed/chunks/cases_train_chunks.jsonl \
    --statutes data/processed/chunks/statute_chunks.jsonl --output indexes/
```

> Always use `python -u` to see live output on Windows.

### 3. Run a single query

```bash
cd scripts
python -u 10_test_pipeline.py \
    --index_dir ../indexes \
    --model gemini \
    --api_key YOUR_GEMINI_API_KEY \
    --query "Can a person accused of murder get anticipatory bail?"
```

### 4. Run the full evaluation

```bash
python -u scripts/13_evaluate_hallucination_v2.py \
    --index_dir indexes/ \
    --model gemini \
    --api_key YOUR_GEMINI_API_KEY \
    --n_queries 15
```

---

## LLM Backends

| Model | Flag | Cost | Get Key |
|---|---|---|---|
| Gemini 2.5 Flash | `--model gemini` | Free tier | [aistudio.google.com](https://aistudio.google.com) |
| Groq llama-3.3-70b | `--model groq` | Free tier | [console.groq.com](https://console.groq.com) |
| GPT-4o-mini | `--model gpt` | Paid | [platform.openai.com](https://platform.openai.com) |
| INLegalLlama | `--model inlegalllama` | Free (Colab) | Self-hosted via ngrok |

---

## Google Colab Workflow

Large indexes (FAISS, ~8GB) live on Google Drive. Scripts run in Colab, code is pulled from this repo:

```python
import os

# Mount Drive (indexes live here)
from google.colab import drive
drive.mount('/content/drive')

# Pull latest code from GitHub
REPO = "https://github.com/Pranshu-Singh04/NyayaMitra-Proj.git"
if not os.path.exists("/content/NyayaMitra"):
    os.system(f"git clone {REPO} /content/NyayaMitra")
else:
    os.system("cd /content/NyayaMitra && git pull")

os.chdir("/content/NyayaMitra/scripts")
```

---

## Citation

> Paper under preparation. Citation will be added on publication.

---

## License

Code: MIT License.
Dataset: Subject to NyayaAnumana dataset terms — not redistributed here.
