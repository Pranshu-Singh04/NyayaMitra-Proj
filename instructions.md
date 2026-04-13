# NyayaMitra — Full Reference Guide

> Run all commands from the project root with the virtual environment active.

```powershell
cd C:\Users\Pranshu\Documents\AI-Legal-Advisor
& venv\Scripts\Activate.ps1
```

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [API Keys](#2-api-keys)
3. [Script Reference (01–16)](#3-script-reference)
4. [Typical Full Run Order](#4-typical-full-run-order)
5. [Google Colab Setup](#5-google-colab-setup)
6. [Results & Baselines](#6-results--baselines)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Environment Setup

### First time (local)
```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install faiss-cpu
```

### Every new terminal session
```powershell
cd C:\Users\Pranshu\Documents\AI-Legal-Advisor
& venv\Scripts\Activate.ps1
```

> **Recommended:** Run on Google Colab (see Section 5). The local machine needs
> a larger paging file (sysdm.cpl → Advanced → Performance → Virtual Memory →
> Custom: 8192 initial / 16384 max) to avoid RAM freezes.

---

## 2. API Keys

Never hardcode keys in files. Pass them via `--api-key` / `--api_key` or set an env var.

### Gemini (Google) — FREE
- **Model:** `gemini-2.5-flash`
- **Get key:** https://aistudio.google.com
- **Env var:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- **Usage:** `--model gemini --api-key YOUR_KEY`

### Groq (Llama 3.3 70B) — FREE
- **Model:** `llama-3.3-70b-versatile`
- **Get key:** https://console.groq.com (starts with `gsk_`)
- **Env var:** `GROQ_API_KEY`
- **Usage:** `--model groq --api-key YOUR_KEY`

### INLegalLlama — FREE (via Colab)
- Open `NyayaMitra_LLM_Colab.ipynb` → Run all → copy ngrok URL
- **Usage:** `--model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate`

### OpenAI GPT-4o-mini — PAID
- **Get key:** https://platform.openai.com (starts with `sk-`)
- **Env var:** `OPENAI_API_KEY`
- **Usage:** `--model gpt --api-key YOUR_KEY`

### Model name quick-reference

| `--model` value | LLM | Cost |
|----------------|-----|------|
| `gemini` | gemini-2.5-flash | Free |
| `groq` / `llama` | llama-3.3-70b-versatile | Free |
| `inlegalllama` | INLegalLlama-7B (Colab) | Free |
| `gpt` / `openai` | gpt-4o-mini | Paid |

---

## 3. Script Reference

---

### `01_explore_dataset.py` — Dataset statistics

Prints document counts, label distributions, text length stats for raw data in `data/`.

```powershell
python -u scripts/01_explore_dataset.py
```

---

### `02_chunk_cases.py` — Chunk case judgments

Splits raw case judgment files into overlapping text chunks. Output: `data/chunks_cases.jsonl`. Run once.

```powershell
python -u scripts/02_chunk_cases.py
```

---

### `03_chunk_statutes.py` — Chunk statute files

Splits IPC, BNS, CrPC, BNSS statute files into section-level chunks. Output: `data/chunks_statutes.jsonl`. Run once.

```powershell
python -u scripts/03_chunk_statutes.py
```

---

### `04_embed_and_index.py` — Build FAISS indexes

Encodes all chunks with E5-large-v2 embeddings and writes FAISS indexes + metadata to `indexes/`. Takes ~5 min on GPU, ~30 min on CPU. Run once.

```powershell
python -u scripts/04_embed_and_index.py
```

Output: `indexes/faiss_cases.index`, `indexes/faiss_statutes.index`, metadata JSONL files, `index_config.json`.

---

### `05_test_retrieval.py` — Test FAISS retrieval

Runs a query directly against the FAISS indexes to verify they were built correctly.

```powershell
python -u scripts/05_test_retrieval.py
python -u scripts/05_test_retrieval.py --query "bail for murder Section 302" --top_k 5
python -u scripts/05_test_retrieval.py --query "BNSS 2023 anticipatory bail Section 482"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--query` | default test | Search query |
| `--top_k` | 5 | Number of results |
| `--index_dir` | `indexes` | Path to indexes/ |

---

### `hybrid_retriever.py` — Hybrid BM25 + FAISS retrieval

**What it does:**
- Dense FAISS (E5-large-v2) + sparse BM25 with Reciprocal Rank Fusion (RRF)
- BM25 runs on statutes only (1,040 docs, fast). Cases use dense-only (100k docs would freeze RAM).
- Adaptive RRF weights: statute queries = BM25-heavy, others = dense-heavy
- Query expansion with legal synonyms (IPC, BNS, BNSS 2023 terms)
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`, CPU)
- Metadata boosts: Supreme Court ×1.25, recent cases ×1.15, exact section ×1.50
- MMR diversity: LJP λ=0.85, statute λ=0.95, QA λ=0.72
- VRAM pre-check: uses CUDA only if ≥1.8GB free, else CPU
- CUDA fallback: if GPU encode fails mid-run, moves to CPU automatically

```powershell
python -u scripts/hybrid_retriever.py --query "bail for murder Section 302"
python -u scripts/hybrid_retriever.py --query "BNSS 2023 anticipatory bail" --top_k 5
python -u scripts/hybrid_retriever.py --query "Section 302 IPC" --no-reranker
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--query` | required | Search query |
| `--top_k` | 3 | Number of results |
| `--index_dir` | `indexes` | Path to indexes/ |
| `--no-reranker` | false | Skip cross-encoder reranking |

---

### `prompt_builder.py` — Prompt construction

**What it does:**
- Builds structured prompts for: legal QA, LJP (4-step CoT), statute lookup, summarise
- Per-model context budgets: Gemini 32k chars, GPT 12k, INLegalLlama 6k
- LJP prompt: 4-step chain-of-thought → charge ID → precedents → distinguishing factors → prediction
- Few-shot LJP examples prepended for Gemini/GPT (2 worked examples)
- Intent classifier: STRONG signals (1 hit = LJP) + WEAK signals (3 hits = LJP)

```powershell
python -u scripts/prompt_builder.py
```

---

### `llm_integration.py` — LLM backends

**What it does:**
- Wraps Gemini, Groq, GPT, INLegalLlama behind common `BaseLLM` interface
- Task-specific temperature: LJP/statute=0.1, QA=0.3, summarise=0.4
- Exponential backoff retry on 429/500/503 (4 attempts: 2s, 4s, 8s, 16s)
- INLegalLlama: supports Colab ngrok endpoint or local HuggingFace load

```powershell
python -u scripts/llm_integration.py --model gemini --api_key YOUR_KEY --query "What is Section 302 IPC?"
python -u scripts/llm_integration.py --model groq   --api_key YOUR_KEY --query "What is Section 302 IPC?"
python -u scripts/llm_integration.py --model gpt    --api_key YOUR_KEY --query "What is Section 302 IPC?"
python -u scripts/llm_integration.py --model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate --query "What is Section 302 IPC?"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini` | `gemini` / `groq` / `gpt` / `inlegalllama` |
| `--api_key` | None | API key |
| `--colab_url` | None | ngrok URL for INLegalLlama |
| `--query` | default | Question to ask |

---

### `hallucination_checker.py` — NLI-based hallucination detection

**What it does:**
- Splits LLM answers into individual factual claims
- Scores each claim against retrieved chunks using DeBERTa NLI (`nli-deberta-v3-small`, CPU)
- Batch NLI inference (32 pairs), sliding window (8 windows/chunk, up to 10 chunks)
- Verbatim match shortcut: >52% word overlap → SUPPORTED without NLI
- Thresholds: SUPPORTED if P(entailment) > 0.35, UNSUPPORTED if P(contradiction) > 0.38
- Coverage gap detection: low grounding + low hallucination = retrieval gap, not error
- Graceful degradation: if NLI model can't load (low RAM), returns NEUTRAL (doesn't crash)

```powershell
python -u scripts/hallucination_checker.py
```

---

### `rag_pipeline.py` — Full end-to-end RAG pipeline

**What it does:**
- Chains retrieval → prompt → LLM → parse into single `pipeline.query()` call
- Auto-detects query type or accepts forced `query_type`
- 3-priority LJP parser: explicit label → line scan → frequency vote
- Confidence-based abstention: LOW confidence → prediction withheld
- `VerifiedNyayaMitraPipeline`: same + NLI hallucination check per response

```powershell
python -u scripts/rag_pipeline.py --model gemini --api_key YOUR_KEY --query "Can I get bail for murder?"
python -u scripts/rag_pipeline.py --model gemini --api_key YOUR_KEY --mode ljp    --query "Accused under Section 302. First offence."
python -u scripts/rag_pipeline.py --model gemini --api_key YOUR_KEY --mode statute --query "Explain Section 498A IPC"
python -u scripts/rag_pipeline.py --model gemini --api_key YOUR_KEY --mode verified --query "Can I get bail for murder?"
python -u scripts/rag_pipeline.py --model groq   --api_key YOUR_KEY --query "What is Section 498A?"
python -u scripts/rag_pipeline.py --model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate --query "Can I get bail?"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini` | LLM backend |
| `--api_key` | None | API key |
| `--colab_url` | None | ngrok URL |
| `--query` | required | Legal question |
| `--mode` | `auto` | `auto` / `ljp` / `qa` / `statute` / `verified` |
| `--top_k` | 8 | Chunks to retrieve |
| `--index_dir` | `indexes` | Path to indexes/ |

---

### `10_test_pipeline.py` — End-to-end test suite (15 queries)

Runs 15 fixed test queries (5 legal QA, 5 LJP, 5 statute) through the full pipeline. Reports latency and success rate per query type. Results saved to `results/`.

```powershell
python -u scripts/10_test_pipeline.py --model gemini      --mode paper_eval --api_key YOUR_KEY
python -u scripts/10_test_pipeline.py --model groq        --mode paper_eval --api_key YOUR_KEY
python -u scripts/10_test_pipeline.py --model inlegalllama --mode paper_eval --colab_url https://XXXX.ngrok-free.app/generate
python -u scripts/10_test_pipeline.py --model gemini      --mode basic      --api_key YOUR_KEY
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini` | LLM backend |
| `--api_key` | None | API key |
| `--colab_url` | None | ngrok URL |
| `--mode` | `paper_eval` | `basic` (1 query) / `paper_eval` (15 queries) |
| `--index_dir` | `indexes` | Path to indexes/ |

Output: `results/eval_MODEL_TIMESTAMP.json`

---

### `12_batch_hallucination_eval.py` — Batch grounding eval (CSV output)

**What it does:**
- Runs N questions through full pipeline + NLI hallucination checker
- Adaptive rate limiting: Gemini 4s/query, GPT 1s, Groq/local 0.2s
- Retry on 429/timeout: 4 attempts with 2s/4s/8s/16s backoff in LLM + 30s/60s/90s in runner
- Timeout protection on NLI checker (60s max per query)
- Full error logging with traceback and per-error-type breakdown
- Graceful fallback on empty retrieval or empty LLM response

```powershell
python -u scripts/12_batch_hallucination_eval.py --n 10 --model gemini --api-key YOUR_KEY
python -u scripts/12_batch_hallucination_eval.py --n 10 --model groq   --api-key YOUR_KEY
python -u scripts/12_batch_hallucination_eval.py --n 50 --model gemini --api-key YOUR_KEY --delay 6.0
python -u scripts/12_batch_hallucination_eval.py --n 10 --model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate
python -u scripts/12_batch_hallucination_eval.py --n 20 --questions eval/questions.json --model gemini --api-key YOUR_KEY
python -u scripts/12_batch_hallucination_eval.py --n 10 --model gemini --api-key YOUR_KEY --out eval/my_run.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--n` | 10 | Number of questions |
| `--model` | `gemini` | LLM backend |
| `--api-key` | None | API key |
| `--colab_url` | None | ngrok URL |
| `--questions` | None | Path to JSON question list |
| `--index-dir` | `indexes` | Path to indexes/ |
| `--top-k` | 5 | Chunks per query |
| `--out` | `eval/hallucination_batch.csv` | Output CSV |
| `--delay` | auto | Override inter-query delay in seconds |

Output: `eval/hallucination_batch.csv` + `eval/hallucination_batch_summary.json`

Output columns: `question`, `query_type_detected`, `grounding`, `hallucination`, `supported`, `neutral`, `contradicted`, `total_claims`, `coverage_gap_warning`, `case_chunks`, `statute_chunks`, `top_case_score`, `top_statute_score`, `num_citations_in_answer`, `answer_word_count`, `verbatim_match_count`, `latency_ms`, `answer_snippet`

---

### `13_evaluate_hallucination_v2.py` — Full paper hallucination evaluation

**What it does:**
- Runs 15 evaluation queries (5 legal QA, 5 LJP, 5 statute)
- Measures grounding + hallucination per query and per type
- RAG vs no-RAG ablation (10 queries)
- Compares against NyayaAnumana paper baselines
- Generates LaTeX tables ready for paper submission

```powershell
python -u scripts/13_evaluate_hallucination_v2.py --model gemini --api_key YOUR_KEY --mode eval
python -u scripts/13_evaluate_hallucination_v2.py --model groq   --api_key YOUR_KEY --mode eval
python -u scripts/13_evaluate_hallucination_v2.py --model gemini --api_key YOUR_KEY --mode rag_vs_no_rag
python -u scripts/13_evaluate_hallucination_v2.py --model gemini --api_key YOUR_KEY --mode full
python -u scripts/13_evaluate_hallucination_v2.py --model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate --mode full
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini` | LLM backend |
| `--api_key` | None | API key |
| `--colab_url` | None | ngrok URL |
| `--mode` | `eval` | `eval` / `rag_vs_no_rag` / `full` |
| `--index_dir` | `indexes` | Path to indexes/ |
| `--output_dir` | `results` | Output folder |

Output files in `results/`:

| File | Description |
|------|-------------|
| `hallucination_eval_MODEL_TIMESTAMP.json` | Per-query grounding + hallucination |
| `rag_vs_no_rag_MODEL_TIMESTAMP.json` | RAG vs no-RAG ablation |
| `graph_data_MODEL_TIMESTAMP.json` | Pre-formatted data for plotting |
| `ablation_graph_data_MODEL_TIMESTAMP.json` | Ablation data for plotting |
| `hallucination_table.tex` | LaTeX table (grounding/hallucination by type) |
| `nyayaanumana_comparison_MODEL_TIMESTAMP.json` | vs NyayaAnumana baselines |
| `nyayaanumana_comparison_table.tex` | LaTeX comparison table |

---

### `14_generate_graphs.py` — Generate all paper figures

Reads JSON output from `13_evaluate_hallucination_v2.py` and generates 8 publication-quality PNG figures.

```powershell
python -u scripts/14_generate_graphs.py --results_dir results

python -u scripts/14_generate_graphs.py \
  --eval_file     results/hallucination_eval_gemini_TIMESTAMP.json \
  --ablation_file results/ablation_graph_data_gemini_TIMESTAMP.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | `results` | Auto-detect latest JSON files |
| `--eval_file` | None | Specific eval JSON |
| `--ablation_file` | None | Specific ablation JSON |

Output in `results/figures/`:

| File | Description |
|------|-------------|
| `fig1_per_query_grounding.png` | Grounding per query, coloured by type |
| `fig2_avg_by_type.png` | Avg grounding vs hallucination by type |
| `fig3_claim_distribution.png` | Supported / neutral / unsupported counts |
| `fig4_latency.png` | LLM vs NLI latency breakdown |
| `fig5_entailment_distribution.png` | Entailment score KDE |
| `fig6_rag_vs_no_rag_per_query.png` | RAG vs no-RAG per query |
| `fig7_rag_vs_no_rag_aggregate.png` | RAG vs no-RAG aggregate (main figure) |
| `fig8_claim_counts.png` | Claim count by query type |

---

### `15_evaluate_ljp_accuracy.py` — LJP accuracy on NyayaAnumana dataset

**What it does:**
- Loads test cases from `Exploration-Lab/NyayaAnumana` on HuggingFace (streaming)
- Falls back to local CSV in `data/raw/NyayaAnumana_Sample_Subset/` if offline
- Runs each case through the LJP pipeline, compares against ground-truth label
- Reports two accuracy numbers: all predictions + HIGH/MEDIUM confidence only
- Saves confusion matrix, per-class P/R/F1, per-row results

```powershell
python -u scripts/15_evaluate_ljp_accuracy.py --model gemini --api_key YOUR_KEY --n 50
python -u scripts/15_evaluate_ljp_accuracy.py --model gemini --api_key YOUR_KEY --n 500
python -u scripts/15_evaluate_ljp_accuracy.py --model groq   --api_key YOUR_KEY --n 500
python -u scripts/15_evaluate_ljp_accuracy.py --model gemini --api_key YOUR_KEY --n 200 --split ternary
python -u scripts/15_evaluate_ljp_accuracy.py --model inlegalllama --colab_url https://XXXX.ngrok-free.app/generate --n 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gemini` | LLM backend |
| `--api_key` | None | API key |
| `--colab_url` | None | ngrok URL |
| `--n` | 500 | Number of cases to evaluate |
| `--split` | `binary` | `binary` (ALLOWED/DISMISSED) / `ternary` (adds PARTIALLY ALLOWED) |
| `--index_dir` | `indexes` | Path to indexes/ |
| `--output_dir` | `eval` | Output folder |
| `--seed` | 42 | Random seed |

Output: `eval/ljp_accuracy_MODEL_SPLIT_TIMESTAMP.json`

---

### `16_ingest_bnss.py` — Add BNSS 2023 to the statute index

**What it does:**
- Ingests BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita) into the existing FAISS statute index
- Three modes: hardcoded critical sections (no PDF), PDF parsing, plain text parsing
- Deduplicates automatically (safe to re-run)
- Updates both the FAISS index and metadata JSONL

Hardcoded sections (11 key sections): S.482 anticipatory bail, S.480 bail bailable, S.481 bail non-bailable, S.483 High Court bail, S.173 FIR, S.35 arrest without warrant, S.187 remand, S.528 inherent powers, S.43 rights on arrest, S.193 chargesheet, S.358 legal aid.

```powershell
# No PDF needed — uses 11 hardcoded critical sections
python -u scripts/16_ingest_bnss.py --hardcoded --index_dir indexes/

# From PDF (pip install PyPDF2 first)
python -u scripts/16_ingest_bnss.py --pdf data/bnss_2023.pdf --index_dir indexes/

# From plain text
python -u scripts/16_ingest_bnss.py --text data/bnss_2023.txt --index_dir indexes/

# Verify it worked
python -u scripts/05_test_retrieval.py --query "bail conditions BNSS 2023 Section 482"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--hardcoded` | false | Use 11 hardcoded critical BNSS sections |
| `--pdf` | None | Path to BNSS 2023 PDF |
| `--text` | None | Path to BNSS 2023 plain text |
| `--index_dir` | `indexes` | Path to indexes/ |

---

## 4. Typical Full Run Order

```powershell
# Always use -u flag so output appears immediately
# 0. Activate env
& venv\Scripts\Activate.ps1

# 1. Build indexes (first time only)
python -u scripts/04_embed_and_index.py

# 2. Ingest BNSS 2023 (first time only)
python -u scripts/16_ingest_bnss.py --hardcoded --index_dir indexes/

# 3. Test retrieval
python -u scripts/hybrid_retriever.py --query "bail murder Section 302"

# 4. Single query test
python -u scripts/rag_pipeline.py --model gemini --api_key YOUR_KEY --query "Can I get bail for murder?"

# 5. Batch eval (10 questions, CSV output)
python -u scripts/12_batch_hallucination_eval.py --n 10 --model gemini --api-key YOUR_KEY

# 6. Full paper eval (15 queries, LaTeX tables)
python -u scripts/13_evaluate_hallucination_v2.py --model gemini --api_key YOUR_KEY --mode full

# 7. LJP accuracy on 500 NyayaAnumana rows
python -u scripts/15_evaluate_ljp_accuracy.py --model gemini --api_key YOUR_KEY --n 500

# 8. Generate paper figures
python -u scripts/14_generate_graphs.py --results_dir results
```

---

## 5. Google Colab Setup

**Recommended** — avoids Windows RAM/paging file issues.

### One-time: Upload indexes to Drive
Upload `AI-Legal-Advisor/indexes/` to `My Drive/NyayaMitra/indexes/` via https://drive.google.com

### Reusable Colab notebook

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone or pull latest from GitHub
import os
REPO    = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
WORKDIR = "/content/NyayaMitra"

if not os.path.exists(WORKDIR):
    !git clone {REPO} {WORKDIR}
else:
    !cd {WORKDIR} && git pull

# Symlink indexes and output dirs from Drive
!mkdir -p /content/drive/MyDrive/NyayaMitra/results
!mkdir -p /content/drive/MyDrive/NyayaMitra/eval
!ln -sf /content/drive/MyDrive/NyayaMitra/indexes {WORKDIR}/indexes
!ln -sf /content/drive/MyDrive/NyayaMitra/results {WORKDIR}/results
!ln -sf /content/drive/MyDrive/NyayaMitra/eval    {WORKDIR}/eval

os.chdir(WORKDIR)
print("Ready:", os.listdir("."))

# Cell 3 — Install dependencies
!pip install sentence-transformers faiss-cpu transformers torch google-genai openai datasets -q

# Cell 4 — Run scripts (change this cell each time)
!python -u scripts/13_evaluate_hallucination_v2.py \
    --model gemini \
    --api_key YOUR_KEY \
    --mode eval
```

### Daily workflow
1. Edit in VSCode
2. `git push` from VSCode terminal
3. In Colab: run Cell 2 (git pull) → run script cell

---

## 6. Results & Baselines

### April 2026 evaluation results (gemini-2.5-flash, 15 queries)

| Query Type | Grounding | Hallucination | Avg Claims |
|-----------|-----------|---------------|------------|
| Legal Q&A | 71.4% | 3.8% | 5.8 |
| LJP | 93.3% | 0.0% | 1.6 |
| Statute Lookup | 80.0% | 0.0% | 1.0 |
| **Overall** | **81.6%** | **1.3%** | **2.8** |

### Pre-improvement baseline (March 2026)

| Metric | Overall | Legal Q&A | LJP | Statute |
|--------|---------|-----------|-----|---------|
| Grounding | 30.9% | 33.9% | 25.5% | 33.3% |
| Hallucination | 6.1% | 6.8% | 6.7% | 4.7% |

### NyayaAnumana paper (target to beat)

| Metric | Value |
|--------|-------|
| LJP Accuracy | 78% |
| ROUGE-L | 0.38 |
| Grounding | N/A (no RAG) |
| Hallucination | N/A (not measured) |

---

## 7. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| Script hangs, no output | Windows stdout buffering | Always use `python -u scripts/...` |
| Process frozen, Ctrl+C doesn't work | Windows RAM exhausted | Run on Colab or increase paging file |
| `OSError 1455` / `paging file too small` | Virtual memory full | sysdm.cpl → Advanced → Performance → Virtual Memory → Custom 8192/16384 |
| `MemoryError` during BM25 | 100k docs too large | Fixed — BM25 now statutes-only (auto) |
| NLI model fails to load | Low RAM | Checker degrades gracefully to NEUTRAL |
| CUDA encode fails / `unknown error` | VRAM full after other models load | Fixed — auto falls back to CPU mid-run |
| `429 RESOURCE_EXHAUSTED` (Gemini) | Rate limit | Retry with backoff built in; use `--delay 6` or switch to `--model groq` |
| `insufficient_quota` (OpenAI) | No free tier | Use `--model groq` or `--model gemini` |
| INLegalLlama 0/15 passed | Colab response field mismatch | Debug line prints actual JSON keys — share output |
| ngrok URL not working | Colab session expired | Re-run Colab notebook, copy new URL |

---

## 8. Quick Script Summary

| Script | What it does | Run standalone |
|--------|-------------|---------------|
| `01_explore_dataset.py` | Dataset statistics | Yes |
| `02_chunk_cases.py` | Chunk case files | Yes (run once) |
| `03_chunk_statutes.py` | Chunk statute files | Yes (run once) |
| `04_embed_and_index.py` | Build FAISS indexes | Yes (run once) |
| `05_test_retrieval.py` | Test retrieval | Yes |
| `hybrid_retriever.py` | BM25 + FAISS + reranker | Yes (CLI) |
| `prompt_builder.py` | Build LLM prompts | Yes (test) |
| `llm_integration.py` | Gemini / Groq / GPT / Llama | Yes (CLI) |
| `hallucination_checker.py` | NLI hallucination detection | Yes (diagnostic) |
| `rag_pipeline.py` | Full RAG pipeline | Yes (CLI) |
| `10_test_pipeline.py` | 15-query test suite | Yes |
| `12_batch_hallucination_eval.py` | N-question batch eval → CSV | Yes |
| `13_evaluate_hallucination_v2.py` | Full paper eval → JSON + LaTeX | Yes |
| `14_generate_graphs.py` | 8 paper figures | Yes |
| `15_evaluate_ljp_accuracy.py` | LJP accuracy on NyayaAnumana | Yes |
| `16_ingest_bnss.py` | Add BNSS 2023 to statute index | Yes |
