"""
12_batch_hallucination_eval.py
==============================
Batch grounding + hallucination eval over real NyayaMitra RAG outputs.

FIX: Bypasses rag_pipeline.py (which hardcodes numbered filenames like
06_hybrid_retriever.py) and wires components directly from your actual files.

Usage:
    python scripts/12_batch_hallucination_eval.py --n 10
    python scripts/12_batch_hallucination_eval.py --n 50 --questions eval/questions.json
    python scripts/12_batch_hallucination_eval.py --n 50 --model gemini --api-key YOUR_KEY
    python scripts/12_batch_hallucination_eval.py --n 50 --model gpt    --api-key sk-...
    python scripts/12_batch_hallucination_eval.py --n 50 --model groq   --api-key gsk_...
"""

import argparse, json, csv, os, sys, time, threading, traceback, io
from pathlib import Path
from collections import Counter, defaultdict

# Fix Windows cp1252 encoding crashes on Unicode output (emojis in NLI summary)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

# ── make scripts/ importable ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── direct imports from your actual (unnumbered) files ───────────────────────
from hallucination_checker import HallucinationCheckerV2            # 11
from hybrid_retriever      import HybridRetrieverV2                  # 06
from prompt_builder        import PromptBuilderV2, IntentClassifier, QueryType  # 07
from llm_integration       import get_llm                            # 08

# ── default questions (used when no --questions file supplied) ────────────────
DEFAULT_QUESTIONS = [
    "What is Section 302 IPC and what is its punishment?",
    "What is the difference between murder and culpable homicide under IPC?",
    "What are the bail conditions under BNSS 2023?",
    "Under what conditions can anticipatory bail be granted?",
    "What is Section 376 IPC and what punishment does it prescribe?",
    "What is Section 420 IPC?",
    "Define cognizable offence and give examples.",
    "What are the rights of an arrested person under Article 22?",
    "What constitutes defamation under IPC Section 499?",
    "What is the punishment for theft under Section 379 IPC?",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _flatten_chunks(retrieved: dict) -> list[dict]:
    """
    HybridRetrieverV2.retrieve() returns {"cases": [...], "statutes": [...]}.
    Flatten to a list of {"text": ..., "source": ...} dicts for the checker.
    """
    chunks = []
    for c in retrieved.get("cases", []):
        text = c.get("text") or c.get("full_text") or c.get("section_text") or ""
        if text.strip():
            chunks.append({"text": text, "source": "case",
                           "id": c.get("case_id", c.get("id", ""))})
    for s in retrieved.get("statutes", []):
        text = s.get("text") or s.get("section_text") or s.get("full_text") or ""
        if text.strip():
            chunks.append({"text": text, "source": "statute",
                           "id": s.get("section", s.get("id", ""))})
    return chunks


def _safe_max_score(chunks, field="score"):
    """Safely extract the max numeric score from a list of chunk dicts."""
    scores = []
    for c in chunks:
        v = c.get(field, None)
        try:
            scores.append(float(v))
        except (TypeError, ValueError):
            pass
    return round(max(scores), 4) if scores else 0.0


def _check_with_timeout(checker, answer, chunks, timeout=60):
    """Run the NLI hallucination checker with a timeout guard."""
    result = [None]
    error  = [None]

    def _run():
        try:
            result[0] = checker.check(answer, chunks)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(
            f"Hallucination checker timed out after {timeout}s"
        )
    if error[0]:
        raise error[0]
    return result[0]


def _build_and_generate(
    query     : str,
    retrieved : dict,
    builder   : PromptBuilderV2,
    classifier: IntentClassifier,
    llm,
    max_retries: int = 3,
) -> str:
    """
    Replicate NyayaMitraPipeline.query() without the broken _load chain.
    Returns the answer string. Retries on rate-limit errors with backoff.
    """
    qt       = classifier.classify(query)
    cases    = retrieved.get("cases",    [])
    statutes = retrieved.get("statutes", [])

    if qt == QueryType.STATUTE_LOOKUP:
        prompt = builder.build_statute_lookup(query, statutes)
    elif qt == QueryType.LJP:
        prompt = builder.build_ljp(query, cases, statutes, "binary")
    else:
        prompt = builder.build_legal_qa(query, cases, statutes)

    task_str = qt.value.replace("legal_", "")

    for attempt in range(1, max_retries + 1):
        response = llm.generate_with_task(
            prompt.system_prompt, prompt.user_prompt, task_type=task_str
        )
        if response.success:
            return response.text.strip()

        err = response.error or "unknown error"
        is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err or "rate" in err.lower()

        if is_rate_limit and attempt < max_retries:
            wait = 30 * attempt   # 30s, 60s, 90s
            print(f"    Rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait)
            continue

        # Non-rate-limit error or out of retries — raise with real message
        raise ValueError(f"LLM error: {err[:200]}")


# ── main batch runner ─────────────────────────────────────────────────────────
def run_batch(
    questions : list[str],
    n         : int,
    out_path  : str,
    model     : str   = "gemini",
    api_key   : str   = None,
    colab_url : str   = None,
    index_dir : str   = "indexes",
    top_k     : int   = 5,
    delay     : float = None,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ── init retriever ────────────────────────────────────────────────────────
    print(f"\nLoading HybridRetrieverV2 from '{index_dir}'...")
    retriever = HybridRetrieverV2(index_dir=index_dir)

    # ── init prompt builder + classifier ─────────────────────────────────────
    builder    = PromptBuilderV2(model_type=model)   # IMPROVEMENT 8: per-model budget
    classifier = IntentClassifier()

    # ── init LLM ─────────────────────────────────────────────────────────────
    resolved_key = (
        api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )
    print(f"Loading LLM: {model}...")
    llm = get_llm(model=model, api_key=resolved_key, colab_url=colab_url)

    # ── init hallucination checker (loads NLI model) ──────────────────────────
    print("Loading HallucinationCheckerV2 (NLI model)...")
    checker = HallucinationCheckerV2()
    if not checker.nli._loaded:
        print("  WARNING: NLI model not loaded — grounding/hallucination scores will be approximate.")
        print("  FIX: Close other programs or increase paging file (sysdm.cpl), then retry.\n")
    print("All models ready. Starting batch eval...\n")

    rows     = []
    q_subset = questions[:n]

    # Adaptive rate limiting — can be overridden with --delay
    if delay is not None:
        inter_query_delay = delay
    elif model == "gemini":
        inter_query_delay = 4.0   # Gemini free tier: ~15 RPM = 4s min gap
    elif model == "gpt":
        inter_query_delay = 1.0
    else:
        inter_query_delay = 0.2   # groq / local models need minimal delay

    for i, q in enumerate(q_subset):
        print(f"[{i+1}/{len(q_subset)}] {q}")
        try:
            # 1. Retrieve
            retrieved = retriever.retrieve(q, top_k=top_k)
            chunks    = _flatten_chunks(retrieved)

            # Guard: skip if no chunks retrieved (coverage gap)
            if not chunks:
                print(f"  WARNING: No chunks retrieved for: {q[:60]}")
                print(f"           Skipping — index may not cover this topic.")
                rows.append({
                    "question"     : q,
                    "grounding"    : 0.0,
                    "hallucination": 0.0,
                    "error_type"   : "EmptyRetrieval",
                    "error_msg"    : "No chunks retrieved — likely a coverage gap",
                    "supported"    : 0, "neutral": 0, "contradicted": 0,
                    "total_claims" : 0,
                })
                continue

            # 2. Detect intent
            qt = classifier.classify(q)

            # 3. Generate answer
            answer = _build_and_generate(q, retrieved, builder, classifier, llm)

            # Guard: skip if LLM returned empty
            if not answer:
                print(f"  WARNING: LLM returned empty response for: {q[:60]}")
                rows.append({
                    "question"     : q,
                    "grounding"    : 0.0,
                    "hallucination": 0.0,
                    "error_type"   : "EmptyLLMResponse",
                    "error_msg"    : "LLM returned empty or failed",
                })
                continue

            # 4. Hallucination check (with timeout protection)
            report = _check_with_timeout(checker, answer, chunks, timeout=60)

            # ── extended row fields ─────────────────────────────────────────
            case_chunks    = [c for c in chunks if c["source"] == "case"]
            statute_chunks = [c for c in chunks if c["source"] == "statute"]

            # Top retrieval scores (safe extraction)
            top_case_score    = _safe_max_score(retrieved.get("cases", []))
            top_statute_score = _safe_max_score(retrieved.get("statutes", []))

            # Citations in answer: [Case N] or [Statute N] patterns
            import re as _re
            num_citations = len(set(_re.findall(
                r'\[(?:Case|Statute)\s*\d+\]', answer, _re.IGNORECASE
            )))

            # Verbatim match count from NLI results
            verbatim_count = sum(1 for c in report.claims if getattr(c, "verbatim_match", False))

            row = {
                "question"             : q,
                "query_type_detected"  : qt.value,
                "grounding"            : round(report.grounding_score,    4),
                "hallucination"        : round(report.hallucination_score, 4),
                "supported"            : report.supported_count,
                "neutral"              : report.neutral_count,
                "contradicted"         : report.unsupported_count,
                "total_claims"         : report.total_claims,
                "coverage_gap_warning" : report.coverage_gap_detected,
                "case_chunks"          : len(case_chunks),
                "statute_chunks"       : len(statute_chunks),
                "top_case_score"       : top_case_score,
                "top_statute_score"    : top_statute_score,
                "num_citations_in_answer": num_citations,
                "answer_word_count"    : len(answer.split()),
                "verbatim_match_count" : verbatim_count,
                "latency_ms"           : round(report.latency_ms, 1),
                "answer_snippet"       : answer[:200].replace("\n", " "),
            }
            rows.append(row)

            # Print summary — use ASCII-safe version to avoid cp1252 crashes on Windows
            try:
                gap = " [COVERAGE GAP]" if report.coverage_gap_detected else ""
                summary_text = (
                    f"  -> G={report.grounding_score:.1%} "
                    f"H={report.hallucination_score:.1%} "
                    f"Claims: {report.supported_count}ok "
                    f"{report.neutral_count}neu "
                    f"{report.unsupported_count}bad "
                    f"/ {report.total_claims} "
                    f"({report.latency_ms:.0f}ms){gap}"
                )
                print(summary_text)

                # Show hallucinated claims immediately so you can spot patterns
                for r in report.claims:
                    if r.label == "UNSUPPORTED":
                        print(f"     X  {r.claim[:90]}")
            except UnicodeEncodeError:
                # Fallback: minimal ASCII output
                print(f"  -> G={row['grounding']:.1%} H={row['hallucination']:.1%}")

        except Exception as e:
            # TASK A1: Full error logging with traceback and error type
            exc_type = type(e).__name__
            exc_tb   = traceback.format_exc()
            # ASCII-safe print to avoid cp1252 crashes on Windows
            safe_msg = str(e)[:200].encode("ascii", errors="replace").decode("ascii")
            safe_tb  = exc_tb[:500].encode("ascii", errors="replace").decode("ascii")
            print(f"  ERROR TYPE : {exc_type}")
            print(f"  TRACEBACK  :\n{safe_tb}")
            rows.append({
                "question"     : q,
                "grounding"    : None,
                "hallucination": None,
                "error_type"   : exc_type,
                "error_msg"    : str(e)[:300],
                "error_tb"     : exc_tb[:500],
            })
            continue

        # Rate-limit guard: adaptive delay between queries
        if inter_query_delay and i < len(q_subset) - 1:
            time.sleep(inter_query_delay)

    # ── write CSV ─────────────────────────────────────────────────────────────
    valid  = [r for r in rows if r.get("grounding") is not None]
    errors = [r for r in rows if r.get("grounding") is None]

    # TASK A1: Per-error-type breakdown
    if errors:
        err_types = Counter(r.get("error_type", "unknown") for r in errors)
        print(f"\n{'='*58}")
        print("ERROR BREAKDOWN:")
        for etype, count in err_types.most_common():
            print(f"  {etype}: {count}")
            # Show first error message for each type
            sample = next(r for r in errors if r.get("error_type") == etype)
            safe = sample.get('error_msg', 'no message')[:120].encode("ascii", errors="replace").decode("ascii")
            print(f"    -> {safe}")
        print(f"{'='*58}")

    if not valid:
        print("\nWARNING: No valid results — check errors above.")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=valid[0].keys())
        writer.writeheader()
        writer.writerows(valid)

    # ── summary ───────────────────────────────────────────────────────────────
    avg_g  = sum(r["grounding"]     for r in valid) / len(valid)
    avg_h  = sum(r["hallucination"] for r in valid) / len(valid)
    gaps   = sum(1 for r in valid if r.get("coverage_gap_warning"))
    high_h = [r for r in valid if r["hallucination"] > 0.20]

    print(f"\n{'='*58}")
    print(f"BATCH EVAL SUMMARY  (n={len(valid)} valid | {len(errors)} errors)")
    print(f"  Model              : {model}")
    print(f"  Avg Grounding      : {avg_g:.1%}")
    print(f"  Avg Hallucination  : {avg_h:.1%}")
    print(f"  Coverage Gaps      : {gaps} ({gaps/len(valid):.0%})")
    print(f"  High Hallucination : {len(high_h)} queries > 20%")
    print(f"  Results CSV        : {out_path}")
    print(f"{'='*58}")

    if high_h:
        print("\nQueries with high hallucination (>20%):")
        for r in high_h:
            print(f"  H={r['hallucination']:.0%}  {r['question']}")

    # ── per-query-type breakdown ─────────────────────────────────────────────
    by_type: dict = defaultdict(list)
    for r in valid:
        by_type[r.get("query_type_detected", "unknown")].append(r)

    if len(by_type) > 1:
        print(f"\n{'─'*58}")
        print("PER QUERY-TYPE BREAKDOWN:")
        for qt_name, qt_rows in sorted(by_type.items()):
            g = sum(r["grounding"]    for r in qt_rows) / len(qt_rows)
            h = sum(r["hallucination"] for r in qt_rows) / len(qt_rows)
            c = sum(r["num_citations_in_answer"] for r in qt_rows) / len(qt_rows)
            v = sum(r["verbatim_match_count"] for r in qt_rows) / len(qt_rows)
            print(f"  {qt_name:20s}  n={len(qt_rows):3d}  "
                  f"G={g:.1%}  H={h:.1%}  "
                  f"citations/q={c:.1f}  verbatim/q={v:.1f}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary_path = out_path.replace(".csv", "_summary.json")
    Path(summary_path).write_text(
        json.dumps({
            "model"             : model,
            "n_valid"           : len(valid),
            "n_errors"          : len(errors),
            "avg_grounding"     : round(avg_g, 4),
            "avg_hallucination" : round(avg_h, 4),
            "coverage_gap_count": gaps,
            "high_hallucination_queries": [r["question"] for r in high_h],
            "error_breakdown"   : dict(Counter(r.get("error_type","unknown") for r in errors)) if errors else {},
            "per_query_type"    : {
                qt_name: {
                    "n"             : len(qt_rows),
                    "avg_grounding" : round(sum(r["grounding"] for r in qt_rows) / len(qt_rows), 4),
                    "avg_hallucination": round(sum(r["hallucination"] for r in qt_rows) / len(qt_rows), 4),
                    "avg_citations" : round(sum(r["num_citations_in_answer"] for r in qt_rows) / len(qt_rows), 2),
                }
                for qt_name, qt_rows in by_type.items()
            },
            "per_query"         : valid,
        }, indent=2),
        encoding="utf-8"
    )
    print(f"  Summary JSON       : {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NyayaMitra hallucination batch eval")
    ap.add_argument("--n",         type=int,  default=10,
                    help="Number of questions to run (default: 10)")
    ap.add_argument("--questions", type=str,  default=None,
                    help="Path to JSON file: list of question strings")
    ap.add_argument("--model",     type=str,  default="gemini",
                    choices=["gemini", "gpt", "groq", "inlegalllama"],
                    help="LLM backend (default: gemini)")
    ap.add_argument("--api-key",   type=str,  default=None,
                    help="API key (or set GOOGLE_API_KEY / OPENAI_API_KEY / GROQ_API_KEY env var)")
    ap.add_argument("--colab_url", type=str,  default=None,
                    help="ngrok URL for INLegalLlama on Colab")
    ap.add_argument("--index-dir", type=str,  default="indexes",
                    help="Path to indexes/ folder (default: indexes)")
    ap.add_argument("--top-k",     type=int,  default=5,
                    help="Chunks to retrieve per query (default: 5)")
    ap.add_argument("--out",       type=str,  default="eval/hallucination_batch.csv",
                    help="Output CSV path")
    ap.add_argument("--delay",     type=float, default=None,
                    help="Override inter-query delay in seconds (default: auto per model)")
    args = ap.parse_args()

    qs = DEFAULT_QUESTIONS
    if args.questions:
        p = Path(args.questions)
        if not p.exists():
            print(f"ERROR: questions file not found: {p}")
            print("  Make sure you're running from the project root:")
            print("  cd C:\\Users\\Pranshu\\Documents\\AI-Legal-Advisor")
            print("  python scripts/12_batch_hallucination_eval.py --questions eval/questions.json")
            sys.exit(1)
        qs = json.loads(p.read_text(encoding="utf-8"))
        print(f"Loaded {len(qs)} questions from {p}")

    run_batch(
        questions = qs,
        n         = args.n,
        out_path  = args.out,
        model     = args.model,
        api_key   = getattr(args, "api_key"),
        colab_url = getattr(args, "colab_url"),
        index_dir = getattr(args, "index_dir"),
        top_k     = getattr(args, "top_k"),
        delay     = args.delay,
    )
