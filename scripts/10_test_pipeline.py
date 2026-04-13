"""
10_test_pipeline.py
====================
End-to-end test suite for the full NyayaMitra RAG pipeline.

Usage:
    python scripts/10_test_pipeline.py --model inlegalllama --mode paper_eval --colab_url "https://xxx.ngrok-free.app/generate"
    python scripts/10_test_pipeline.py --model gemini --mode paper_eval --api_key YOUR_KEY
    python scripts/10_test_pipeline.py --model inlegalllama --mode basic --colab_url "https://xxx.ngrok-free.app/generate"
"""

import os
import json
import time
import argparse
import sys
from pathlib import Path

# Force unbuffered output so prints appear immediately on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

_base = Path(__file__).parent
sys.path.insert(0, str(_base))

from rag_pipeline  import NyayaMitraPipeline, ExperimentRunner
from prompt_builder import QueryType

TEST_QUERIES = {
    "legal_qa": [
        "Can I get bail for a murder charge under Section 302 IPC?",
        "What are my rights if police arrest me without a warrant?",
        "How do I file an FIR for domestic violence?",
        "What is the difference between IPC and BNS 2023?",
        "Can a person be charged under both IPC Section 420 and BNS Section 318?",
    ],
    "ljp": [
        "Accused charged with murder under Section 302. Evidence is circumstantial. First offence. No prior criminal record. Supreme Court appeal.",
        "Petitioner seeks bail in cheating case under IPC Section 420. Amount involved is Rs 50 lakhs. Accused is a flight risk. High Court hearing.",
        "Wife filed case under Section 498A for dowry harassment. Husband denies all charges. Three witnesses for prosecution. District Court.",
        "Appeal against conviction for theft under Section 379 IPC. Stolen property recovered. Accused claims false implication. High Court.",
        "Anticipatory bail application in rape case Section 376 IPC. Accused is a public servant. Victim is a minor. Sessions Court.",
    ],
    "statute": [
        "What is Section 302 IPC?",
        "Explain Section 498A Indian Penal Code",
        "What does Section 420 IPC say about cheating?",
        "What is the BNS 2023 equivalent of IPC Section 302?",
        "Explain Article 21 of the Constitution of India",
    ],
}

QT_MAP = {
    "legal_qa" : None,
    "ljp"      : None,
    "statute"  : None,
    "summarise": None,
}


def _build_qt_map():
    global QT_MAP
    QT_MAP = {
        "legal_qa" : QueryType.LEGAL_QA,
        "ljp"      : QueryType.LJP,
        "statute"  : QueryType.STATUTE_LOOKUP,
        "summarise": QueryType.SUMMARISE,
    }


def run_basic_tests(pipeline, model_name):
    _build_qt_map()
    print(f"\n{'='*55}")
    print(f"BASIC TESTS — {model_name}")
    print(f"{'='*55}")

    test_cases = [
        (TEST_QUERIES["legal_qa"][0],  QueryType.LEGAL_QA,       "Legal Q&A"),
        (TEST_QUERIES["ljp"][0],       QueryType.LJP,            "Judgment Prediction"),
        (TEST_QUERIES["statute"][0],   QueryType.STATUTE_LOOKUP, "Statute Lookup"),
    ]

    results = []
    for query, qt, label in test_cases:
        print(f"\n── {label}")
        print(f"   Query: {query[:70]}...")
        try:
            response = pipeline.query(query, query_type=qt)
            if response.success:
                print(f"   ✅ Response ({response.latency_ms:.0f}ms)")
                print(f"   Preview: {response.raw_text[:200]}...")
                if response.prediction:
                    print(f"   Prediction: {response.prediction} ({response.confidence})")
                if response.citations:
                    print(f"   Citations: {response.citations[:3]}")
                results.append({"test": label, "status": "PASS", "latency_ms": response.latency_ms})
            else:
                print(f"   ❌ Error: {response.error}")
                results.append({"test": label, "status": "FAIL", "error": response.error})
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({"test": label, "status": "ERROR", "error": str(e)})

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{'─'*55}")
    print(f"Results: {passed}/{len(results)} tests passed")
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        lat  = f" ({r.get('latency_ms',0):.0f}ms)" if "latency_ms" in r else ""
        print(f"  {icon} {r['test']}{lat}")
    return results


def run_paper_eval(pipeline, model_name, output_dir="results"):
    _build_qt_map()
    print(f"\n{'='*55}")
    print(f"PAPER EVALUATION — {model_name}")
    print(f"{'='*55}")

    Path(output_dir).mkdir(exist_ok=True)
    all_results = []

    for qt_str, queries in TEST_QUERIES.items():
        qt = QT_MAP.get(qt_str, QueryType.LEGAL_QA)
        print(f"\n── {qt_str.upper()} ({len(queries)} queries)")

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query[:60]}...")
            try:
                response = pipeline.query(query, query_type=qt)
                result = {
                    "query_type" : qt_str,
                    "query"      : query,
                    "model"      : model_name,
                    "latency_ms" : response.latency_ms,
                    "answer"     : response.answer or response.raw_text,
                    "prediction" : response.prediction,
                    "confidence" : response.confidence,
                    "citations"  : response.citations,
                    "success"    : response.success,
                    "error"      : response.error,
                }
                status = "✅" if response.success else "❌"
                print(f"         {status} {response.latency_ms:.0f}ms")
            except Exception as e:
                result = {
                    "query_type": qt_str, "query": query,
                    "model": model_name, "error": str(e), "success": False
                }
                print(f"         ❌ {e}")

            all_results.append(result)
            time.sleep(0.5)

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"eval_{model_name}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print("EVALUATION SUMMARY")
    print(f"{'='*55}")
    total     = len(all_results)
    succeeded = sum(1 for r in all_results if r.get("success"))
    avg_lat   = sum(r.get("latency_ms", 0) for r in all_results) / max(total, 1)

    print(f"  Model        : {model_name}")
    print(f"  Total queries: {total}")
    print(f"  Succeeded    : {succeeded}/{total}")
    print(f"  Avg latency  : {avg_lat:.0f}ms")

    for qt_str in ["legal_qa", "ljp", "statute"]:
        qt_results = [r for r in all_results if r.get("query_type") == qt_str]
        qt_success = sum(1 for r in qt_results if r.get("success"))
        qt_lat     = sum(r.get("latency_ms", 0) for r in qt_results) / max(len(qt_results), 1)
        print(f"  {qt_str:15s}: {qt_success}/{len(qt_results)} passed | avg {qt_lat:.0f}ms")

    print(f"\n  Results saved: {output_path}")
    return all_results


def compare_models(index_dir, api_keys, colab_url=None):
    _build_qt_map()
    models = []
    if api_keys.get("gemini"):
        models.append({"name": "gemini", "api_key": api_keys["gemini"]})
    if api_keys.get("gpt"):
        models.append({"name": "gpt", "api_key": api_keys["gpt"]})
    if colab_url:
        models.append({"name": "inlegalllama", "colab_url": colab_url})

    if len(models) < 2:
        print("Need at least 2 models to compare.")
        return

    queries = []
    for qt_str, qs in TEST_QUERIES.items():
        for q in qs[:2]:
            queries.append({"query": q, "query_type": qt_str})

    runner = ExperimentRunner(index_dir=index_dir, models=models, output_dir="results")
    runner.run(queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="gemini", choices=["gemini", "gpt", "inlegalllama"])
    parser.add_argument("--mode",      default="basic",  choices=["basic", "paper_eval", "compare"])
    parser.add_argument("--api_key",   default=None)
    parser.add_argument("--colab_url", default=None)
    parser.add_argument("--index_dir", default="indexes")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if args.mode == "compare":
        compare_models(
            index_dir = args.index_dir,
            api_keys  = {
                "gemini": os.environ.get("GEMINI_API_KEY", ""),
                "gpt"   : os.environ.get("OPENAI_API_KEY", ""),
            },
            colab_url = args.colab_url,
        )
        return

    pipeline = NyayaMitraPipeline(
        index_dir = args.index_dir,
        model     = args.model,
        api_key   = api_key,
        colab_url = args.colab_url,
        verbose   = True,
    )

    if args.mode == "basic":
        run_basic_tests(pipeline, args.model)
    elif args.mode == "paper_eval":
        run_paper_eval(pipeline, args.model)


if __name__ == "__main__":
    main()