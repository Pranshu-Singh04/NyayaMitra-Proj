"""
12_rag_with_hallucination.py
=============================
Full RAG pipeline with integrated hallucination checking.
Extends 09_rag_pipeline.py by adding a post-generation
hallucination check on every response.

Usage:
    python scripts/12_rag_with_hallucination.py --model inlegalllama --colab_url "https://xxx.ngrok-free.app/generate" --query "Can I get bail for murder?"
    python scripts/12_rag_with_hallucination.py --model inlegalllama --colab_url "https://xxx.ngrok-free.app/generate" --mode compare_rag
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from dataclasses import dataclass

_base = Path(__file__).parent

def _load(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_hr  = _load("hybrid_retriever",       _base / "06_hybrid_retriever.py")
_pb  = _load("prompt_builder",         _base / "07_prompt_builder.py")
_li  = _load("llm_integration",        _base / "08_llm_integration.py")
_rp  = _load("rag_pipeline",           _base / "09_rag_pipeline.py")
_hc  = _load("hallucination_checker",  _base / "11_hallucination_checker.py")

NyayaMitraPipeline   = _rp.NyayaMitraPipeline
QueryType            = _pb.QueryType
HallucinationChecker = _hc.HallucinationChecker
HallucinationReport  = _hc.HallucinationReport


# ══════════════════════════════════════════════════════════════════════════════
# ENHANCED RESPONSE — adds hallucination report to pipeline response
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class VerifiedResponse:
    parsed_response      : object           # ParsedResponse from pipeline
    hallucination_report : HallucinationReport
    retrieved_chunks     : list[dict]

    @property
    def answer(self) -> str:
        return self.parsed_response.answer or self.parsed_response.raw_text

    @property
    def grounding_score(self) -> float:
        return self.hallucination_report.grounding_score

    @property
    def hallucination_score(self) -> float:
        return self.hallucination_report.hallucination_score

    def to_dict(self) -> dict:
        return {
            "answer"              : self.answer,
            "prediction"          : self.parsed_response.prediction,
            "confidence"          : self.parsed_response.confidence,
            "citations"           : self.parsed_response.citations,
            "grounding_score"     : round(self.grounding_score, 4),
            "hallucination_score" : round(self.hallucination_score, 4),
            "hallucination_report": self.hallucination_report.to_dict(),
        }

    def print_summary(self):
        print(f"\n{'='*55}")
        print("VERIFIED RESPONSE")
        print(f"{'='*55}")
        print(f"\nAnswer:\n{self.answer}")
        if self.parsed_response.prediction:
            print(f"\nPrediction : {self.parsed_response.prediction} ({self.parsed_response.confidence})")
        if self.parsed_response.citations:
            print(f"Citations  : {', '.join(self.parsed_response.citations)}")
        print(f"\n{self.hallucination_report.summary()}")
        print(f"\nAnnotated (first 600 chars):")
        print(self.hallucination_report.annotated_answer[:600])


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class VerifiedNyayaMitraPipeline:
    """
    NyayaMitra pipeline with hallucination checking on every response.

    Steps:
      1. Retrieve (FAISS + BM25)
      2. Build prompt
      3. Generate (LLM)
      4. Parse response
      5. Check hallucination (NLI)
      6. Return VerifiedResponse

    Usage:
        pipeline = VerifiedNyayaMitraPipeline(
            index_dir = "indexes",
            model     = "inlegalllama",
            colab_url = "https://xxx.ngrok-free.app/generate",
        )
        result = pipeline.query("Can I get bail for murder?")
        result.print_summary()
    """

    def __init__(
        self,
        index_dir  : str   = "indexes",
        model      : str   = "gemini",
        api_key    : str   = None,
        colab_url  : str   = None,
        top_k      : int   = 5,
        verbose    : bool  = True,
        nli_device : str   = None,
        max_chunks_per_claim: int = 5,
    ):
        self.verbose              = verbose
        self.max_chunks_per_claim = max_chunks_per_claim

        if verbose:
            print(f"\n{'='*55}")
            print(f"Verified NyayaMitra Pipeline — Model: {model}")
            print(f"{'='*55}")

        # Base pipeline (retrieval + generation)
        self.pipeline = NyayaMitraPipeline(
            index_dir = index_dir,
            model     = model,
            api_key   = api_key,
            colab_url = colab_url,
            top_k     = top_k,
            verbose   = verbose,
        )

        # Hallucination checker
        if verbose:
            print("\nLoading hallucination checker...")
        self.checker = HallucinationChecker(device=nli_device)

        if verbose:
            print("\nVerified pipeline ready ✅")

    def query(
        self,
        user_query  : str,
        query_type  : QueryType = None,
    ) -> VerifiedResponse:
        """
        Run query through full pipeline + hallucination check.
        """
        # Step 1-4: Standard RAG pipeline
        parsed = self.pipeline.query(user_query, query_type=query_type)

        # Get the retrieved chunks that were used
        retriever = self.pipeline.retriever
        if query_type == QueryType.STATUTE_LOOKUP:
            retrieved = retriever.retrieve_statutes_only(user_query)
        else:
            retrieved = retriever.retrieve(user_query)

        all_chunks = retrieved.get("cases", []) + retrieved.get("statutes", [])

        # Step 5: Hallucination check
        if self.verbose:
            print("Running hallucination check...")

        report = self.checker.check(
            answer = parsed.raw_text,
            chunks = all_chunks,
            max_chunks_per_claim = self.max_chunks_per_claim,
        )

        return VerifiedResponse(
            parsed_response      = parsed,
            hallucination_report = report,
            retrieved_chunks     = all_chunks,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RAG vs NO-RAG COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def run_rag_vs_no_rag_comparison(
    index_dir  : str,
    model      : str,
    api_key    : str  = None,
    colab_url  : str  = None,
    output_dir : str  = "results",
):
    """
    Compare hallucination rates between RAG and no-RAG for the same queries.
    This generates the ablation study data for the paper.

    RAG    : query → retrieve → prompt with context → generate
    no-RAG : query → prompt WITHOUT context → generate (LLM knowledge only)
    """
    from llm_integration   import get_llm
    from prompt_builder    import PromptBuilder
    from hybrid_retriever  import HybridRetriever

    test_queries = [
        "What is the punishment for murder under Indian law?",
        "Can a person get bail for a non-bailable offence?",
        "What is the procedure to file an FIR?",
        "What is Section 498A IPC?",
        "Can an accused person be held in custody without trial?",
    ]

    print(f"\n{'='*55}")
    print("RAG vs NO-RAG HALLUCINATION COMPARISON")
    print(f"{'='*55}")

    # Load components
    retriever = HybridRetriever(index_dir=index_dir)
    checker   = HallucinationChecker()
    llm       = get_llm(model=model, api_key=api_key, colab_url=colab_url)
    builder   = PromptBuilder(max_context_tokens=3000)

    system_prompt = "You are NyayaMitra, an AI legal advisor for Indian law. Answer concisely."

    all_comparisons = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] {query}")

        # Retrieve context
        retrieved  = retriever.retrieve(query, top_k=5)
        cases      = retrieved.get("cases", [])
        statutes   = retrieved.get("statutes", [])
        all_chunks = cases + statutes

        # RAG answer (with retrieved context)
        print("  Generating RAG answer...")
        rag_prompt  = builder.build_legal_qa(query, cases, statutes)
        rag_resp    = llm.generate_from_prompt(rag_prompt)
        rag_answer  = rag_resp.text

        # No-RAG answer (LLM only, no context)
        print("  Generating no-RAG answer...")
        no_rag_resp   = llm.generate(system_prompt, query)
        no_rag_answer = no_rag_resp.text

        # Compare hallucination
        print("  Checking hallucination...")
        comparison = checker.compare_rag_vs_no_rag(
            query           = query,
            rag_answer      = rag_answer,
            no_rag_answer   = no_rag_answer,
            retrieved_chunks= all_chunks,
        )
        comparison["rag_answer"]    = rag_answer
        comparison["no_rag_answer"] = no_rag_answer

        all_comparisons.append(comparison)

        # Print mini summary
        rag_g    = comparison["rag"]["grounding_score"]
        no_rag_g = comparison["no_rag"]["grounding_score"]
        delta    = comparison["improvement"]["grounding_delta"]
        print(f"  RAG grounding: {rag_g:.1%} | No-RAG grounding: {no_rag_g:.1%} | Δ: +{delta:.1%}")

    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"rag_vs_no_rag_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)

    # Print aggregate summary
    print(f"\n{'='*55}")
    print("AGGREGATE RESULTS")
    print(f"{'='*55}")

    avg_rag_grounding    = sum(c["rag"]["grounding_score"]    for c in all_comparisons) / len(all_comparisons)
    avg_no_rag_grounding = sum(c["no_rag"]["grounding_score"] for c in all_comparisons) / len(all_comparisons)
    avg_rag_halluc       = sum(c["rag"]["hallucination_score"]    for c in all_comparisons) / len(all_comparisons)
    avg_no_rag_halluc    = sum(c["no_rag"]["hallucination_score"] for c in all_comparisons) / len(all_comparisons)

    print(f"\n  RAG Pipeline:")
    print(f"    Avg grounding score    : {avg_rag_grounding:.1%}")
    print(f"    Avg hallucination score: {avg_rag_halluc:.1%}")
    print(f"\n  No-RAG (LLM only):")
    print(f"    Avg grounding score    : {avg_no_rag_grounding:.1%}")
    print(f"    Avg hallucination score: {avg_no_rag_halluc:.1%}")
    print(f"\n  Improvement from RAG:")
    print(f"    Grounding  : +{avg_rag_grounding - avg_no_rag_grounding:.1%}")
    print(f"    Hallucination reduction: {avg_no_rag_halluc - avg_rag_halluc:.1%}")
    print(f"\n  Results saved: {output_path}")

    return all_comparisons


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="inlegalllama",
                        choices=["gemini", "gpt", "inlegalllama"])
    parser.add_argument("--query",     default="Can I get bail for a murder charge?")
    parser.add_argument("--mode",      default="single",
                        choices=["single", "compare_rag"])
    parser.add_argument("--index_dir", default="indexes")
    parser.add_argument("--api_key",   default=None)
    parser.add_argument("--colab_url", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if args.mode == "compare_rag":
        run_rag_vs_no_rag_comparison(
            index_dir = args.index_dir,
            model     = args.model,
            api_key   = api_key,
            colab_url = args.colab_url,
        )
        return

    # Single query mode
    pipeline = VerifiedNyayaMitraPipeline(
        index_dir = args.index_dir,
        model     = args.model,
        api_key   = api_key,
        colab_url = args.colab_url,
        verbose   = True,
    )

    result = pipeline.query(args.query)
    result.print_summary()


if __name__ == "__main__":
    main()
