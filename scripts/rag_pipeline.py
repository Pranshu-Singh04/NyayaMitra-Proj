"""
rag_pipeline.py
===============
Full end-to-end NyayaMitra RAG pipeline.
Combines: HybridRetrieverV2 + PromptBuilderV2 + LLM + ResponseParser.

Also contains VerifiedNyayaMitraPipeline — the pipeline with integrated
hallucination checking (merged from 12_rag_with_hallucination.py).

Classes:
    ParsedResponse              ← structured LLM output
    ResponseParser              ← parses raw LLM text into ParsedResponse
    NyayaMitraPipeline          ← main RAG pipeline (retrieve → generate → parse)
    VerifiedNyayaMitraPipeline  ← pipeline + NLI hallucination check per response
    VerifiedResponse            ← ParsedResponse + HallucinationReport
    ExperimentRunner            ← runs multiple models on same queries (ablation)

Usage:
    from rag_pipeline import NyayaMitraPipeline
    pipeline = NyayaMitraPipeline(index_dir="indexes", model="gemini", api_key="...")
    result   = pipeline.query("Can I get bail for murder?")
    print(result.answer)

    # With hallucination check:
    from rag_pipeline import VerifiedNyayaMitraPipeline
    vp     = VerifiedNyayaMitraPipeline(index_dir="indexes", model="gemini", api_key="...")
    result = vp.query("Can I get bail for murder?")
    result.print_summary()
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent))

# ── direct imports (no numbered-file hacks) ───────────────────────────────────
from hybrid_retriever  import HybridRetrieverV2
from prompt_builder    import PromptBuilderV2, IntentClassifier, QueryType
from llm_integration   import get_llm, LLMResponse, BaseLLM

# Aliases so any code that referenced the old names still works
HybridRetriever = HybridRetrieverV2
PromptBuilder   = PromptBuilderV2


# ══════════════════════════════════════════════════════════════════════════════
# ROBUST LJP PREDICTION PARSER  (IMPROVEMENT 10)
# ══════════════════════════════════════════════════════════════════════════════
def parse_ljp_prediction(raw_text: str) -> dict:
    """
    Three-priority extractor for LJP predictions.

    Priority 1 — explicit PREDICTION: label line (most reliable)
    Priority 2 — first line that contains a unique verdict keyword
    Priority 3 — keyword frequency vote (last resort)

    Returns {"prediction": str|None, "confidence": str, "citations": list[str]}
    """
    text = raw_text.upper().strip()
    prediction = None

    # Priority 1: explicit PREDICTION: label
    m = re.search(
        r'PREDICTION\s*:\s*\[?(PARTIALLY\s+ALLOWED|ALLOWED|DISMISSED)\]?', text
    )
    if m:
        prediction = m.group(1).strip()

    # Priority 2: scan lines for first unambiguous verdict word
    if not prediction:
        for line in text.split("\n"):
            if "PARTIALLY ALLOWED" in line or "PARTLY ALLOWED" in line:
                prediction = "PARTIALLY ALLOWED"
                break
            if (re.search(r"\bALLOWED\b", line)
                    and "NOT ALLOWED" not in line
                    and "DISMISSED" not in line):
                prediction = "ALLOWED"
                break
            if re.search(r"\bDISMISSED\b", line):
                prediction = "DISMISSED"
                break

    # Priority 3: keyword frequency vote
    if not prediction:
        a_count = len(re.findall(r"\bALLOWED\b", text))
        d_count = len(re.findall(r"\bDISMISSED\b", text))
        p_count = len(re.findall(r"\bPARTIALLY\b", text))
        if p_count > 0 and p_count >= min(a_count, d_count):
            prediction = "PARTIALLY ALLOWED"
        elif a_count > d_count:
            prediction = "ALLOWED"
        elif d_count > 0:
            prediction = "DISMISSED"

    # Confidence
    confidence = "MEDIUM"
    m = re.search(r"CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)", text)
    if m:
        confidence = m.group(1)

    # Citations — [Case N] and [Statute N] from the raw (mixed-case) text
    citations = re.findall(r"\[(?:Case|Statute)\s*\d+\]", raw_text, re.IGNORECASE)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "citations" : list(dict.fromkeys(citations)),   # deduplicate, preserve order
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARSED RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ParsedResponse:
    raw_text   : str
    query_type : QueryType
    model_name : str
    latency_ms : float

    # Legal Q&A
    answer     : str  = ""

    # LJP
    prediction : str  = ""    # ALLOWED / DISMISSED / PARTIALLY ALLOWED
    confidence : str  = ""    # HIGH / MEDIUM / LOW
    key_issues : list = field(default_factory=list)
    reasoning  : str  = ""

    # Common
    citations  : list = field(default_factory=list)
    disclaimer : str  = ""
    error      : str  = ""
    abstained  : bool = False   # TASK 2/6: True when LOW confidence → prediction withheld

    @property
    def success(self) -> bool:
        return not self.error and bool(self.raw_text.strip())

    def to_dict(self) -> dict:
        return {
            "query_type" : self.query_type.value,
            "model"      : self.model_name,
            "latency_ms" : round(self.latency_ms, 1),
            "answer"     : self.answer or self.raw_text,
            "prediction" : self.prediction,
            "confidence" : self.confidence,
            "key_issues" : self.key_issues,
            "reasoning"  : self.reasoning,
            "citations"  : self.citations,
            "disclaimer" : self.disclaimer,
            "error"      : self.error,
        }


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════════════════════════
class ResponseParser:
    """Parses raw LLM text into a structured ParsedResponse."""

    OUTCOME_MAP = {
        "allowed"          : "ALLOWED",
        "dismissed"        : "DISMISSED",
        "partially allowed": "PARTIALLY ALLOWED",
        "partially"        : "PARTIALLY ALLOWED",
        "rejected"         : "DISMISSED",
        "granted"          : "ALLOWED",
    }

    def parse(self, response: LLMResponse, query_type: QueryType) -> ParsedResponse:
        if not response.success:
            return ParsedResponse(
                raw_text   = "",
                query_type = query_type,
                model_name = response.model_name,
                latency_ms = response.latency_ms,
                error      = response.error,
            )

        text   = response.text.strip()
        parsed = ParsedResponse(
            raw_text   = text,
            query_type = query_type,
            model_name = response.model_name,
            latency_ms = response.latency_ms,
        )

        if query_type == QueryType.LJP:
            self._parse_ljp(text, parsed)
        else:
            parsed.answer = text

        parsed.citations  = self._extract_citations(text)
        parsed.disclaimer = self._extract_disclaimer(text)
        return parsed

    def _parse_ljp(self, text: str, parsed: ParsedResponse):
        # IMPROVEMENT 10: Use robust 3-priority parser
        result = parse_ljp_prediction(text)
        parsed.prediction = result["prediction"] or ""
        parsed.confidence = result["confidence"]
        # Citations from [Case N]/[Statute N] labels take priority
        if result["citations"]:
            parsed.citations = result["citations"]

        # KEY ISSUES
        m = re.search(r'KEY ISSUES\s*:\s*\n((?:[-•*]\s*.+\n?)+)', text, re.IGNORECASE)
        if m:
            parsed.key_issues = [
                line.lstrip("-•* ").strip()
                for line in m.group(1).split("\n")
                if line.strip() and not line.strip().startswith("REASONING")
            ]

        # REASONING
        m = re.search(r'REASONING\s*:\s*\n?(.*?)(?=DISCLAIMER|$)',
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            parsed.reasoning = m.group(1).strip()

        if not parsed.prediction:
            parsed.answer = text

    def _extract_citations(self, text: str) -> list[str]:
        citations = []
        section_refs = re.findall(
            r'(?:Section|§|Sec\.?)\s*(\d{1,4}[A-Z]?(?:\s+IPC|\s+BNS)?)',
            text, re.IGNORECASE
        )
        citations.extend([f"§{r.strip()}" for r in section_refs])
        case_refs = re.findall(
            r'\b([A-Z][a-z]+(?:_[A-Z][a-z]+)*_\d{4}_\d+)\b', text
        )
        citations.extend(case_refs)
        return list(dict.fromkeys(citations))

    def _extract_disclaimer(self, text: str) -> str:
        m = re.search(
            r'(?:DISCLAIMER|disclaimer)\s*:?\s*(.+?)(?:\n\n|$)',
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            return m.group(1).strip()
        if "not professional legal advice" in text.lower():
            return "This is AI-generated legal information, not professional legal advice."
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class NyayaMitraPipeline:
    """
    Full NyayaMitra RAG pipeline.
    retrieve → build prompt → generate → parse → return ParsedResponse

    Example:
        pipeline = NyayaMitraPipeline(index_dir="indexes", model="gemini", api_key="xxx")
        result   = pipeline.query("Can I get bail for murder?")
        print(result.answer)
    """

    def __init__(
        self,
        index_dir                : str  = "indexes",
        model                    : str  = "gemini",
        api_key                  : str  = None,
        colab_url                : str  = None,
        top_k                    : int  = 8,
        verbose                  : bool = True,
        abstain_on_low_confidence: bool = True,   # TASK 2/6: withhold LOW-confidence predictions
    ):
        self.top_k                     = top_k
        self.verbose                   = verbose
        self.model_name                = model
        self.abstain_on_low_confidence = abstain_on_low_confidence

        if verbose:
            print(f"\n{'='*55}")
            print(f"NyayaMitra Pipeline — Model: {model}")
            print(f"{'='*55}")

        self.retriever  = HybridRetrieverV2(index_dir=index_dir)
        # IMPROVEMENT 8: pass model_type so PromptBuilderV2 picks the right budget
        self.builder    = PromptBuilderV2(model_type=model)
        self.classifier = IntentClassifier()
        self.parser     = ResponseParser()
        self.llm        = get_llm(model=model, api_key=api_key, colab_url=colab_url)

        if verbose:
            print("Pipeline ready ✅")

    def query(
        self,
        user_query      : str,
        query_type      : QueryType = None,
        prediction_type : str       = "binary",
        top_k           : int       = None,
    ) -> ParsedResponse:
        """
        Main query method.

        Args:
            user_query     : the user's question or case facts
            query_type     : force a QueryType, or None for auto-detect
            prediction_type: "binary" or "ternary" for LJP queries
            top_k          : override default retrieval count

        Returns:
            ParsedResponse with .answer, .citations, .prediction (LJP), etc.
        """
        k  = top_k or self.top_k
        t0 = time.time()

        qt = query_type or self.classifier.classify(user_query)
        if self.verbose:
            print(f"\nQuery     : {user_query}")
            print(f"Intent    : {qt.value}")

        # Retrieve
        if qt == QueryType.STATUTE_LOOKUP:
            retrieved = self.retriever.retrieve_statutes_only(user_query, top_k=k)
        elif qt == QueryType.LJP:
            retrieved = self.retriever.retrieve_for_ljp(user_query, top_k=k)
        else:
            retrieved = self.retriever.retrieve(user_query, top_k=k)

        cases    = retrieved.get("cases",    [])
        statutes = retrieved.get("statutes", [])

        if self.verbose:
            print(f"Retrieved : {len(cases)} cases, {len(statutes)} statutes")

        # Build prompt
        if qt == QueryType.LEGAL_QA:
            prompt = self.builder.build_legal_qa(user_query, cases, statutes)
        elif qt == QueryType.LJP:
            prompt = self.builder.build_ljp(user_query, cases, statutes, prediction_type)
        elif qt == QueryType.STATUTE_LOOKUP:
            prompt = self.builder.build_statute_lookup(user_query, statutes)
        elif qt == QueryType.SUMMARISE:
            prompt = self.builder.build_summarise(user_query)
        else:
            prompt = self.builder.build_legal_qa(user_query, cases, statutes)

        if self.verbose:
            print(f"Prompt    : ~{prompt.token_estimate()} tokens")

        # Generate — IMPROVEMENT 12: use task-specific temperature
        if self.verbose:
            print(f"Generating with {self.model_name}...")
        task_str = qt.value.replace("legal_", "")   # "legal_qa" → "qa", "ljp" → "ljp", etc.
        llm_resp = self.llm.generate_with_task(
            prompt.system_prompt, prompt.user_prompt, task_type=task_str
        )

        # Parse
        parsed   = self.parser.parse(llm_resp, qt)
        total_ms = (time.time() - t0) * 1000

        # TASK 2/6: confidence-based abstention for LJP
        if (qt == QueryType.LJP
                and self.abstain_on_low_confidence
                and parsed.confidence == "LOW"):
            parsed.abstained  = True
            parsed.prediction = ""   # withhold prediction rather than guess

        if self.verbose:
            print(f"Done      : {total_ms:.0f}ms (LLM: {llm_resp.latency_ms:.0f}ms)")
            if qt == QueryType.LJP:
                if parsed.abstained:
                    print(f"LJP       : ABSTAINED (LOW confidence)")
                else:
                    print(f"LJP       : {parsed.prediction} ({parsed.confidence})")

        return parsed

    def batch_query(self, queries: list[str], **kwargs) -> list[ParsedResponse]:
        """Run multiple queries sequentially."""
        results = []
        for i, q in enumerate(queries, 1):
            if self.verbose:
                print(f"\n[{i}/{len(queries)}]")
            results.append(self.query(q, **kwargs))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# VERIFIED RESPONSE  (for VerifiedNyayaMitraPipeline)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class VerifiedResponse:
    """ParsedResponse + hallucination report from NLI check."""
    parsed_response      : ParsedResponse
    hallucination_report : object           # HallucinationReport from hallucination_checker.py
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
# VERIFIED PIPELINE  (RAG + NLI hallucination check)
# Merged from 12_rag_with_hallucination.py — same domain, extends pipeline.
# ══════════════════════════════════════════════════════════════════════════════
class VerifiedNyayaMitraPipeline:
    """
    NyayaMitraPipeline with integrated NLI hallucination checking.

    Every call to .query() runs:
      1. Retrieve (FAISS + BM25)
      2. Build prompt
      3. Generate (LLM)
      4. Parse response
      5. NLI hallucination check against retrieved chunks
      6. Return VerifiedResponse

    Example:
        vp     = VerifiedNyayaMitraPipeline(index_dir="indexes", model="gemini", api_key="...")
        result = vp.query("Can I get bail for murder?")
        result.print_summary()
    """

    def __init__(
        self,
        index_dir           : str  = "indexes",
        model               : str  = "gemini",
        api_key             : str  = None,
        colab_url           : str  = None,
        top_k               : int  = 5,
        verbose             : bool = True,
        nli_device          : str  = None,
        max_chunks_per_claim: int  = 5,
    ):
        self.verbose               = verbose
        self.max_chunks_per_claim  = max_chunks_per_claim

        if verbose:
            print(f"\n{'='*55}")
            print(f"Verified NyayaMitra Pipeline — Model: {model}")
            print(f"{'='*55}")

        self.pipeline = NyayaMitraPipeline(
            index_dir = index_dir,
            model     = model,
            api_key   = api_key,
            colab_url = colab_url,
            top_k     = top_k,
            verbose   = verbose,
        )

        # Lazy import — hallucination_checker may not be available in all envs
        if verbose:
            print("\nLoading hallucination checker (NLI model)...")
        from hallucination_checker import HallucinationCheckerV2
        self.checker = HallucinationCheckerV2(device=nli_device)

        if verbose:
            print("Verified pipeline ready ✅")

    def query(
        self,
        user_query : str,
        query_type : QueryType = None,
    ) -> VerifiedResponse:
        """Run query + hallucination check. Returns VerifiedResponse."""
        parsed    = self.pipeline.query(user_query, query_type=query_type)
        retriever = self.pipeline.retriever

        if query_type == QueryType.STATUTE_LOOKUP:
            retrieved = retriever.retrieve_statutes_only(user_query)
        else:
            retrieved = retriever.retrieve(user_query)

        all_chunks = retrieved.get("cases", []) + retrieved.get("statutes", [])

        if self.verbose:
            print("Running hallucination check...")

        report = self.checker.check(
            answer    = parsed.raw_text,
            chunks    = all_chunks,
        )

        return VerifiedResponse(
            parsed_response      = parsed,
            hallucination_report = report,
            retrieved_chunks     = all_chunks,
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER  (multi-model ablation study)
# ══════════════════════════════════════════════════════════════════════════════
class ExperimentRunner:
    """
    Runs the same queries across multiple LLM backends.
    Saves results as JSON for the research paper ablation study.

    Example:
        runner = ExperimentRunner(
            index_dir = "indexes",
            models    = [{"name": "gemini", "api_key": "..."}, {"name": "gpt", "api_key": "..."}],
        )
        runner.run([{"query": "What is Section 302?", "query_type": "legal_qa"}])
    """

    def __init__(
        self,
        index_dir : str,
        models    : list[dict],    # [{"name": "gemini", "api_key": "xxx"}, ...]
        output_dir: str = "results",
    ):
        self.index_dir  = index_dir
        self.models     = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.pipelines: dict[str, NyayaMitraPipeline] = {}

        print(f"\nInitialising {len(models)} pipelines...")
        for m in models:
            name = m["name"]
            try:
                self.pipelines[name] = NyayaMitraPipeline(
                    index_dir = index_dir,
                    model     = name,
                    api_key   = m.get("api_key"),
                    colab_url = m.get("colab_url"),
                    verbose   = False,
                )
                print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")

    def run(
        self,
        queries     : list[dict],   # [{"query": "...", "query_type": "legal_qa"}, ...]
        save_results: bool = True,
    ) -> dict:
        all_results = {name: [] for name in self.pipelines}

        for i, q_item in enumerate(queries, 1):
            query  = q_item["query"]
            qt_str = q_item.get("query_type", "legal_qa")
            qt     = QueryType(qt_str)

            print(f"\n{'─'*55}")
            print(f"Query {i}/{len(queries)}: {query[:60]}")

            for model_name, pipeline in self.pipelines.items():
                print(f"  [{model_name}]", end=" ", flush=True)
                try:
                    response = pipeline.query(query, query_type=qt)
                    result   = {"query": query, "query_type": qt_str, **response.to_dict()}
                    print(f"✅ {response.latency_ms:.0f}ms")
                    if qt == QueryType.LJP and response.prediction:
                        print(f"    Prediction: {response.prediction} ({response.confidence})")
                except Exception as e:
                    result = {"query": query, "error": str(e)}
                    print(f"❌ {e}")
                all_results[model_name].append(result)

        if save_results:
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"experiment_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Results saved: {path}")

        self._print_summary(all_results, queries)
        return all_results

    def _print_summary(self, results: dict, queries: list[dict]):
        print(f"\n{'='*55}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*55}")
        for model_name, model_results in results.items():
            avg_lat = sum(r.get("latency_ms", 0) for r in model_results) / max(len(model_results), 1)
            errors  = sum(1 for r in model_results if r.get("error"))
            print(f"\n  {model_name}:")
            print(f"    Queries     : {len(model_results)}")
            print(f"    Errors      : {errors}")
            print(f"    Avg latency : {avg_lat:.0f}ms")
            ljp_res = [r for r in model_results if r.get("query_type") == "ljp"]
            if ljp_res:
                from collections import Counter
                preds = Counter(r.get("prediction", "?") for r in ljp_res)
                print(f"    LJP preds   : {dict(preds)}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="NyayaMitra RAG Pipeline")
    parser.add_argument("--model",      default="gemini",
                        choices=["gemini", "gpt", "groq", "inlegalllama"])
    parser.add_argument("--query",      default="Can I get bail for a murder charge?")
    parser.add_argument("--mode",       default="auto",
                        choices=["auto", "qa", "ljp", "statute", "summarise",
                                 "compare", "verified"])
    parser.add_argument("--index_dir",  default="indexes")
    parser.add_argument("--api_key",    default=None)
    parser.add_argument("--colab_url",  default=None)
    parser.add_argument("--top_k",      type=int, default=5)
    args = parser.parse_args()

    api_key = (
        args.api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    # Compare mode
    if args.mode == "compare":
        models = []
        if api_key:
            models.append({"name": "gemini", "api_key": api_key})
        if os.environ.get("OPENAI_API_KEY"):
            models.append({"name": "gpt", "api_key": os.environ.get("OPENAI_API_KEY")})
        if args.colab_url:
            models.append({"name": "inlegalllama", "colab_url": args.colab_url})
        if not models:
            print("❌ No API keys found. Set GEMINI_API_KEY and/or OPENAI_API_KEY.")
            return
        runner = ExperimentRunner(index_dir=args.index_dir, models=models)
        runner.run([{"query": args.query, "query_type": "legal_qa"}])
        return

    # Verified mode (with NLI hallucination check)
    if args.mode == "verified":
        vp     = VerifiedNyayaMitraPipeline(
            index_dir = args.index_dir,
            model     = args.model,
            api_key   = api_key,
            colab_url = args.colab_url,
            top_k     = args.top_k,
        )
        result = vp.query(args.query)
        result.print_summary()
        return

    # Standard single-model mode
    qt_map = {
        "qa"      : QueryType.LEGAL_QA,
        "ljp"     : QueryType.LJP,
        "statute" : QueryType.STATUTE_LOOKUP,
        "summarise": QueryType.SUMMARISE,
        "auto"    : None,
    }

    pipeline = NyayaMitraPipeline(
        index_dir = args.index_dir,
        model     = args.model,
        api_key   = api_key,
        colab_url = args.colab_url,
        top_k     = args.top_k,
        verbose   = True,
    )

    result = pipeline.query(args.query, query_type=qt_map.get(args.mode))

    print(f"\n{'='*55}")
    print(f"RESPONSE ({result.model_name})")
    print(f"{'='*55}")
    print(result.raw_text)
    if result.citations:
        print(f"\nCitations  : {', '.join(result.citations)}")
    if result.prediction:
        print(f"Prediction : {result.prediction} ({result.confidence})")
    if result.disclaimer:
        print(f"\n⚠  {result.disclaimer}")
    print(f"\nLatency    : {result.latency_ms:.0f}ms")


if __name__ == "__main__":
    main()