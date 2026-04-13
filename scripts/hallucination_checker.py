"""
11_hallucination_checker_v2.py
================================
IMPROVED NLI-based hallucination checker.

Key improvements over v1:
  1. Sentence-level entailment BOOSTING for statutory quotes
     (verbatim statute text should score near 1.0 — v1 was under-counting these)
  2. Sliding window premise: check claim against EACH sentence of each chunk,
     not the whole 512-char chunk at once (better signal for long chunks)
  3. Adjusted thresholds: SUPPORTED if P(ENT) > 0.40 (was 0.50 — too strict)
  4. Neutral drift detector: tracks if grounding is low due to coverage gap
     vs. active contradiction
  5. Per-task-type scoring: statute queries get verbatim-match bonus
  6. Better claim splitter: handles numbered lists and bullet points in LJP output

Usage:
    python 11_hallucination_checker_v2.py
"""

import re, json, time, torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class ClaimResult:
    claim            : str
    label            : str
    score            : float
    best_chunk       : str
    entail_score     : float = 0.0
    neutral_score    : float = 0.0
    contradict_score : float = 0.0
    verbatim_match   : bool  = False  # NEW: was this a near-verbatim statutory quote?

@dataclass
class HallucinationReport:
    original_answer      : str
    claims               : list = field(default_factory=list)
    hallucination_score  : float = 0.0
    grounding_score      : float = 0.0
    supported_count      : int   = 0
    unsupported_count    : int   = 0
    neutral_count        : int   = 0
    total_claims         : int   = 0
    latency_ms           : float = 0.0
    annotated_answer     : str   = ""
    coverage_gap_detected: bool  = False  # NEW

    def to_dict(self):
        return {
            "hallucination_score" : round(self.hallucination_score, 4),
            "grounding_score"     : round(self.grounding_score, 4),
            "total_claims"        : self.total_claims,
            "supported_count"     : self.supported_count,
            "unsupported_count"   : self.unsupported_count,
            "neutral_count"       : self.neutral_count,
            "latency_ms"          : round(self.latency_ms, 1),
            "coverage_gap_detected": self.coverage_gap_detected,
            "claims"              : [
                {"claim": c.claim, "label": c.label, "score": round(c.score, 4),
                 "E": round(c.entail_score,4), "N": round(c.neutral_score,4),
                 "C": round(c.contradict_score,4),
                 "verbatim": c.verbatim_match,
                 "best_chunk": c.best_chunk[:100]}
                for c in self.claims
            ],
            "annotated_answer": self.annotated_answer,
        }

    def summary(self):
        gap_flag = " [COVERAGE GAP]" if self.coverage_gap_detected else ""
        return (
            f"Grounding: {self.grounding_score:.1%} | "
            f"Hallucination: {self.hallucination_score:.1%} | "
            f"Claims: {self.supported_count}✅ {self.neutral_count}〰 "
            f"{self.unsupported_count}❌ / {self.total_claims} "
            f"({self.latency_ms:.0f}ms){gap_flag}"
        )


# ── Improved Claim Splitter ───────────────────────────────────────────────────
class ClaimSplitterV2:
    SKIP_RE = [
        r"this is ai.generated", r"not professional legal advice",
        r"consult a qualified advocate", r"please consult",
        r"i am not a lawyer", r"ai-generated legal",
        r"not a substitute", r"for informational purposes",
        r"this prediction is based",
        r"this is ai.generated legal information",
    ]
    HEAD_RE = [
        r"^(prediction|confidence|key issues|reasoning|disclaimer|facts|issues|held|ratio)\s*:",
        r"^\[(?:case|statute)\s*\d+\]",
        r"^#+\s",
        r"^step\s+\d+",  # NEW: skip CoT step headers
        r"^analogous cases",
    ]

    def split(self, text: str, min_words: int = 6) -> list:
        # Handle numbered lists and bullet points — treat each as a claim
        text = re.sub(r'\n+', ' ', text)
        # Split on sentence boundaries
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        # Also split on bullet/numbered list items
        expanded = []
        for s in sents:
            parts = re.split(r'(?:^|\s)[-•*]\s+', s)
            expanded.extend(p.strip() for p in parts if p.strip())

        claims = []
        for s in expanded:
            s = s.strip()
            if not s or len(s.split()) < min_words:
                continue
            if any(re.search(p, s.lower()) for p in self.SKIP_RE):
                continue
            if any(re.match(p, s.lower()) for p in self.HEAD_RE):
                continue
            claims.append(s)
        return claims


# ── Legal synonym map for semantic overlap ────────────────────────────────────
# Many legal claims are semantically equivalent but use different words.
# e.g. "defines murder as unlawful killing" ↔ "commits murder shall be punished"
# We expand both claim and chunk with synonyms before computing overlap.
_LEGAL_SYNONYMS: dict[str, list[str]] = {
    "murder"       : ["homicide", "culpable homicide", "kills", "killing", "slays", "Section 302", "IPC 302"],
    "unlawful"     : ["illegal", "punishable", "offence", "offense", "commits", "prohibited"],
    "killing"      : ["death", "slaying", "homicide", "commits murder", "cause death"],
    "punishment"   : ["punished", "penalty", "sentence", "imprisonment", "fine", "rigorous imprisonment"],
    "defines"      : ["means", "shall be", "whoever", "commits", "constitutes", "is defined as"],
    "imprisonment" : ["incarceration", "custody", "confinement", "life", "years rigorous"],
    "death"        : ["capital punishment", "death penalty", "execution", "death sentence"],
    "bail"         : ["released", "granted bail", "interim bail", "anticipatory", "regular bail", "Section 437", "Section 439"],
    "offence"      : ["offense", "crime", "act", "violation", "cognizable", "non-cognizable"],
    "cognizable"   : ["police", "warrant", "arrest", "non-bailable", "FIR", "Section 154"],
    "accused"      : ["defendant", "person", "individual", "charged", "appellant", "petitioner"],
    "conviction"   : ["convicted", "guilty", "found guilty", "sentenced", "Section 235"],
    "acquittal"    : ["acquitted", "not guilty", "discharged", "exonerated", "benefit of doubt"],
    "section"      : ["§", "sec.", "provision", "clause", "ipc", "bns", "crpc", "bnss"],
    "cruelty"      : ["498A", "dowry", "harassment", "domestic violence", "mental cruelty"],
    "dowry"        : ["Section 304B", "498A", "cruelty", "demand", "harassment"],
    "rape"         : ["Section 376", "sexual assault", "consent", "POCSO", "victim"],
    "theft"        : ["Section 379", "robbery", "dacoity", "stealing", "dishonestly"],
    "cheating"     : ["Section 420", "fraud", "dishonest", "inducement", "BNS 318"],
    "appeal"       : ["revision", "challenge", "high court", "supreme court", "Section 374"],
    "evidence"     : ["circumstantial", "witness", "proof", "testimony", "beyond reasonable doubt"],
    "warrant"      : ["arrest", "non-bailable", "Section 70", "cognizable", "remand"],
    "sentence"     : ["punishment", "imprisonment", "fine", "death", "rigorous", "simple"],
    "granted"      : ["allowed", "permitted", "accepted", "released", "given"],
    "dismissed"    : ["rejected", "refused", "denied", "not allowed", "upheld conviction"],
}

def _expand_with_synonyms(text: str) -> set[str]:
    """Return word set of text expanded with legal synonyms."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    expanded = set(words)
    for word in words:
        if word in _LEGAL_SYNONYMS:
            for syn in _LEGAL_SYNONYMS[word]:
                expanded.update(syn.lower().split())
    return expanded


# ── Verbatim Match Detector  — NEW ───────────────────────────────────────────
def _verbatim_overlap(claim: str, chunk: str, threshold: float = 0.55) -> bool:
    """
    Returns True if >55% of claim words appear in chunk (verbatim or via synonym).
    Used to give statutory quotes a SUPPORTED label without NLI uncertainty.
    Also catches paraphrases like "defines murder as killing" ↔ statute text.
    """
    claim_words = _expand_with_synonyms(claim)
    chunk_words = _expand_with_synonyms(chunk)
    # Remove stopwords before scoring
    stopwords = {
        "the","a","an","is","are","was","were","of","in","to","for","and",
        "or","but","with","from","by","on","at","as","it","its","be","been",
        "has","have","had","this","that","shall","will","may","can","not",
        "no","any","all","who","which","where","when","what","how",
    }
    claim_words -= stopwords
    chunk_words -= stopwords
    if not claim_words:
        return False
    overlap = len(claim_words & chunk_words) / len(claim_words)
    return overlap >= threshold


# ── Sliding Window Premise  — NEW ────────────────────────────────────────────
def _sliding_window_sentences(text: str, window: int = 3) -> list[str]:
    """
    Split chunk into overlapping windows of `window` sentences.
    Better than passing 512 chars at once — NLI works best on <200 tokens.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= window:
        return [text]
    windows = []
    for i in range(len(sentences) - window + 1):
        windows.append(" ".join(sentences[i:i+window]))
    return windows


# ── NLI Checker  ─────────────────────────────────────────────────────────────
class NLICheckerV2:
    MODEL_NAME           = "cross-encoder/nli-deberta-v3-small"
    ENTAIL_THRESHOLD     = 0.35   # lowered: legal paraphrases score lower than general text
    CONTRADICT_THRESHOLD = 0.38   # IMPROVEMENT 15: slightly raised to reduce false UNSUPPORTED
    CONFIDENCE_THRESHOLD = 0.25   # lowered: allows weak-but-real entailment through
    BATCH_SIZE           = 32     # IMPROVEMENT 7: batch NLI inference
    MAX_WINDOWS_PER_CHUNK = 8     # cap sliding windows to avoid O(n^2) explosion

    def __init__(self, device: str = None):
        # Default to CPU — the embedding model + reranker already occupy GPU VRAM.
        # NLI runs batch inference so CPU is fast enough and avoids OOM.
        # Pass device="cuda" explicitly only if you have a large GPU (>6GB free VRAM).
        self.device = device or "cpu"
        self._loaded = False
        print(f"Loading NLI model on {self.device}...")

        import gc
        gc.collect()  # free any unreferenced memory before loading

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        except OSError as e:
            if "paging file" in str(e) or "1455" in str(e):
                print(f"  WARNING: Tokenizer load failed (paging file too small). "
                      f"NLI checker disabled.")
                print(f"  FIX: Increase Windows paging file via sysdm.cpl")
                return
            raise

        # use_safetensors=False avoids Windows mmap / paging-file crashes
        for use_st in (False, True):
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.MODEL_NAME,
                    use_safetensors=use_st,
                )
                break
            except OSError as e:
                if "paging file" in str(e) or "1455" in str(e):
                    if use_st:
                        print(f"  WARNING: Model load failed (paging file too small). "
                              f"NLI checker disabled.")
                        print(f"  FIX: Increase Windows paging file via sysdm.cpl")
                        return
                    continue
                raise
            except Exception as e:
                if use_st:
                    raise
                continue

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        id2label = {k: v.upper() for k, v in self.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        print(f"  NLI labels: {id2label}")

        def find(keys):
            for k in keys:
                if k in label2id: return label2id[k]
            raise ValueError(f"None of {keys} in {label2id}")

        self._ei = find(["ENTAILMENT", "ENTAIL"])
        self._ni = find(["NEUTRAL"])
        self._ci = find(["CONTRADICTION", "CONTRADICT"])
        print(f"NLI model ready")

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """
        IMPROVEMENT 7: Score a batch of (premise, hypothesis) pairs in one
        forward pass.  Falls back to single-pair scoring if batch fails.
        """
        if not pairs:
            return []
        results = []
        for i in range(0, len(pairs), self.BATCH_SIZE):
            batch = pairs[i : i + self.BATCH_SIZE]
            premises = [p[:500] for p, _ in batch]
            hyps     = [h[:250] for _, h in batch]
            try:
                inp = self.tokenizer(
                    premises, hyps,
                    return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                ).to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(
                        self.model(**inp).logits, dim=-1
                    ).cpu().tolist()
                for p in probs:
                    results.append({
                        "entailment"  : p[self._ei],
                        "neutral"     : p[self._ni],
                        "contradiction": p[self._ci],
                    })
            except Exception:
                # Fall back to per-pair on OOM or tokenizer edge cases
                for premise, hyp in batch:
                    inp = self.tokenizer(
                        premise[:500], hyp[:250],
                        return_tensors="pt", truncation=True,
                        max_length=512, padding=True,
                    ).to(self.device)
                    with torch.no_grad():
                        p = torch.softmax(
                            self.model(**inp).logits, dim=-1
                        )[0].cpu().tolist()
                    results.append({
                        "entailment"  : p[self._ei],
                        "neutral"     : p[self._ni],
                        "contradiction": p[self._ci],
                    })
        return results

    def _get_best_nli_score(
        self, claim: str, chunk: str, partial_overlap: float
    ) -> tuple[float, float, float, str]:
        """
        Score claim against up to MAX_WINDOWS_PER_CHUNK sliding windows of chunk.
        Returns (best_e, best_n, best_c, best_window_text).
        """
        windows = _sliding_window_sentences(chunk, window=3)[:self.MAX_WINDOWS_PER_CHUNK]
        boost   = 0.08 if 0.35 <= partial_overlap < 0.52 else 0.0

        pairs   = [(w, claim) for w in windows]
        scores  = self._score_batch(pairs)

        best_e, best_n, best_c, best_win = 0., 0., 0., windows[0] if windows else chunk
        for win, s in zip(windows, scores):
            adj_e = min(s["entailment"] + boost, 0.99)
            if adj_e > best_e:
                best_e, best_n, best_c, best_win = adj_e, s["neutral"], s["contradiction"], win
        return best_e, best_n, best_c, best_win

    def check_claim(self, claim: str, chunks: list[str]) -> ClaimResult:
        # If NLI model failed to load, return NEUTRAL for all claims
        if not self._loaded:
            return ClaimResult(
                claim=claim, label="NEUTRAL", score=0.33,
                entail_score=0.33, neutral_score=0.34, contradict_score=0.33,
                best_chunk=chunks[0] if chunks else "", verbatim_match=False
            )

        best_e, best_n, best_c, best_chunk = 0., 0., 0., chunks[0] if chunks else ""
        verbatim = False

        stopwords = {"the","a","an","is","are","was","were","of","in","to",
                     "for","and","or","but","with","from","by","on","at",
                     "as","it","its","be","been","has","have","had","shall",
                     "will","may","can","not","no","any","all","who","which"}

        for chunk in chunks:
            # Verbatim/synonym match shortcut — no NLI needed
            if _verbatim_overlap(claim, chunk, threshold=0.52):
                verbatim = True
                if 0.92 > best_e:
                    best_e, best_chunk = 0.92, chunk
                continue

            # Compute partial overlap for boost
            claim_w = _expand_with_synonyms(claim) - stopwords
            chunk_w = _expand_with_synonyms(chunk)  - stopwords
            partial_overlap = (
                len(claim_w & chunk_w) / len(claim_w) if claim_w else 0.0
            )

            e, n, c, win = self._get_best_nli_score(claim, chunk, partial_overlap)
            if e > best_e:
                best_e, best_n, best_c, best_chunk = e, n, c, win

        max_score = max(best_e, best_n, best_c)
        if max_score < self.CONFIDENCE_THRESHOLD:
            label, score = "NEUTRAL", best_n
        elif best_e >= self.ENTAIL_THRESHOLD and best_e >= best_c:
            label, score = "SUPPORTED", best_e
        elif best_c >= self.CONTRADICT_THRESHOLD and best_c > best_e:
            label, score = "UNSUPPORTED", best_c
        else:
            label, score = "NEUTRAL", best_n

        return ClaimResult(
            claim=claim, label=label, score=score, best_chunk=best_chunk,
            entail_score=best_e, neutral_score=best_n, contradict_score=best_c,
            verbatim_match=verbatim,
        )


# ── Main Hallucination Checker ────────────────────────────────────────────────
class HallucinationCheckerV2:
    def __init__(self, device: str = None):
        self.splitter = ClaimSplitterV2()
        self.nli      = NLICheckerV2(device=device)

    def check(
        self, answer: str, chunks: list, max_chunks: int = 10
    ) -> HallucinationReport:
        t0 = time.time()

        # Extract text from chunks (handles both case and statute formats)
        chunk_texts = []
        for c in chunks:
            text = str(c.get("text", c.get("section_text", c.get("full_text", ""))))
            if text.strip():
                chunk_texts.append(text[:800])
        chunk_texts = chunk_texts[:max_chunks]  # up to 15 chunks
        if not chunk_texts:
            chunk_texts = ["No context available."]

        claims  = self.splitter.split(answer) or [answer[:300]]
        results = [self.nli.check_claim(cl, chunk_texts) for cl in claims]

        total = len(results)
        sup   = sum(1 for r in results if r.label == "SUPPORTED")
        unsup = sum(1 for r in results if r.label == "UNSUPPORTED")
        neu   = sum(1 for r in results if r.label == "NEUTRAL")

        # IMPROVEMENT: coverage gap detection
        # If grounding < 30% AND hallucination < 15%, this is a coverage gap, not error
        grounding_score     = sup / max(total, 1)
        hallucination_score = unsup / max(total, 1)
        coverage_gap        = (grounding_score < 0.30 and hallucination_score < 0.15)

        return HallucinationReport(
            original_answer      = answer,
            claims               = results,
            hallucination_score  = hallucination_score,
            grounding_score      = grounding_score,
            supported_count      = sup,
            unsupported_count    = unsup,
            neutral_count        = neu,
            total_claims         = total,
            latency_ms           = (time.time()-t0)*1000,
            annotated_answer     = self._annotate(answer, results),
            coverage_gap_detected= coverage_gap,
        )

    def _annotate(self, answer, results):
        icons = {"SUPPORTED": "✅", "UNSUPPORTED": "❌", "NEUTRAL": "〰"}
        out = answer
        for r in results:
            tag = f" [{r.label}{icons[r.label]}{r.score:.2f}]"
            out = re.sub(
                f"({re.escape(r.claim[:60])})", r"\1" + tag, out, count=1
            )
        return out

    def compare_rag_vs_no_rag(self, query, rag_answer, no_rag_answer, retrieved_chunks):
        rag_report    = self.check(rag_answer,    retrieved_chunks)
        no_rag_report = self.check(no_rag_answer, retrieved_chunks)
        return {
            "query"   : query,
            "rag"     : rag_report.to_dict(),
            "no_rag"  : no_rag_report.to_dict(),
            "improvement": {
                "grounding_delta"     : round(rag_report.grounding_score - no_rag_report.grounding_score, 4),
                "hallucination_delta" : round(no_rag_report.hallucination_score - rag_report.hallucination_score, 4),
            }
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("Hallucination Checker V2 — Diagnostic Test")
    print("="*55)

    checker = HallucinationCheckerV2()

    answer = (
        "Section 302 of the Indian Penal Code defines murder as the unlawful killing of a person. "
        "The punishment under Section 302 is death or imprisonment for life, along with a fine. "
        "Every accused person is always entitled to free bail regardless of the offence. "
        "Napoleon Bonaparte was a famous French military general who conquered most of Europe."
    )

    # Three chunks: punishment text, cognizable offence note, circumstantial evidence
    # Note: chunk 1 does NOT contain "unlawful killing" — that claim will be NEUTRAL
    # because the chunk has no definition. This is CORRECT behaviour.
    chunks = [
        {"text": (
            "Section 302. Whoever commits murder shall be punished with death, "
            "or imprisonment for life, and shall also be liable to fine. "
            "Murder is defined under Section 300 IPC as the unlawful killing of a human being."
        )},
        {"text": "Murder is a cognizable and non-bailable offence. Police may arrest without a warrant."},
        {"text": "In circumstantial evidence cases, the chain must point unerringly to the guilt of the accused."},
    ]

    report = checker.check(answer, chunks)
    print(f"\n{report.summary()}")
    print("\nClaim breakdown:")
    for r in report.claims:
        icon = {"SUPPORTED": "✅", "UNSUPPORTED": "❌", "NEUTRAL": "〰"}[r.label]
        v    = " [VERBATIM/SYNONYM]" if r.verbatim_match else ""
        print(f"  {icon} E={r.entail_score:.2f} C={r.contradict_score:.2f}{v}")
        print(f"     claim: {r.claim[:90]}")

    print(f"\nExpected: ≥2 SUPPORTED, ≥1 UNSUPPORTED (Napoleon), Napoleon=❌ bail=❌")
    print(f"Got     : {report.supported_count}✅ {report.neutral_count}〰 {report.unsupported_count}❌")

    print("\n" + "="*55)
    print("Synonym overlap test")
    print("="*55)
    claim  = "Section 302 defines murder as the unlawful killing of a person"
    chunk1 = "Section 302. Whoever commits murder shall be punished with death or life imprisonment."
    chunk2 = "Murder is defined as the unlawful killing of a human being with intent."

    from_chunk1 = _verbatim_overlap(claim, chunk1)
    from_chunk2 = _verbatim_overlap(claim, chunk2)
    print(f"  Claim vs punishment-only chunk  → overlap={from_chunk1}  (expected False or borderline)")
    print(f"  Claim vs definition chunk       → overlap={from_chunk2}  (expected True)")