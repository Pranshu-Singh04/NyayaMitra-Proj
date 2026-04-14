"""
prompt_builder.py
=================
Prompt construction for NyayaMitra.

v3 IMPROVEMENTS (April 2026):
  - IMPROVEMENT 5 : Chain-of-Thought LJP prompt overhaul (4 explicit steps,
                   mandatory [Case N]/[Statute N] citations, anti-drift guards)
  - IMPROVEMENT 6 : Split LJP intent signals into STRONG + WEAK for better
                   classification accuracy
  - IMPROVEMENT 8 : Per-model context window allocation (MODEL_LIMITS dict)
  - TASK 3        : Few-shot LJP examples prepended for gemini/gpt models

Classes (all public):
    QueryType          ← Enum of query intents
    Prompt             ← dataclass wrapping system + user prompt
    PromptBuilderV2    ← main builder class
    PromptBuilder      ← alias for PromptBuilderV2 (backward compat)
    IntentClassifier   ← classifies free-text query into QueryType

Helper functions:
    format_cases(cases, max_chars_per_case)     → str
    format_statutes(statutes, max_chars_per_statute) → str

Usage:
    from prompt_builder import PromptBuilderV2, IntentClassifier, QueryType
    builder    = PromptBuilderV2(model_type="gemini")
    classifier = IntentClassifier()
    qt         = classifier.classify("Can I get bail for murder?")  # → LEGAL_QA
    prompt     = builder.build_legal_qa(query, cases, statutes)
    llm.generate(prompt.system_prompt, prompt.user_prompt)
"""

from dataclasses import dataclass
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# QUERY TYPE
# ══════════════════════════════════════════════════════════════════════════════
class QueryType(Enum):
    LEGAL_QA       = "legal_qa"
    LJP            = "ljp"           # Legal Judgment Prediction
    STATUTE_LOOKUP = "statute_lookup"
    SUMMARISE      = "summarise"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Prompt:
    system_prompt : str
    user_prompt   : str
    query_type    : QueryType
    context_chars : int = 0      # how many context chars were included

    def to_messages(self) -> list:
        """OpenAI-style messages list."""
        return [
            {"role": "system",  "content": self.system_prompt},
            {"role": "user",    "content": self.user_prompt},
        ]

    def to_single_string(self) -> str:
        """For models that take a single text input (e.g. INLegalLlama)."""
        return f"{self.system_prompt}\n\n{self.user_prompt}"

    def token_estimate(self) -> int:
        """Rough token count (chars / 4)."""
        return (len(self.system_prompt) + len(self.user_prompt)) // 4


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════
def format_cases(cases: list, max_chars_per_case: int = 1800) -> str:
    """Format retrieved case chunks into a numbered context block."""
    if not cases:
        return "No relevant cases retrieved."
    parts = []
    for i, c in enumerate(cases, 1):
        text    = str(c.get("text", c.get("full_text", ""))).strip()
        name    = c.get("case_name",  c.get("filename",  f"Case {i}"))
        court   = c.get("court_level", c.get("court",    "Unknown Court"))
        outcome = c.get("outcome",     "Unknown")
        date    = c.get("date",        "")
        header  = f"[Case {i}] {name} | {court} | Outcome: {outcome}"
        if date:
            header += f" | {date}"
        body    = text[:max_chars_per_case]
        if len(text) > max_chars_per_case:
            body += "..."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def format_statutes(statutes: list, max_chars_per_statute: int = 1200) -> str:
    """Format retrieved statute chunks into a numbered context block."""
    if not statutes:
        return "No relevant statutes retrieved."
    parts = []
    for i, s in enumerate(statutes, 1):
        text    = str(s.get("text", s.get("section_text", ""))).strip()
        source  = s.get("source",       "Unknown")
        num     = s.get("section_num",  "?")
        title   = s.get("section_title","")
        header  = f"[Statute {i}] {source} §{num}"
        if title:
            header += f" — {title}"
        body    = text[:max_chars_per_statute]
        if len(text) > max_chars_per_statute:
            body += "..."
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT LJP EXAMPLES (TASK 3)
# ══════════════════════════════════════════════════════════════════════════════
FEW_SHOT_LJP = """=== FEW-SHOT EXAMPLES (study these to understand the required format) ===

EXAMPLE 1 — Murder appeal based on circumstantial evidence
CASE FACTS: Appellant convicted under Section 302 IPC for murder of his wife.
Conviction based solely on circumstantial evidence: last-seen-together, motive
(marital discord), and recovery of blood-stained clothes. No direct eyewitness.
Appellant appeals against conviction.

STEP 1 — CHARGE IDENTIFICATION:
The charge is murder under Section 302 IPC [Statute 1]. Conviction is based
purely on circumstantial evidence — no direct eyewitness testimony.

STEP 2 — ANALOGOUS CASE ANALYSIS:
[Case 2] involved a circumstantial-evidence murder conviction where the
Supreme Court laid down the "five golden principles" requiring the chain of
circumstances to be complete and point unerringly to guilt. The appeal was
DISMISSED because the chain was complete. [Case 3] had similar last-seen +
motive evidence and the conviction was upheld.

STEP 3 — DISTINGUISHING FACTORS:
The facts here mirror [Case 2] and [Case 3]: last-seen-together is established,
motive is proved, and recovery of blood-stained clothes completes the chain.
There is no break in the chain of circumstances.

STEP 4 — FINAL PREDICTION:
PREDICTION: DISMISSED
CONFIDENCE: HIGH
KEY ISSUES:
- Chain of circumstantial evidence is complete, satisfying the test in [Case 2]
- Motive and recovery corroborate last-seen-together as in [Case 3]
- Section 302 IPC [Statute 1] punishment applies in full
REASONING:
The prosecution's chain of circumstances is unbroken and points unerringly to
the appellant's guilt, as required by [Case 2]. The factual parallel with
[Case 3] is strong. Appeal is likely to be DISMISSED.

---

EXAMPLE 2 — Anticipatory bail for cheating (first offence)
CASE FACTS: Petitioner, a first-time offender, seeks anticipatory bail under
Section 438 CrPC in an FIR under Section 420 IPC (cheating) for an alleged
online fraud of Rs. 2 lakh. Petitioner has cooperated with investigation and
has deep roots in the community.

STEP 1 — CHARGE IDENTIFICATION:
The offence is cheating under Section 420 IPC [Statute 1], which is a bailable
or non-bailable depending on circumstances. Petitioner invokes Section 438
CrPC [Statute 2] for pre-arrest bail.

STEP 2 — ANALOGOUS CASE ANALYSIS:
[Case 1] granted anticipatory bail in a Section 420 matter where the accused
was a first-time offender and had cooperated with the investigation — appeal
ALLOWED. [Case 4] laid down factors for anticipatory bail: nature of
accusation, antecedents, possibility of flight, and cooperation.

STEP 3 — DISTINGUISHING FACTORS:
The petitioner here satisfies all the [Case 4] factors: first-time offender,
full cooperation, deep community roots, fixed-value economic offence. Facts
closely mirror [Case 1].

STEP 4 — FINAL PREDICTION:
PREDICTION: ALLOWED
CONFIDENCE: HIGH
KEY ISSUES:
- First-offender status satisfies [Case 1] precedent
- [Case 4] factors (cooperation, roots, flight risk) all weigh in petitioner's favour
- Section 438 CrPC [Statute 2] permits conditional pre-arrest bail in such cases
REASONING:
The petition tracks the factual matrix in [Case 1] almost exactly, and the
[Case 4] multi-factor test is satisfied. Section 438 CrPC [Statute 2] permits
conditional anticipatory bail here. The application is likely to be ALLOWED
with standard conditions.

=== END OF EXAMPLES — NOW APPLY THE SAME 4-STEP FORMAT TO THE CASE BELOW ===

"""


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER V2
# ══════════════════════════════════════════════════════════════════════════════
class PromptBuilderV2:
    """
    Builds structured prompts for each query type.
    Respects a PER-MODEL context budget so long contexts are trimmed before
    being injected into the prompt.
    """

    # ── IMPROVEMENT 8 : per-model context-character budgets ──────────────────
    MODEL_LIMITS = {
        "gemini"        : 32000,   # Gemini has a 1M context window — be generous
        "gpt"           : 12000,   # GPT-3.5 has 16k total, leave room for output
        "groq"          : 4000,    # llama-3.1-8b has 20k TPM; keep prompts ~2-3k tokens
        "inlegalllama"  : 6000,    # 4k context window, small prompts only
        "mistral"       : 10000,   # 8k context window
    }

    SYSTEM_BASE = (
        "You are NyayaMitra, an AI legal advisor specialising in Indian law. "
        "You have deep knowledge of the Indian Penal Code (IPC), the Bharatiya "
        "Nyaya Sanhita 2023 (BNS), the Code of Criminal Procedure (CrPC), the "
        "Bharatiya Nagarik Suraksha Sanhita (BNSS), and landmark Supreme Court "
        "and High Court judgments. "
        "CRITICAL RULES: "
        "(1) Every factual claim you make MUST be directly grounded in the retrieved cases or statutes provided in the context. "
        "(2) Cite every case as [Case N] and every statute as [Statute N] using the numbered labels in the context. "
        "(3) If the provided context does not contain enough information to answer a part of the question, explicitly say so — do NOT invent facts, section numbers, or case names. "
        "(4) Prefer quoting the exact statutory text over paraphrasing it. "
        "(5) Do NOT bring in external legal knowledge that is not present in the retrieved context. "
        "Be precise, structured, and honest about the limits of the provided context."
    )

    DISCLAIMER = (
        "\n\nDISCLAIMER: This is AI-generated legal information, not professional "
        "legal advice. Please consult a qualified advocate for advice specific to "
        "your situation."
    )

    def __init__(
        self,
        max_context_tokens : int = None,
        model_type         : str = "gemini",
    ):
        """
        Args:
            max_context_tokens: optional override. If None, picks from MODEL_LIMITS.
            model_type        : "gemini" | "gpt" | "inlegalllama" | "mistral"
                                Drives (a) per-model context budget, and
                                (b) whether few-shot examples are prepended to LJP.
        """
        self.model_type = (model_type or "gemini").lower().strip()
        if max_context_tokens is not None:
            self.max_context_chars = max_context_tokens * 4
        else:
            self.max_context_chars = self.MODEL_LIMITS.get(self.model_type, 12000)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _trim_context(self, text: str) -> str:
        if len(text) > self.max_context_chars:
            return text[:self.max_context_chars] + "\n[...context trimmed...]"
        return text

    def _supports_fewshot(self) -> bool:
        """Few-shot examples only fit in large-context models."""
        return self.model_type in ("gemini", "gpt")

    def _is_small_model(self) -> bool:
        """Small/limited-context models that need a shorter, simpler LJP prompt."""
        return self.model_type in ("groq", "inlegalllama")

    # ── Public build methods ──────────────────────────────────────────────────
    def build_legal_qa(
        self, query: str, cases: list, statutes: list
    ) -> Prompt:
        """Build prompt for general legal Q&A."""
        case_ctx    = format_cases(cases)
        statute_ctx = format_statutes(statutes)
        context     = self._trim_context(
            f"RELEVANT CASES:\n{case_ctx}\n\nRELEVANT STATUTES:\n{statute_ctx}"
        )
        user = (
            f"LEGAL QUESTION: {query}\n\n"
            f"=== CONTEXT FROM LEGAL DATABASE (USE ONLY THIS) ===\n{context}\n"
            f"=== END OF CONTEXT ===\n\n"
            "Answer using ONLY the context above. Structure your response as:\n"
            "1. Direct answer to the question\n"
            "2. Relevant statutory provisions — quote the exact text from [Statute N] and state the punishment\n"
            "3. Key precedents — cite as [Case N]: state the court's holding and how it applies here\n"
            "4. Practical advice for the person asking\n"
            "For every claim, cite the [Case N] or [Statute N] it comes from. "
            "If the context does not cover part of the question, say 'The provided context does not address this.'"
            f"{self.DISCLAIMER}"
        )
        return Prompt(
            system_prompt = self.SYSTEM_BASE,
            user_prompt   = user,
            query_type    = QueryType.LEGAL_QA,
            context_chars = len(context),
        )

    def build_ljp(
        self,
        case_facts      : str,
        cases           : list,
        statutes        : list,
        prediction_type : str  = "binary",
        few_shot        : bool = True,
    ) -> Prompt:
        """
        IMPROVEMENT 5 — Chain-of-Thought LJP prompt.
        TASK 3        — Few-shot prepended for gemini/gpt only.

        The model is forced through 4 explicit reasoning steps, and MUST cite
        every factual claim with [Case N] or [Statute N] labels from the
        provided context. No external knowledge, no drift.
        """
        if prediction_type == "ternary":
            pred_options = "ALLOWED / PARTIALLY ALLOWED / DISMISSED"
        else:
            pred_options = "ALLOWED / DISMISSED"

        # ── Simplified prompt for small / low-TPM models (groq, inlegalllama) ─
        # The 4-step CoT + few-shot exceeds their token budget and confuses 8b.
        # Use a short, direct classification prompt instead.
        if self._is_small_model():
            brief_cases    = format_cases(cases[:3], max_chars_per_case=600)
            brief_statutes = format_statutes(statutes[:2], max_chars_per_statute=400)
            user = (
                f"You are a legal judgment predictor for Indian courts.\n\n"
                f"CASE FACTS:\n{case_facts[:1800]}\n\n"
                f"SIMILAR CASES FROM DATABASE:\n{brief_cases}\n\n"
                f"APPLICABLE STATUTES:\n{brief_statutes}\n\n"
                f"Predict the outcome of the case above.\n"
                f"Reply in EXACTLY this format (no other text before or after):\n\n"
                f"PREDICTION: {pred_options}\n"
                f"CONFIDENCE: HIGH or MEDIUM or LOW\n"
                f"REASONING: One sentence explaining your prediction.\n\n"
                f"CRITICAL: The PREDICTION line must contain exactly one of: "
                f"{pred_options}. No other words on that line."
            )
            return Prompt(
                system_prompt = self.SYSTEM_BASE,
                user_prompt   = user,
                query_type    = QueryType.LJP,
                context_chars = len(brief_cases) + len(brief_statutes),
            )

        # ── Full 4-step CoT prompt for large-context models (gemini, gpt) ─────
        case_ctx    = format_cases(cases, max_chars_per_case=1500)
        statute_ctx = format_statutes(statutes, max_chars_per_statute=1000)
        context     = self._trim_context(
            f"ANALOGOUS CASES:\n{case_ctx}\n\nRELEVANT STATUTES:\n{statute_ctx}"
        )

        # ── Few-shot prefix (only for large-context models) ──────────────────
        fewshot_prefix = ""
        if few_shot and self._supports_fewshot():
            fewshot_prefix = FEW_SHOT_LJP

        user = (
            f"{fewshot_prefix}"
            f"CASE FACTS FOR PREDICTION:\n{case_facts}\n\n"
            f"=== RETRIEVED LEGAL CONTEXT (USE ONLY THIS — NO EXTERNAL KNOWLEDGE) ===\n"
            f"{context}\n"
            f"=== END OF CONTEXT ===\n\n"
            f"MANDATORY INSTRUCTIONS — READ CAREFULLY:\n"
            f"You MUST follow the 4-step chain-of-thought below. Every factual\n"
            f"claim MUST cite [Case N] or [Statute N] from the context above.\n"
            f"Do NOT invent cases, statutes, or section numbers. Do NOT rely on\n"
            f"legal knowledge outside the provided context.\n\n"
            f"STEP 1 — CHARGE IDENTIFICATION:\n"
            f"Identify the specific charges/reliefs sought in the case facts.\n"
            f"Name the exact statutory provisions that apply, citing each as\n"
            f"[Statute N] from the context. If the governing statute is NOT in\n"
            f"the context, state: 'Governing statute not in retrieved context.'\n\n"
            f"STEP 2 — ANALOGOUS CASE ANALYSIS:\n"
            f"Identify the 2-3 cases from the context that are factually closest\n"
            f"to the present matter. For each, cite as [Case N], briefly state\n"
            f"its facts, its outcome (ALLOWED/DISMISSED/PARTIALLY ALLOWED), and\n"
            f"WHY it is analogous. Do NOT cite cases not in the context.\n\n"
            f"STEP 3 — DISTINGUISHING FACTORS:\n"
            f"Compare the present facts to the analogous cases from STEP 2. Note\n"
            f"material similarities AND differences. If a distinguishing factor\n"
            f"(e.g. court level, severity, procedural posture) cuts against the\n"
            f"analogy, say so. Cite [Case N] / [Statute N] throughout.\n\n"
            f"STEP 4 — FINAL PREDICTION:\n"
            f"Apply the statutory requirements from STEP 1 and the precedent from\n"
            f"STEP 2 to the STEP 3 analysis. Then output in this EXACT format:\n\n"
            f"PREDICTION: [{pred_options}]\n"
            f"CONFIDENCE: [HIGH/MEDIUM/LOW]\n"
            f"KEY ISSUES:\n"
            f"- [issue 1 — cite [Case N] or [Statute N]]\n"
            f"- [issue 2 — cite [Case N] or [Statute N]]\n"
            f"- [issue 3 — cite [Case N] or [Statute N]]\n"
            f"REASONING:\n"
            f"[Paragraph 1: most analogous case, cite as [Case N], explain relevance]\n"
            f"[Paragraph 2: governing statute, cite as [Statute N], quote exact text]\n"
            f"[Paragraph 3: weigh distinguishing factors, justify final prediction]\n\n"
            f"CRITICAL: The PREDICTION line MUST contain exactly one of: "
            f"{pred_options}. Do NOT hedge, do NOT add qualifications on the\n"
            f"PREDICTION line itself — put hedging in the REASONING section.\n"
            f"{self.DISCLAIMER}"
        )
        return Prompt(
            system_prompt = self.SYSTEM_BASE,
            user_prompt   = user,
            query_type    = QueryType.LJP,
            context_chars = len(context),
        )

    def build_summarise(
        self, document_text: str, mode: str = "detailed"
    ) -> Prompt:
        """Build prompt for document summarisation."""
        trimmed = self._trim_context(document_text)
        if mode == "brief":
            instruction = "Provide a brief 3-5 sentence summary of the key findings and outcome."
        else:
            instruction = (
                "Provide a structured summary covering:\n"
                "1. Facts of the case\n"
                "2. Legal issues raised\n"
                "3. Arguments of each party\n"
                "4. Court's holding and reasoning\n"
                "5. Statutes and precedents cited"
            )
        user = f"DOCUMENT:\n{trimmed}\n\n{instruction}{self.DISCLAIMER}"
        return Prompt(
            system_prompt = self.SYSTEM_BASE,
            user_prompt   = user,
            query_type    = QueryType.SUMMARISE,
            context_chars = len(trimmed),
        )

    def build_statute_lookup(
        self, query: str, statutes: list
    ) -> Prompt:
        """Build prompt for statute lookup queries."""
        statute_ctx = format_statutes(statutes, max_chars_per_statute=1000)
        context     = self._trim_context(statute_ctx)
        user = (
            f"STATUTE QUERY: {query}\n\n"
            f"=== RETRIEVED STATUTE TEXT (USE ONLY THIS) ===\n{context}\n"
            f"=== END OF CONTEXT ===\n\n"
            "Using ONLY the statute text retrieved above, provide:\n"
            "1. Verbatim quote of the exact provision from [Statute N] (copy the text precisely)\n"
            "2. Punishment / legal consequence as stated in [Statute N]\n"
            "3. Key ingredients / elements that must be proven (derived from the statute text)\n"
            "4. Equivalent provision in BNS 2023 (if IPC) or IPC (if BNS) — cite as [Statute N] if present in context\n"
            "5. Important case law interpreting this section — cite as [Case N] if present in context\n"
            "Label every statement with the [Statute N] or [Case N] it comes from. "
            "If the retrieved text does not contain a particular detail, state 'Not covered in retrieved context.'"
            f"{self.DISCLAIMER}"
        )
        return Prompt(
            system_prompt = self.SYSTEM_BASE,
            user_prompt   = user,
            query_type    = QueryType.STATUTE_LOOKUP,
            context_chars = len(context),
        )


# ── backward-compatibility alias ─────────────────────────────────────────────
PromptBuilder = PromptBuilderV2


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER  (IMPROVEMENT 6)
# ══════════════════════════════════════════════════════════════════════════════
class IntentClassifier:
    """
    Rule-based intent classifier.
    Classifies a free-text query into one of the QueryType values.
    No model required.

    IMPROVEMENT 6: LJP signals split into STRONG (any 1 triggers) + WEAK
    (need 3+ to trigger). Previously a single list with threshold >=2 caused
    generic legal QA queries containing words like "accused" to be
    misclassified as LJP.
    """

    # Strong LJP signals — a single hit is enough to classify as LJP
    LJP_STRONG_SIGNALS = [
        "predict the outcome", "will the appeal", "chance of winning",
        "likelihood of bail", "accused charged with", "petitioner seeks bail",
        "anticipatory bail application", "appeal against conviction",
        "first offence", "prior criminal record", "circumstantial evidence",
        "sessions court", "district court hearing",
    ]

    # Weak LJP signals — need 3+ hits to classify as LJP
    LJP_WEAK_SIGNALS = [
        "accused", "charged", "bail application", "predict", "outcome",
        "will i win", "chances", "first offence", "prior record", "evidence is",
    ]

    STATUTE_SIGNALS = [
        "what is section", "explain section", "define section",
        "what does section", "ipc section", "bns section",
        "what is article", "what is rule", "punishment for",
        "penalty under", "text of", "provision of",
        "what is §", "sec.", "what does ipc say",
    ]
    SUMMARISE_SIGNALS = [
        "summarise", "summarize", "summary of", "brief of",
        "explain this judgment", "what happened in", "key points of",
    ]

    def classify(self, query: str) -> QueryType:
        q = query.lower().strip()

        if any(sig in q for sig in self.SUMMARISE_SIGNALS):
            return QueryType.SUMMARISE

        if any(sig in q for sig in self.STATUTE_SIGNALS):
            return QueryType.STATUTE_LOOKUP

        # LJP: strong signals (any 1) OR weak signals (3+)
        strong_hits = sum(1 for sig in self.LJP_STRONG_SIGNALS if sig in q)
        weak_hits   = sum(1 for sig in self.LJP_WEAK_SIGNALS   if sig in q)
        if strong_hits >= 1 or weak_hits >= 3:
            return QueryType.LJP

        return QueryType.LEGAL_QA
