"""
hybrid_retriever.py
===================
Hybrid retriever for NyayaMitra.
Combines dense FAISS search (E5-large-v2) + sparse BM25,
with RRF fusion, MMR diversity selection, query expansion,
contextual compression, and cross-encoder reranking.

Classes (all public):
    BM25
    QueryExpander
    ContextualCompressor
    HybridRetrieverV2         ← main class
    HybridRetriever           ← alias for HybridRetrieverV2 (backward compat)

Usage:
    from hybrid_retriever import HybridRetrieverV2
    r = HybridRetrieverV2(index_dir="indexes")
    results = r.retrieve("Can I get bail for murder?", top_k=5)
    # returns {"query": ..., "cases": [...], "statutes": [...]}
"""

import json
import math
import re
import argparse
import numpy as np
import faiss
import torch
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder


# ══════════════════════════════════════════════════════════════════════════════
# BM25  (pure-Python, no external deps)
# ══════════════════════════════════════════════════════════════════════════════
class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.tokenised   = [self._tokenise(d) for d in corpus]
        lengths          = [len(d) for d in self.tokenised]
        self.avg_dl      = sum(lengths) / max(len(lengths), 1)
        self.df: dict[str, int] = defaultdict(int)
        for doc in self.tokenised:
            for term in set(doc):
                self.df[term] += 1

    def _tokenise(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def get_scores(self, query: str) -> np.ndarray:
        terms  = self._tokenise(query)
        scores = np.zeros(self.corpus_size)
        for term in terms:
            if term not in self.df:
                continue
            idf = math.log((self.corpus_size - self.df[term] + 0.5) /
                           (self.df[term] + 0.5) + 1)
            for i, doc in enumerate(self.tokenised):
                tf = doc.count(term)
                dl = len(doc)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * num / den
        return scores

    def get_top_k(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores  = self.get_scores(query)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANDER  (IMPROVEMENT 2 — tripled dictionary)
# ══════════════════════════════════════════════════════════════════════════════
class QueryExpander:
    """
    Expands legal queries with synonyms and related terms.
    Rule-based — no model required.
    """
    EXPANSIONS: dict[str, list[str]] = {
        # ── Original entries ──────────────────────────────────────────────────
        "murder"        : ["homicide", "Section 302", "culpable homicide", "killing", "IPC 302"],
        "bail"          : ["anticipatory bail", "regular bail", "interim bail", "BNSS 480", "CrPC 437", "Section 439"],
        "theft"         : ["Section 379", "stealing", "robbery", "dacoity", "IPC 379"],
        "rape"          : ["Section 376", "sexual assault", "POCSO", "Section 354", "IPC 376"],
        "cheating"      : ["Section 420", "fraud", "dishonest", "BNS 318", "IPC 420"],
        "domestic"      : ["Section 498A", "cruelty", "dowry", "harassment", "IPC 498A"],
        "fir"           : ["first information report", "police complaint", "cognizable", "Section 154 CrPC"],
        "arrest"        : ["custody", "detention", "remand", "warrant", "Section 41 CrPC"],
        "appeal"        : ["revision", "high court", "supreme court", "challenge", "Section 374"],
        "evidence"      : ["circumstantial", "witness", "testimony", "proof", "Section 3 Evidence Act"],
        "ipc"           : ["Indian Penal Code", "BNS", "Bharatiya Nyaya Sanhita"],
        "bns"           : ["Bharatiya Nyaya Sanhita 2023", "IPC", "Indian Penal Code"],
        "punishment"    : ["sentence", "imprisonment", "fine", "death penalty", "rigorous imprisonment"],
        "cognizable"    : ["non-bailable", "police arrest", "FIR", "warrant", "schedule I CrPC"],
        "conviction"    : ["guilty", "sentenced", "Section 235", "found guilty", "beyond reasonable doubt"],
        "dowry"         : ["Section 304B", "dowry death", "498A", "cruelty", "harassment"],
        "cruelty"       : ["Section 498A", "domestic violence", "mental cruelty", "dowry harassment"],
        "section 302"   : ["murder", "IPC 302", "culpable homicide", "death sentence", "life imprisonment"],
        "section 420"   : ["cheating", "fraud", "IPC 420", "dishonest inducement", "BNS 318"],
        "section 376"   : ["rape", "sexual assault", "IPC 376", "consent", "POCSO"],
        "section 498a"  : ["cruelty", "domestic violence", "dowry", "harassment", "IPC 498A"],
        "habeas corpus" : ["illegal detention", "writ petition", "Article 226", "Article 32", "custody"],
        "anticipatory"  : ["Section 438", "pre-arrest bail", "anticipatory bail", "CrPC 438", "BNSS 484"],
        "crpc"          : ["CrPC", "Criminal Procedure Code", "BNSS", "Bharatiya Nagarik Suraksha Sanhita"],
        "bnss"          : ["BNSS", "Bharatiya Nagarik Suraksha Sanhita", "CrPC", "Criminal Procedure Code"],

        # ── IMPROVEMENT 2 additions: Tripled dictionary ───────────────────────
        "abetment"         : ["Section 107", "Section 109", "IPC 107", "instigate", "aid"],
        "attempt"          : ["Section 511", "IPC 511", "incomplete offence", "preparation"],
        "kidnapping"       : ["Section 359", "Section 363", "IPC 363", "abduction", "Section 362"],
        "extortion"        : ["Section 383", "Section 384", "IPC 384", "threat", "wrongful gain"],
        "forgery"          : ["Section 463", "Section 465", "IPC 465", "false document"],
        "assault"          : ["Section 351", "Section 352", "IPC 352", "criminal force"],
        "defamation"       : ["Section 499", "Section 500", "IPC 499", "reputation", "libel"],
        "trespass"         : ["Section 441", "Section 442", "IPC 441", "criminal trespass"],
        "mischief"         : ["Section 425", "Section 426", "IPC 425", "damage property"],
        "affray"           : ["Section 159", "Section 160", "IPC 159", "public peace"],
        "rioting"          : ["Section 146", "Section 147", "IPC 147", "unlawful assembly"],
        "sedition"         : ["Section 124A", "IPC 124A", "BNS 152", "disaffection"],
        "writ"             : ["Article 32", "Article 226", "habeas corpus", "mandamus", "certiorari"],
        "juvenile"         : ["POCSO", "JJ Act", "child", "minor", "Section 376AB"],
        "sc/st"            : ["Atrocities Act", "Section 3", "SC ST Prevention", "scheduled caste"],
        "narcotics"        : ["NDPS Act", "Section 20", "Section 21", "drug trafficking"],
        "money laundering" : ["PMLA", "proceeds of crime", "attachment", "Section 3 PMLA"],
        "cybercrime"       : ["IT Act", "Section 66", "Section 67", "hacking", "phishing"],
        "insolvency"       : ["IBC", "CIRP", "Section 7", "corporate debtor", "resolution plan"],
        "acquittal"        : ["benefit of doubt", "Section 232 CrPC", "not guilty", "discharge", "Section 227"],
        "life imprisonment": ["Section 302", "Section 376A", "death sentence", "capital punishment"],
        "section 304b"     : ["dowry death", "Section 113B Evidence Act", "presumption", "within 7 years"],
        "section 307"      : ["attempt to murder", "IPC 307", "grievous hurt", "intention to kill"],
        "section 323"      : ["voluntarily causing hurt", "IPC 323", "simple hurt", "Section 324"],
        "appeal allowed"   : ["ALLOWED", "set aside", "reversed", "quashed"],
        "appeal dismissed" : ["DISMISSED", "upheld", "confirmed", "affirmed"],

        # ── BNSS 2023 entries (TASK B2) ──────────────────────────────────────
        "bnss bail"         : ["Section 480 BNSS", "Section 482 BNSS",
                               "Section 483 BNSS", "anticipatory bail BNSS"],
        "bnss arrest"       : ["Section 35 BNSS", "Section 38 BNSS",
                               "Section 43 BNSS", "CrPC Section 41"],
        "bnss fir"          : ["Section 173 BNSS", "cognizable offence BNSS",
                               "police report BNSS"],
        "bnss remand"       : ["Section 187 BNSS", "Section 167 CrPC",
                               "24 hours custody"],
        "bnss 2023"         : ["Bharatiya Nagarik Suraksha Sanhita",
                               "criminal procedure", "CrPC replacement"],
        "section 480 bnss"  : ["bail bailable offence", "CrPC 437",
                               "released on bail"],
        "section 482 bnss"  : ["anticipatory bail", "apprehending arrest",
                               "CrPC 438", "High Court Session Court bail"],
        "section 483 bnss"  : ["High Court bail powers", "CrPC 439",
                               "special bail powers"],
    }

    def expand(self, query: str, max_expansions: int = 6) -> list[str]:
        """Return a list of query variants (original + expanded)."""
        queries  = [query]
        q_lower  = query.lower()
        added    = 0
        # Sort keywords by length desc so multi-word matches come first
        sorted_keys = sorted(self.EXPANSIONS.keys(), key=len, reverse=True)
        for keyword in sorted_keys:
            if keyword in q_lower and added < max_expansions:
                expansions = self.EXPANSIONS[keyword]
                expanded = query + " " + " ".join(expansions[:2])
                queries.append(expanded)
                added += 1
        return queries


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL COMPRESSOR
# ══════════════════════════════════════════════════════════════════════════════
class ContextualCompressor:
    """
    Reduces retrieved chunk to the most query-relevant sentences.
    Keeps top-N sentences by keyword overlap.
    """
    def compress(self, query: str, text: str, max_sentences: int = 3) -> str:
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= max_sentences:
            return text
        def score(s):
            words = set(re.findall(r'\b\w+\b', s.lower()))
            return len(words & query_terms)
        ranked = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
        top    = sorted(ranked[:max_sentences], key=lambda x: x[0])
        return " ".join(s for _, s in top)


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER V2
# ══════════════════════════════════════════════════════════════════════════════
class HybridRetrieverV2:
    """
    Hybrid retriever combining dense (FAISS/E5) + sparse (BM25).

    Pipeline per query:
      1. Optionally expand query with legal synonyms
      2. Dense search (FAISS inner product)
      3. Sparse search (BM25)
      4. RRF fusion with ADAPTIVE dense/sparse weights (IMPROVEMENT 3)
      5. Metadata boost (court level, recency, outcome alignment — IMPROVEMENT 4)
      6. Cross-encoder reranking (IMPROVEMENT 1 — default ON)
      7. MMR diversity selection with task-specific lambda (IMPROVEMENT 14)
      8. Optional contextual compression of selected chunks
    """

    # IMPROVEMENT 11 — CrPC/BNSS topics for coverage gap detection
    CRPC_BNSS_TOPICS = [
        # existing CrPC terms
        "crpc", "section 154", "section 161", "section 41",
        "section 437", "section 438", "section 439", "fir procedure",
        "arrest without warrant", "chargesheet", "remand", "section 167",
        "cognizable offence definition", "search warrant", "section 93",
        # BNSS 2023 terms (TASK B3)
        "bnss", "bnss 2023", "section 480 bnss", "section 482 bnss",
        "section 483 bnss", "section 173 bnss", "section 35 bnss",
        "section 187 bnss", "nagarik suraksha", "bharatiya nagarik",
        "bail conditions bnss", "bnss bail",
    ]

    def __init__(
        self,
        index_dir          : str | Path = "indexes",
        dense_weight       : float = 0.6,
        sparse_weight      : float = 0.4,
        top_k_dense        : int   = 80,     # IMPROVEMENT 9: 50 → 80
        top_k_sparse       : int   = 80,     # IMPROVEMENT 9: 50 → 80
        use_reranker       : bool  = True,   # IMPROVEMENT 1: default ON
        use_compression    : bool  = True,
        use_query_expansion: bool  = True,
        mmr_lambda         : float = 0.72,
        rerank_threshold   : float = -5.0,   # IMPROVEMENT 1: score floor
        rerank_fallback_threshold: float = -8.0,
    ):
        self.index_dir           = Path(index_dir)
        self.dense_weight        = dense_weight
        self.sparse_weight       = sparse_weight
        self.top_k_dense         = top_k_dense
        self.top_k_sparse        = top_k_sparse
        self.use_reranker        = use_reranker
        self.use_compression     = use_compression
        self.use_query_expansion = use_query_expansion
        self.mmr_lambda          = mmr_lambda
        self.rerank_threshold    = rerank_threshold
        self.rerank_fallback_threshold = rerank_fallback_threshold
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker            = None

        self._load_config()
        self._load_model()
        self._load_indexes()
        self._build_bm25()
        self.expander   = QueryExpander()
        self.compressor = ContextualCompressor()

        if use_reranker:
            self._load_reranker()

    # ── setup ─────────────────────────────────────────────────────────────────
    def _load_config(self):
        config_path = self.index_dir / "index_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No index_config.json in {self.index_dir}")
        self.config       = json.loads(config_path.read_text())
        self.model_name   = self.config["model_name"]
        self.query_prefix = self.config.get("query_prefix", "query: ")
        self.normalize    = self.config.get("normalize", True)
        print(f"Config loaded | Model: {self.model_name}")

    def _load_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model on {self.device}...")

        # Check free VRAM before attempting CUDA.
        # E5-large-v2 in float32 needs ~1.5GB. Only try GPU if enough is free.
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # GB
            use_cuda  = free_vram >= 1.8
            if not use_cuda:
                print(f"  Only {free_vram:.1f}GB VRAM free — loading on CPU instead.")
        else:
            use_cuda = False

        strategies = []
        if use_cuda:
            strategies.append({"device": "cuda", "model_kwargs": {"use_safetensors": False}})
        strategies += [
            {"device": "cpu", "model_kwargs": {"use_safetensors": False}},
            {"device": "cpu", "model_kwargs": {}},
        ]
        last_err = None
        for kwargs in strategies:
            try:
                self.device = kwargs["device"]
                self.model  = SentenceTransformer(
                    self.model_name,
                    device=kwargs["device"],
                    model_kwargs=kwargs["model_kwargs"],
                )
                print(f"  Embedding model loaded on {self.device}")
                return
            except (OSError, RuntimeError, Exception) as e:
                if "paging file" in str(e) or "1455" in str(e) or "memory" in str(e).lower():
                    print(f"  Strategy {kwargs} failed: {e.__class__.__name__}. Trying next...")
                    last_err = e
                    continue
                raise   # non-memory error — don't swallow it
        raise RuntimeError(
            f"Could not load embedding model after all fallbacks. "
            f"Last error: {last_err}\n"
            f"Fix: increase Windows paging file (sysdm.cpl → Advanced → Performance → "
            f"Virtual Memory → set to 8192–16384 MB on C:)"
        )

    def _load_reranker(self):
        try:
            print(f"Loading cross-encoder reranker ({self.reranker_model_name})...")
            # Always load reranker on CPU — it only scores ~80 candidates per query
            # so CPU is fast enough, and this keeps VRAM free for the embedding model.
            self.reranker = CrossEncoder(self.reranker_model_name, device="cpu")
            print("Reranker ready")
        except Exception as e:
            print(f"  Warning: reranker failed to load ({e}). Disabling reranker.")
            self.reranker     = None
            self.use_reranker = False

    def _load_jsonl(self, path: Path) -> list[dict]:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_indexes(self):
        case_idx_path     = self.index_dir / "faiss_cases.index"
        statute_idx_path  = self.index_dir / "faiss_statutes.index"
        case_meta_path    = self.index_dir / "faiss_cases_metadata.jsonl"
        statute_meta_path = self.index_dir / "faiss_statutes_metadata.jsonl"

        self.case_index    = faiss.read_index(str(case_idx_path))
        self.statute_index = faiss.read_index(str(statute_idx_path))
        self.case_meta     = self._load_jsonl(case_meta_path)
        self.statute_meta  = self._load_jsonl(statute_meta_path)

        print(f"Indexes loaded | Cases: {self.case_index.ntotal:,} | Statutes: {self.statute_index.ntotal:,}")

    def _build_bm25(self):
        # BM25 is only built for statutes (1,040 docs — fast, <1s).
        # Cases (100,000 docs) would require ~4GB RAM to tokenize and cause
        # system freezes on Windows. Case retrieval uses dense-only (FAISS).
        # The cross-encoder reranker compensates for the missing BM25 on cases.
        print(f"Building BM25 index for statutes ({len(self.statute_meta):,} docs)...", flush=True)
        try:
            statute_texts     = [m.get("text", "") for m in self.statute_meta]
            self.case_bm25    = None   # dense-only for cases
            self.statute_bm25 = BM25(statute_texts)
            self._bm25_available = True
            print("BM25 ready (statutes only — cases use dense retrieval)", flush=True)
        except MemoryError:
            print("  WARNING: Not enough RAM even for statute BM25 — using dense-only retrieval.")
            self.case_bm25    = None
            self.statute_bm25 = None
            self._bm25_available = False

    # ── search primitives ─────────────────────────────────────────────────────
    def _embed_query(self, query: str) -> np.ndarray:
        text = f"{self.query_prefix}{query}" if self.query_prefix else query
        try:
            with torch.no_grad():
                emb = self.model.encode(
                    [text], normalize_embeddings=self.normalize,
                    convert_to_numpy=True, device=self.device,
                )
            return emb.astype("float32")
        except RuntimeError as e:
            if self.device == "cuda":
                # CUDA failed mid-run (OOM or allocation failure after other models loaded).
                # Move to CPU permanently for this session and retry.
                print(f"  CUDA encode failed ({e}). Moving embedding model to CPU...")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = self.model.to("cpu")
                with torch.no_grad():
                    emb = self.model.encode(
                        [text], normalize_embeddings=self.normalize,
                        convert_to_numpy=True, device="cpu",
                    )
                print("  Embedding model now running on CPU.")
                return emb.astype("float32")
            raise

    def _dense_search(self, query_emb: np.ndarray, index: faiss.Index, k: int):
        scores, idxs = index.search(query_emb, k)
        return list(zip(idxs[0].tolist(), scores[0].tolist()))

    # ── IMPROVEMENT 3: Adaptive RRF weights ──────────────────────────────────
    def _get_adaptive_weights(self, query: str) -> tuple[float, float]:
        """
        Returns (dense_weight, sparse_weight) based on query characteristics.

        - Queries with exact section numbers or act names → BM25 (sparse) wins
          because exact identifiers are high-IDF sparse signals.
        - Semantic queries → dense wins because word overlap is weak.
        """
        q = query.lower()
        has_section_num = bool(re.search(
            r'section\s+\d+|ipc\s+\d+|bns\s+\d+|§\s*\d+|\barticle\s+\d+|\b\d{3}[a-z]?\b', q
        ))
        has_act_name = any(k in q for k in [
            "crpc", "bnss", "ipc", "bns", "pocso", "ndps", "pmla", "ibc", "it act"
        ])
        if has_section_num or has_act_name:
            return 0.4, 0.6   # sparse-heavy for identifier queries
        else:
            return 0.75, 0.25  # dense-heavy for semantic queries

    def _rrf_fusion(
        self,
        dense_results : list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        dense_weight  : float = None,
        sparse_weight : float = None,
        k_rrf         : int = 60,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion with optional per-query weights."""
        dw = dense_weight  if dense_weight  is not None else self.dense_weight
        sw = sparse_weight if sparse_weight is not None else self.sparse_weight

        scores: dict[int, float] = defaultdict(float)
        for rank, (idx, _) in enumerate(dense_results):
            if idx >= 0:
                scores[idx] += dw * 1.0 / (k_rrf + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            if idx >= 0:
                scores[idx] += sw * 1.0 / (k_rrf + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── IMPROVEMENT 4: Enhanced metadata boost ───────────────────────────────
    def _apply_metadata_boost(
        self,
        fused   : list[tuple[int, float]],
        metadata: list[dict],
        query   : str = "",
    ) -> list[tuple[int, float]]:
        """
        Apply multiplicative boosts based on:
          - Court level (Supreme ×1.25, High ×1.10)
          - Recency     (2015+ ×1.15, 2010-14 ×1.05)
          - Outcome-query alignment (bail granted for bail queries, etc.)
          - Statute section number exact match (×1.50)
        """
        q_lower = query.lower()
        boosted = []
        for idx, score in fused:
            if idx < 0 or idx >= len(metadata):
                continue
            meta  = metadata[idx]
            boost = 1.0

            # Court level
            court = str(meta.get("court_level", meta.get("court", ""))).lower()
            if "supreme" in court:
                boost *= 1.25
            elif "high" in court:
                boost *= 1.10

            # Recency
            date_str = str(meta.get("date", meta.get("year", "")))
            yr_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if yr_match:
                year = int(yr_match.group())
                if year >= 2015:
                    boost *= 1.15
                elif year >= 2010:
                    boost *= 1.05

            # Outcome-query alignment
            outcome = str(meta.get("outcome", "")).lower()
            if "bail" in q_lower and "granted" in outcome:
                boost *= 1.20
            if any(w in q_lower for w in ["murder", "conviction", "convicted"]) and \
               any(w in outcome for w in ["convicted", "dismissed", "rejected"]):
                boost *= 1.15

            # Statute section number exact match
            sec_num = str(meta.get("section_num", "")).lower().strip()
            if sec_num and sec_num != "?" and sec_num in q_lower:
                boost *= 1.50

            boosted.append((idx, score * boost))

        return sorted(boosted, key=lambda x: x[1], reverse=True)

    # ── IMPROVEMENT 14: Task-specific MMR lambda ─────────────────────────────
    def _get_mmr_lambda(self, query_type: str = None) -> float:
        """Relevance vs diversity trade-off, tuned per task type."""
        if query_type is None:
            return self.mmr_lambda
        return {
            "ljp"     : 0.85,   # prefer relevance — want most similar cases
            "statute" : 0.95,   # maximum relevance — exact statute
            "qa"      : 0.72,   # balanced
            "summarise": 0.70,  # slight diversity ok
        }.get(query_type, self.mmr_lambda)

    def _mmr_select(
        self,
        query_emb   : np.ndarray,
        candidates  : list[tuple[int, float]],
        metadata    : list[dict],
        top_k       : int,
        lambda_     : float = None,
    ) -> list[dict]:
        """
        Maximal Marginal Relevance: balance relevance vs. diversity.
        lambda_=1.0 → pure relevance, 0.0 → pure diversity.
        """
        lam       = lambda_ if lambda_ is not None else self.mmr_lambda
        cand_idxs = [idx for idx, _ in candidates if 0 <= idx < len(metadata)]
        if not cand_idxs:
            return []

        texts = [
            f"{self.query_prefix}{metadata[i].get('text','')[:300]}"
            if self.query_prefix
            else metadata[i].get("text", "")[:300]
            for i in cand_idxs
        ]
        with torch.no_grad():
            embs = self.model.encode(
                texts, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")

        q      = query_emb[0]
        rel    = (embs @ q).tolist()
        selected_embs : list[np.ndarray] = []
        selected_meta : list[dict]       = []
        remaining     : list[int]        = list(range(len(cand_idxs)))

        while len(selected_meta) < top_k and remaining:
            if not selected_embs:
                best = max(remaining, key=lambda i: rel[i])
            else:
                sel_stack = np.vstack(selected_embs)
                def mmr_score(i):
                    sim_to_query    = rel[i]
                    sim_to_selected = float(np.max(embs[i] @ sel_stack.T))
                    return lam * sim_to_query - (1 - lam) * sim_to_selected
                best = max(remaining, key=mmr_score)

            selected_embs.append(embs[best])
            meta_item = dict(metadata[cand_idxs[best]])
            if self.use_compression and selected_meta:
                meta_item["text"] = self.compressor.compress(
                    texts[0], meta_item.get("text", ""), max_sentences=5
                )
            selected_meta.append(meta_item)
            remaining.remove(best)

        return selected_meta

    # ── IMPROVEMENT 1: Cross-encoder reranker with threshold ─────────────────
    def _rerank(
        self,
        query   : str,
        fused   : list[tuple[int, float]],
        metadata: list[dict],
        top_n   : int,
    ) -> list[tuple[int, float]]:
        """
        Cross-encoder reranking.
        - Scores top_n RRF candidates with ms-marco cross-encoder
        - Applies score threshold (> -5.0) to filter clearly irrelevant chunks
        - Falls back to relaxed threshold (-8.0) if too few pass
        """
        if self.reranker is None:
            return fused

        candidates = [(idx, s) for idx, s in fused if 0 <= idx < len(metadata)][:top_n]
        if not candidates:
            return fused

        pairs  = [(query, metadata[idx].get("text", "")[:400]) for idx, _ in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(
            zip([idx for idx, _ in candidates], scores.tolist()),
            key=lambda x: x[1], reverse=True
        )

        # Apply score threshold
        above_threshold = [(idx, s) for idx, s in reranked if s > self.rerank_threshold]
        min_required    = min(3, top_n // 4)

        if len(above_threshold) < min_required:
            # Fallback: relax threshold
            above_threshold = [
                (idx, s) for idx, s in reranked if s > self.rerank_fallback_threshold
            ]
            if not above_threshold:
                above_threshold = reranked[:min_required]

        return above_threshold

    def _multi_query_search(
        self,
        query      : str,
        index      : faiss.Index,
        bm25       : BM25,
        k_dense    : int,
        k_sparse   : int,
    ) -> tuple[list, list]:
        """Search with original + optionally expanded queries; merge results."""
        queries = self.expander.expand(query) if self.use_query_expansion else [query]

        all_dense : dict[int, float] = {}
        all_sparse: dict[int, float] = {}

        for q in queries:
            q_emb = self._embed_query(q)
            for idx, score in self._dense_search(q_emb, index, k_dense):
                if idx >= 0:
                    all_dense[idx] = max(all_dense.get(idx, 0.0), float(score))
            if bm25 is not None:
                for idx, score in bm25.get_top_k(q, k=k_sparse):
                    all_sparse[idx] = max(all_sparse.get(idx, 0.0), score)

        dense_sorted  = sorted(all_dense.items(),  key=lambda x: x[1], reverse=True)
        sparse_sorted = sorted(all_sparse.items(), key=lambda x: x[1], reverse=True)
        return dense_sorted, sparse_sorted

    # ── IMPROVEMENT 11: Coverage gap detection ───────────────────────────────
    def _detect_coverage_gap(self, query: str, num_results: int) -> bool:
        """
        Returns True if query touches CrPC/BNSS procedural law AND fewer than
        3 relevant chunks were retrieved. Signals dataset coverage limitations.
        """
        q = query.lower()
        is_procedural = any(t in q for t in self.CRPC_BNSS_TOPICS)
        return is_procedural and num_results < 3

    # ── public retrieval API ──────────────────────────────────────────────────
    def retrieve(
        self,
        query          : str,
        top_k          : int  = 5,
        search_cases   : bool = True,
        search_statutes: bool = True,
        query_type     : str  = None,   # "ljp" | "statute" | "qa" | "summarise"
    ) -> dict:
        """
        Main retrieval method.
        Returns {"query": str, "cases": [...], "statutes": [...],
                 "coverage_gap_warning": bool}
        """
        results = {
            "query": query,
            "cases": [],
            "statutes": [],
            "coverage_gap_warning": False,
        }

        # IMPROVEMENT 3: Get adaptive weights once per query
        dw, sw = self._get_adaptive_weights(query)

        # ── Cases ─────────────────────────────────────────────────────────────
        if search_cases and self.case_index.ntotal > 0:
            dense_c, sparse_c = self._multi_query_search(
                query, self.case_index, self.case_bm25,
                self.top_k_dense, self.top_k_sparse
            )
            fused_c = self._rrf_fusion(dense_c, sparse_c,
                                        dense_weight=dw, sparse_weight=sw)
            fused_c = self._apply_metadata_boost(fused_c, self.case_meta, query=query)

            # IMPROVEMENT 1: Cross-encoder reranking on top-N candidates
            if self.use_reranker:
                fused_c = self._rerank(query, fused_c, self.case_meta, top_n=top_k * 6)

            q_emb = self._embed_query(query)
            lam   = self._get_mmr_lambda(query_type)
            results["cases"] = self._mmr_select(
                q_emb, fused_c, self.case_meta, top_k, lambda_=lam
            )

        # ── Statutes ──────────────────────────────────────────────────────────
        if search_statutes and self.statute_index.ntotal > 0:
            dense_s, sparse_s = self._multi_query_search(
                query, self.statute_index, self.statute_bm25,
                self.top_k_dense, self.top_k_sparse
            )
            fused_s = self._rrf_fusion(dense_s, sparse_s,
                                        dense_weight=dw, sparse_weight=sw)
            fused_s = self._apply_metadata_boost(fused_s, self.statute_meta, query=query)

            if self.use_reranker:
                fused_s = self._rerank(query, fused_s, self.statute_meta, top_n=top_k * 4)

            q_emb = self._embed_query(query)
            lam   = self._get_mmr_lambda(query_type)
            results["statutes"] = self._mmr_select(
                q_emb, fused_s, self.statute_meta, top_k, lambda_=lam
            )

        # IMPROVEMENT 11: Coverage gap detection
        total = len(results["cases"]) + len(results["statutes"])
        results["coverage_gap_warning"] = self._detect_coverage_gap(query, total)

        return results

    # ── TASK 4: LJP-specific retrieval ───────────────────────────────────────
    def retrieve_for_ljp(self, case_facts: str, top_k: int = 10) -> dict:
        """
        Specialised retrieval for Judgment Prediction.
        - Retrieves 2x cases (top_k=10 default) — LJP needs more analogous cases
        - Filters cases to those with KNOWN outcomes (not 'Unknown')
        - Applies court-level matching boost
        - Uses LJP-specific MMR lambda (0.85 — prefer relevance)
        """
        results = self.retrieve(
            case_facts, top_k=top_k,
            search_cases=True, search_statutes=True,
            query_type="ljp",
        )

        # Filter out cases with unknown outcomes
        known_outcomes = [
            c for c in results["cases"]
            if str(c.get("outcome", "Unknown")).lower() not in ("unknown", "", "?", "none")
        ]
        if len(known_outcomes) >= 3:
            results["cases"] = known_outcomes

        # Court-level matching boost: if query mentions a court level,
        # re-sort so matching-court cases come first.
        facts_lower = case_facts.lower()
        court_hint = None
        if "supreme court" in facts_lower or "sci" in facts_lower:
            court_hint = "supreme"
        elif "high court" in facts_lower:
            court_hint = "high"
        elif "sessions court" in facts_lower or "session court" in facts_lower:
            court_hint = "sessions"
        elif "district court" in facts_lower:
            court_hint = "district"

        if court_hint and results["cases"]:
            def court_match_key(c):
                court = str(c.get("court_level", c.get("court", ""))).lower()
                return 0 if court_hint in court else 1
            results["cases"].sort(key=court_match_key)

        # Keep top_k cases for cases, top_k/2 for statutes
        results["cases"]    = results["cases"][:top_k]
        results["statutes"] = results["statutes"][:max(5, top_k // 2)]
        return results

    # ── IMPROVEMENT 9: Double statute retrieval for statute lookups ──────────
    def retrieve_statutes_only(self, query: str, top_k: int = 5) -> dict:
        """Retrieve only statute chunks (for statute lookup queries)."""
        return self.retrieve(
            query, top_k=top_k * 2,   # double for statute-only queries
            search_cases=False, search_statutes=True,
            query_type="statute",
        )


# ── backward-compatibility alias ─────────────────────────────────────────────
HybridRetriever = HybridRetrieverV2


# ══════════════════════════════════════════════════════════════════════════════
# CLI  (python hybrid_retriever.py --query "murder bail")
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Test HybridRetrieverV2")
    parser.add_argument("--index_dir", default="indexes")
    parser.add_argument("--query",     default="murder Section 302 bail application")
    parser.add_argument("--top_k",     type=int, default=3)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    r       = HybridRetrieverV2(
        index_dir=args.index_dir,
        use_reranker=not args.no_reranker,
    )
    results = r.retrieve(args.query, top_k=args.top_k)

    print(f"\n-- CASES ({len(results['cases'])}) --")
    for i, c in enumerate(results["cases"], 1):
        print(f"  {i}. {c.get('case_name','?')[:55]} | {c.get('court','?')} | {c.get('outcome','?')}")
        print(f"     {c.get('text','')[:120]}...")

    print(f"\n-- STATUTES ({len(results['statutes'])}) --")
    for i, s in enumerate(results["statutes"], 1):
        print(f"  {i}. {s.get('source','?')} S.{s.get('section_num','?')} - {s.get('section_title','?')[:45]}")
        print(f"     {s.get('text','')[:120]}...")

    if results.get("coverage_gap_warning"):
        print("\nWARNING: Coverage gap detected — query touches CrPC/BNSS topics "
              "with limited index coverage. Results may be sparse.")


if __name__ == "__main__":
    main()
