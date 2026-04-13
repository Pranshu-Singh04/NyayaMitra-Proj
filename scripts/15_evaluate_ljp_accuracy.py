"""
15_evaluate_ljp_accuracy.py
============================
Batch Legal Judgment Prediction accuracy evaluator using the
NyayaAnumana dataset from HuggingFace.

Runs the full NyayaMitra pipeline on N rows from the test split and reports:
  - Accuracy (all predictions, including LOW-confidence)
  - Accuracy (HIGH + MEDIUM confidence only — LOW abstained, TASK 6)
  - Macro Precision / Recall / F1
  - Per-class breakdown
  - Confusion matrix
  - Results saved as JSON in eval/

Usage:
  python scripts/15_evaluate_ljp_accuracy.py --model gemini --n 50
  python scripts/15_evaluate_ljp_accuracy.py --model gemini --n 500 --split ternary
  python scripts/15_evaluate_ljp_accuracy.py --model gpt    --n 100 --api_key sk-...
  python scripts/15_evaluate_ljp_accuracy.py --model inlegalllama --colab_url URL --n 100
"""

import os, sys, json, time, argparse, random
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Force unbuffered output so prints appear immediately on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline   import NyayaMitraPipeline
from prompt_builder import QueryType


# ══════════════════════════════════════════════════════════════════════════════
# LABEL NORMALISER
# ══════════════════════════════════════════════════════════════════════════════
BINARY_MAP = {
    "0": "DISMISSED", "1": "ALLOWED",
    0:   "DISMISSED", 1:   "ALLOWED",
    "dismissed": "DISMISSED", "allowed": "ALLOWED",
    "rejected":  "DISMISSED", "granted": "ALLOWED",
}
TERNARY_MAP = {
    **BINARY_MAP,
    "2":                "PARTIALLY ALLOWED",
    2:                  "PARTIALLY ALLOWED",
    "partially allowed": "PARTIALLY ALLOWED",
    "partly allowed":   "PARTIALLY ALLOWED",
}

def normalise_gold(raw, split: str) -> str | None:
    m = TERNARY_MAP if split == "ternary" else BINARY_MAP
    key = str(raw).strip().lower()
    return m.get(key, m.get(raw, None))

def normalise_prediction(raw: str) -> str:
    """Map raw LLM prediction string to canonical label."""
    r = raw.strip().upper()
    if "PARTIALLY" in r or "PARTLY" in r:
        return "PARTIALLY ALLOWED"
    if "ALLOWED" in r or "GRANTED" in r or "ALLOW" in r:
        return "ALLOWED"
    if "DISMISSED" in r or "REJECTED" in r or "DENY" in r or "DENIED" in r:
        return "DISMISSED"
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_nyaya_anumana(n: int, split: str = "binary", seed: int = 42) -> list[dict]:
    """
    Load N random rows from the NyayaAnumana HuggingFace dataset.
    Falls back to the local CSV subset if the HF dataset is not available.

    Expected columns: 'facts' (or similar) + 'label' (0/1 or 0/1/2).
    Returns list of {"text": str, "gold_label": str}.
    """
    print(f"Loading Exploration-Lab/NyayaAnumana ({split}, n={n})...")

    try:
        from datasets import load_dataset
        # NyayaAnumana has configurations: 'binary' and 'ternary'
        ds = load_dataset(
            "Exploration-Lab/NyayaAnumana",
            name        = split,
            split       = "test",
            streaming   = True,
            trust_remote_code = True,
        )
        rows = []
        for ex in ds:
            if len(rows) >= n * 3:   # over-sample then random-select
                break
            rows.append(ex)

        random.seed(seed)
        random.shuffle(rows)

    except Exception as e:
        print(f"  HuggingFace load failed ({e}). Trying local CSV fallback...")
        rows = _load_local_csv(n * 3, split)

    usable = []
    for row in rows:
        # Try common column names for text
        text = (
            row.get("facts") or row.get("case_facts") or row.get("text")
            or row.get("judgment_text") or row.get("petition_text") or ""
        )
        text = str(text).strip()
        if not text:
            continue

        # Try common column names for label
        raw_label = row.get("label", row.get("outcome", row.get("gold_label", "")))
        gold = normalise_gold(raw_label, split)
        if gold is None:
            continue

        usable.append({
            "text"      : text[:2500],   # truncate to keep prompts manageable
            "gold_label": gold,
        })

        if len(usable) >= n:
            break

    print(f"  Loaded {len(usable)} usable rows (requested {n})")
    return usable


def _load_local_csv(n: int, split: str) -> list[dict]:
    """Fallback: load from the local NyayaAnumana CSV subset."""
    import csv
    data_root = Path(__file__).parent.parent / "data" / "raw" / "NyayaAnumana_Sample_Subset"
    csv_candidates = [
        data_root / "test" / "binary_test" / "CJPE_ext_SCI_HCs_Tribunals_daily_orders_test.csv",
        data_root / "test" / "ternary_test" / "CJPE_ext_SCI_HCs_tribunals_dailyorder_test_wo_RoD_ternary.csv",
        data_root / "NyayaAnumana_binary_test.csv",
        data_root / "NyayaAnumana_ternary_test.csv",
    ]
    for path in csv_candidates:
        if path.exists():
            print(f"  Using local CSV: {path}")
            rows = []
            with open(path, encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    rows.append(dict(row))
                    if len(rows) >= n:
                        break
            return rows
    raise FileNotFoundError(
        "NyayaAnumana dataset not found locally. "
        "Set HUGGINGFACE_TOKEN or download the dataset to data/raw/NyayaAnumana_Sample_Subset/."
    )


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, per-class precision/recall/F1, macro averages."""
    labels = sorted(
        {r["gold"] for r in results}
        | {r["pred"] for r in results if r["pred"] not in ("UNKNOWN", "ABSTAINED")}
    )

    correct   = sum(1 for r in results if r["gold"] == r["pred"])
    total     = len(results)
    unknown   = sum(1 for r in results if r["pred"] in ("UNKNOWN", "ABSTAINED"))
    accuracy  = correct / max(total - unknown, 1)

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for r in results:
        g, p = r["gold"], r["pred"]
        if p in ("UNKNOWN", "ABSTAINED"):
            fn[g] += 1
            continue
        if g == p: tp[g] += 1
        else:      fp[p] += 1; fn[g] += 1

    per_class = {}
    for lbl in labels:
        prec = tp[lbl] / max(tp[lbl] + fp[lbl], 1)
        rec  = tp[lbl] / max(tp[lbl] + fn[lbl], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall"   : round(rec,  4),
            "f1"       : round(f1,   4),
            "support"  : tp[lbl] + fn[lbl],
        }

    macro_p  = sum(v["precision"] for v in per_class.values()) / max(len(per_class), 1)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / max(len(per_class), 1)
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / max(len(per_class), 1)

    conf = defaultdict(lambda: defaultdict(int))
    for r in results:
        conf[r["gold"]][r["pred"]] += 1

    return {
        "accuracy"          : round(accuracy, 4),
        "total"             : total,
        "correct"           : correct,
        "unknown_count"     : unknown,
        "macro_precision"   : round(macro_p,  4),
        "macro_recall"      : round(macro_r,  4),
        "macro_f1"          : round(macro_f1, 4),
        "per_class"         : per_class,
        "label_distribution": {lbl: sum(1 for r in results if r["gold"] == lbl) for lbl in labels},
        "confusion_matrix"  : {g: dict(p_counts) for g, p_counts in conf.items()},
    }


def _print_metrics(metrics: dict, tag: str = ""):
    tag_str = f" [{tag}]" if tag else ""
    print(f"\n{'='*60}")
    print(f"RESULTS{tag_str}")
    print(f"{'='*60}")
    print(f"Accuracy  : {metrics['accuracy']:.1%}  ({metrics['correct']}/{metrics['total']})")
    print(f"Macro F1  : {metrics['macro_f1']:.1%}")
    print(f"Macro P   : {metrics['macro_precision']:.1%}")
    print(f"Macro R   : {metrics['macro_recall']:.1%}")
    print(f"Unknown/Abstained : {metrics['unknown_count']}")
    print(f"\nPer-class breakdown:")
    for lbl, m in metrics["per_class"].items():
        print(f"  {lbl:22s}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
              f"F1={m['f1']:.2f}  support={m['support']}")
    print(f"\nLabel distribution (gold): {metrics['label_distribution']}")
    print(f"\nConfusion matrix:")
    all_lbls = sorted(metrics["per_class"].keys())
    header   = f"{'':22s}" + "".join(f"{l:22s}" for l in all_lbls)
    print(f"  {header}")
    for g in all_lbls:
        row_str = f"  {g:22s}" + "".join(
            f"{metrics['confusion_matrix'].get(g, {}).get(p, 0):22d}" for p in all_lbls
        )
        print(row_str)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
class LJPAccuracyEvaluator:

    def __init__(
        self,
        index_dir   : str = "indexes",
        model       : str = "gemini",
        api_key     : str = None,
        colab_url   : str = None,
        output_dir  : str = "eval",
        split       : str = "binary",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_name = model
        self.split      = split

        print(f"\n{'='*60}")
        print(f"NyayaMitra LJP Accuracy Evaluator — {model} — {split}")
        print(f"{'='*60}")

        # abstain_on_low_confidence=True so pipeline returns abstained=True for LOW
        self.pipeline = NyayaMitraPipeline(
            index_dir                 = index_dir,
            model                     = model,
            api_key                   = api_key,
            colab_url                 = colab_url,
            verbose                   = False,
            abstain_on_low_confidence = True,
        )
        self.prediction_type = split

    def run(self, rows: list[dict]) -> dict:
        all_results = []
        errors      = 0
        t_start     = time.time()

        print(f"\nRunning {len(rows)} predictions...")
        print("-" * 60)

        for i, row in enumerate(rows, 1):
            try:
                parsed = self.pipeline.query(
                    row["text"],
                    query_type      = QueryType.LJP,
                    prediction_type = self.prediction_type,
                )

                gold = row["gold_label"]

                if parsed.abstained:
                    pred = "ABSTAINED"
                elif parsed.prediction:
                    pred = normalise_prediction(parsed.prediction)
                else:
                    pred = "UNKNOWN"

                match = (pred == gold)
                all_results.append({
                    "idx"       : i,
                    "gold"      : gold,
                    "pred"      : pred,
                    "confidence": parsed.confidence,
                    "abstained" : parsed.abstained,
                    "correct"   : match,
                    "latency_ms": parsed.latency_ms,
                    "raw_pred"  : (parsed.prediction or "")[:100],
                })

                icon = "✅" if match else ("⏭" if parsed.abstained else "❌")
                print(f"  [{i:3d}/{len(rows)}] {icon} gold={gold:20s} pred={pred:20s} "
                      f"conf={parsed.confidence:6s}  ({parsed.latency_ms:.0f}ms)")

            except Exception as e:
                errors += 1
                print(f"  [{i:3d}/{len(rows)}] ⚠  ERROR: {e}")
                all_results.append({
                    "idx": i, "gold": row["gold_label"], "pred": "UNKNOWN",
                    "confidence": "", "abstained": False, "correct": False,
                    "latency_ms": 0, "raw_pred": str(e),
                })

        elapsed = time.time() - t_start

        # ── TASK 6: two accuracy numbers ─────────────────────────────────────
        # (a) ALL queries: LOW = guessed wrong (treated as UNKNOWN)
        metrics_all  = compute_metrics(all_results)

        # (b) HIGH + MEDIUM only: LOW = removed from denominator (abstained)
        hm_results   = [r for r in all_results if r["confidence"] in ("HIGH", "MEDIUM")]
        metrics_hm   = compute_metrics(hm_results) if hm_results else {}

        _print_metrics(metrics_all, tag=f"{self.model_name} / {self.split} / ALL")
        if metrics_hm:
            _print_metrics(
                metrics_hm,
                tag=f"{self.model_name} / {self.split} / HIGH+MEDIUM only"
            )
            print(f"\n  Abstained (LOW confidence): "
                  f"{len(all_results) - len(hm_results)} / {len(all_results)} queries "
                  f"({(len(all_results) - len(hm_results)) / max(len(all_results), 1):.1%})")

        print(f"\nErrors    : {errors}")
        print(f"Time      : {elapsed:.1f}s  ({elapsed/max(len(rows),1):.1f}s/query)")

        output = {
            "timestamp"           : datetime.now().isoformat(),
            "model"               : self.model_name,
            "split"               : self.split,
            "n_total"             : len(all_results),
            "n_errors"            : errors,
            "elapsed_s"           : round(elapsed, 1),
            "metrics_all"         : metrics_all,
            "metrics_high_medium" : metrics_hm,
            "n_abstained"         : sum(1 for r in all_results if r.get("abstained")),
            "per_row"             : all_results,
        }

        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.output_dir / f"ljp_accuracy_{self.model_name}_{self.split}_{ts}.json"
        fname.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nSaved: {fname}")

        return output


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="NyayaMitra LJP accuracy evaluation on NyayaAnumana"
    )
    parser.add_argument("--model",      required=True,
                        choices=["inlegalllama", "gemini", "gpt"],
                        help="LLM backend")
    parser.add_argument("--split",      default="binary",
                        choices=["binary", "ternary"],
                        help="NyayaAnumana split to evaluate on (default: binary)")
    parser.add_argument("--n",          type=int, default=500,
                        help="Number of test rows (default: 500)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for row sampling")
    parser.add_argument("--index_dir",  default="indexes",
                        help="Path to FAISS indexes directory")
    parser.add_argument("--output_dir", default="eval",
                        help="Where to save result JSON files")
    parser.add_argument("--api_key",    default=None,
                        help="Gemini/OpenAI API key")
    parser.add_argument("--colab_url",  default=None,
                        help="Colab FastAPI endpoint for INLegalLlama")
    args = parser.parse_args()

    if not args.api_key:
        if args.model == "gemini":
            args.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif args.model == "gpt":
            args.api_key = os.getenv("OPENAI_API_KEY")

    rows = load_nyaya_anumana(args.n, split=args.split, seed=args.seed)
    if not rows:
        print("ERROR: No usable rows loaded. Check dataset access and column names.")
        sys.exit(1)

    evaluator = LJPAccuracyEvaluator(
        index_dir  = args.index_dir,
        model      = args.model,
        api_key    = args.api_key,
        colab_url  = args.colab_url,
        output_dir = args.output_dir,
        split      = args.split,
    )
    evaluator.run(rows)


if __name__ == "__main__":
    main()
