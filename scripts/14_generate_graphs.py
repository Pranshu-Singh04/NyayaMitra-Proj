"""
14_generate_graphs.py
======================
Generates all paper figures from the evaluation JSON files.
Run AFTER 13_evaluate_hallucination.py has produced its outputs.

Produces 8 publication-quality figures:
  Fig 1: Grounding score per query (bar chart, coloured by query type)
  Fig 2: Avg grounding vs hallucination by query type (grouped bar)
  Fig 3: Claim label distribution by query type (stacked bar)
  Fig 4: LLM vs NLI latency comparison (side-by-side bars)
  Fig 5: Entailment score distribution per query type (overlapping KDE)
  Fig 6: RAG vs no-RAG grounding per query (paired bar)
  Fig 7: RAG vs no-RAG aggregate (main paper figure)
  Fig 8: Claim count comparison RAG vs no-RAG

Usage:
  python scripts/14_generate_graphs.py --results_dir results
  python scripts/14_generate_graphs.py --results_dir results --eval_file results/hallucination_eval_XYZ.json --ablation_file results/ablation_graph_data_XYZ.json
"""

import os, sys, json, argparse, glob
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Install: pip install matplotlib numpy")
    sys.exit(1)


# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "legal_qa"  : "#2E75B6",
    "ljp"       : "#E86B3A",
    "statute"   : "#70AD47",
    "rag"       : "#1F4E79",
    "no_rag"    : "#F4B942",
    "grounding" : "#2E75B6",
    "halluc"    : "#C00000",
    "neutral"   : "#A5A5A5",
}
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 12,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
})

QT_LABELS = {"legal_qa":"Legal Q&A","ljp":"Judgment Pred.","statute":"Statute Lookup"}


def load_latest(results_dir, pattern):
    files = sorted(glob.glob(str(Path(results_dir)/pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def fig1_grounding_per_query(data, out_dir):
    """Bar chart: grounding score for each of 15 queries, coloured by type."""
    items  = data["fig1_grounding_per_query"]
    n      = len(items)
    x      = list(range(n))
    g      = [d["grounding_score"] for d in items]
    h      = [d["hallucination_score"] for d in items]
    colors = [COLORS[d["query_type"]] for d in items]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(x, g, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].axhline(sum(g)/len(g), color="black", linestyle="--", linewidth=1, label=f"Mean={sum(g)/len(g):.1%}")
    axes[0].set_ylabel("Grounding Score")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Figure 1: Per-Query Grounding & Hallucination Scores (NyayaMitra)")
    axes[0].legend(fontsize=9)
    for i,v in enumerate(g):
        axes[0].text(i, v+0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, h, color=[COLORS["halluc"]]*n, alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[1].axhline(sum(h)/len(h), color="black", linestyle="--", linewidth=1, label=f"Mean={sum(h)/len(h):.1%}")
    axes[1].set_ylabel("Hallucination Score")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel("Query Number")
    axes[1].legend(fontsize=9)

    # X-tick labels and type dividers
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(i+1) for i in x], fontsize=9)
    patches = [mpatches.Patch(color=COLORS[qt], label=QT_LABELS[qt])
               for qt in ["legal_qa","ljp","statute"]]
    fig.legend(handles=patches, loc="upper right", ncol=3, fontsize=9, framealpha=0.9)

    plt.tight_layout()
    p = Path(out_dir)/"fig1_per_query_grounding.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig2_avg_by_type(data, out_dir):
    """Grouped bar: avg grounding vs hallucination by query type."""
    d2    = data["fig2_avg_by_type"]
    types = list(d2.keys())
    g     = [d2[t]["grounding_mean"] for t in types]
    h     = [d2[t]["hallucination_mean"] for t in types]
    x     = np.arange(len(types))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x-w/2, g, w, label="Grounding ↑", color=[COLORS[t] for t in types], edgecolor="white")
    b2 = ax.bar(x+w/2, h, w, label="Hallucination ↓", color=COLORS["halluc"], alpha=0.75, edgecolor="white")

    for bar,v in zip(b1, g): ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.0%}", ha="center",fontsize=10,fontweight="bold")
    for bar,v in zip(b2, h): ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.0%}", ha="center",fontsize=10,fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([QT_LABELS[t] for t in types], fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.2)
    ax.set_title("Figure 2: Average Grounding & Hallucination by Query Type")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p = Path(out_dir)/"fig2_avg_by_type.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig3_claim_distribution(data, out_dir):
    """Stacked bar: SUPPORTED / NEUTRAL / UNSUPPORTED claims per query type."""
    d3    = data["fig3_claim_distribution"]
    types = list(d3.keys())
    sup   = [d3[t]["supported"]   for t in types]
    neu   = [d3[t]["neutral"]     for t in types]
    uns   = [d3[t]["unsupported"] for t in types]
    totals= [sup[i]+neu[i]+uns[i] for i in range(len(types))]
    x     = np.arange(len(types))

    fig, ax = plt.subplots(figsize=(9, 6))
    p1 = ax.bar(x, sup, label="SUPPORTED",   color=COLORS["grounding"])
    p2 = ax.bar(x, neu, bottom=sup,          label="NEUTRAL",   color=COLORS["neutral"])
    p3 = ax.bar(x, uns, bottom=[s+n for s,n in zip(sup,neu)], label="UNSUPPORTED", color=COLORS["halluc"])

    for i,t in enumerate(totals):
        ax.text(i, t+0.3, f"n={t}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([QT_LABELS[t] for t in types], fontsize=11)
    ax.set_ylabel("Number of Claims")
    ax.set_title("Figure 3: Claim Label Distribution by Query Type")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p = Path(out_dir)/"fig3_claim_distribution.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig4_latency(data, out_dir):
    """Box plots: LLM latency vs NLI latency."""
    d4  = data["fig4_latency"]
    llm = [v/1000 for v in d4["llm_latencies_ms"]]  # → seconds
    nli = [v/1000 for v in d4["nli_latencies_ms"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([llm, nli], labels=["LLM Generation\n(Mistral-7B)", "NLI Hallucination\nChecker"],
                    patch_artist=True, notch=False,
                    boxprops=dict(facecolor=COLORS["rag"], alpha=0.6),
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][1].set_facecolor(COLORS["no_rag"])

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Figure 4: Latency Comparison — LLM vs NLI Checker")
    ax.text(1, max(llm)+0.5, f"Median: {sorted(llm)[len(llm)//2]:.1f}s", ha="center", fontsize=10)
    ax.text(2, max(nli)+0.05, f"Median: {sorted(nli)[len(nli)//2]:.2f}s", ha="center", fontsize=10)
    plt.tight_layout()
    p = Path(out_dir)/"fig4_latency.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig5_entailment_kde(data, out_dir):
    """Overlapping histograms: entailment score distribution per query type."""
    d5  = data["fig5_entailment_scores"]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 25)
    for qt, scores in d5.items():
        if not scores: continue
        ax.hist(scores, bins=bins, alpha=0.55, label=QT_LABELS[qt],
                color=COLORS[qt], edgecolor="white", linewidth=0.3)
        mean = sum(scores)/len(scores)
        ax.axvline(mean, color=COLORS[qt], linestyle="--", linewidth=1.5)

    ax.axvline(0.35, color="black", linestyle=":", linewidth=1.5, label="Support threshold (0.35)")
    ax.set_xlabel("Entailment Score (NLI)")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Figure 5: Distribution of Per-Claim Entailment Scores by Query Type")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p = Path(out_dir)/"fig5_entailment_distribution.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig6_rag_vs_no_rag_per_query(data, out_dir):
    """Paired bar: RAG vs no-RAG grounding per query."""
    items = data["fig6_rag_vs_no_rag_per_query"]
    n  = len(items)
    x  = np.arange(n)
    w  = 0.35
    rg = [d["rag_grounding"] for d in items]
    ng = [d["no_rag_grounding"] for d in items]

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x-w/2, rg, w, label="RAG",    color=COLORS["rag"])
    b2 = ax.bar(x+w/2, ng, w, label="no-RAG", color=COLORS["no_rag"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(n)])
    ax.set_xlabel("Query Number")
    ax.set_ylabel("Grounding Score")
    ax.set_ylim(0, 1.2)
    ax.set_title("Figure 6: RAG vs no-RAG Grounding Score per Query")
    ax.legend(fontsize=10)
    ax.axhline(sum(rg)/n, color=COLORS["rag"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(sum(ng)/n, color=COLORS["no_rag"], linestyle="--", linewidth=1, alpha=0.7)
    plt.tight_layout()
    p = Path(out_dir)/"fig6_rag_vs_no_rag_per_query.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig7_aggregate(data, out_dir):
    """Main paper figure: aggregate RAG vs no-RAG grounding and hallucination."""
    d7  = data["fig7_aggregate"]
    metrics = ["Grounding ↑", "Hallucination ↓"]
    rag_vals    = [d7["rag_avg_grounding"],    d7["rag_avg_hallucination"]]
    no_rag_vals = [d7["no_rag_avg_grounding"], d7["no_rag_avg_hallucination"]]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x-w/2, rag_vals,    w, label="RAG (NyayaMitra)",  color=COLORS["rag"],    edgecolor="white")
    b2 = ax.bar(x+w/2, no_rag_vals, w, label="no-RAG (LLM only)", color=COLORS["no_rag"], edgecolor="white")

    for bar,v in zip(list(b1)+list(b2), rag_vals+no_rag_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.015, f"{v:.1%}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Delta annotation
    dg = d7["rag_avg_grounding"] - d7["no_rag_avg_grounding"]
    dh = d7["no_rag_avg_hallucination"] - d7["rag_avg_hallucination"]
    ax.annotate(f"Δ={dg:+.1%}", xy=(0, max(rag_vals[0],no_rag_vals[0])+0.05),
                ha="center", fontsize=11, color=COLORS["rag"], fontweight="bold")
    ax.annotate(f"Δ={-dh:+.1%}", xy=(1, max(rag_vals[1],no_rag_vals[1])+0.05),
                ha="center", fontsize=11, color=COLORS["halluc"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.3)
    ax.set_title(f"Figure 7: RAG vs no-RAG — Aggregate Hallucination Metrics\n(N={d7['n_queries']} queries)")
    ax.legend(fontsize=11)
    plt.tight_layout()
    p = Path(out_dir)/"fig7_rag_vs_no_rag_aggregate.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


def fig8_claim_counts(data, out_dir):
    """Stacked bar: supported vs total claims, RAG vs no-RAG."""
    items  = data["fig8_claim_counts"]
    n      = len(items)
    x      = np.arange(n)
    w      = 0.35
    rt     = [d["rag_total_claims"] for d in items]
    nt     = [d["no_rag_total_claims"] for d in items]
    rs     = [d["rag_supported"] for d in items]
    ns     = [d["no_rag_supported"] for d in items]
    ru     = [rt[i]-rs[i] for i in range(n)]
    nu     = [nt[i]-ns[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x-w/2, rs, w, label="RAG supported",    color=COLORS["rag"])
    ax.bar(x-w/2, ru, w, bottom=rs,                label="RAG unsup/neutral", color=COLORS["rag"], alpha=0.35)
    ax.bar(x+w/2, ns, w, label="no-RAG supported", color=COLORS["no_rag"])
    ax.bar(x+w/2, nu, w, bottom=ns,                label="no-RAG unsup/neutral", color=COLORS["no_rag"], alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(n)])
    ax.set_xlabel("Query Number")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Figure 8: Supported vs Unsupported Claims — RAG vs no-RAG")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    p = Path(out_dir)/"fig8_claim_counts.png"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",   default="results")
    parser.add_argument("--eval_file",     default=None)
    parser.add_argument("--ablation_file", default=None)
    parser.add_argument("--output_dir",    default="results/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_file     = args.eval_file     or None
    ablation_file = args.ablation_file or None

    # Load eval graph data
    eval_data = None
    if eval_file:
        with open(eval_file) as f: eval_data = json.load(f)
    else:
        eval_data = load_latest(args.results_dir, "graph_data_*.json")

    # Load ablation graph data
    abl_data = None
    if ablation_file:
        with open(ablation_file) as f: abl_data = json.load(f)
    else:
        abl_data = load_latest(args.results_dir, "ablation_graph_data_*.json")

    print(f"\n{'='*55}")
    print("NyayaMitra — Generating Paper Figures")
    print(f"{'='*55}")

    if eval_data:
        print("\n── Evaluation figures (Figs 1-5):")
        try: fig1_grounding_per_query(eval_data, args.output_dir)
        except Exception as e: print(f"  Fig1 error: {e}")
        try: fig2_avg_by_type(eval_data, args.output_dir)
        except Exception as e: print(f"  Fig2 error: {e}")
        try: fig3_claim_distribution(eval_data, args.output_dir)
        except Exception as e: print(f"  Fig3 error: {e}")
        try: fig4_latency(eval_data, args.output_dir)
        except Exception as e: print(f"  Fig4 error: {e}")
        try: fig5_entailment_kde(eval_data, args.output_dir)
        except Exception as e: print(f"  Fig5 error: {e}")
    else:
        print("  No eval graph data found. Run: python scripts/13_evaluate_hallucination.py --mode eval")

    if abl_data:
        print("\n── Ablation figures (Figs 6-8):")
        try: fig6_rag_vs_no_rag_per_query(abl_data, args.output_dir)
        except Exception as e: print(f"  Fig6 error: {e}")
        try: fig7_aggregate(abl_data, args.output_dir)
        except Exception as e: print(f"  Fig7 error: {e}")
        try: fig8_claim_counts(abl_data, args.output_dir)
        except Exception as e: print(f"  Fig8 error: {e}")
    else:
        print("  No ablation data found. Run: python scripts/13_evaluate_hallucination.py --mode rag_vs_no_rag")

    print(f"\nAll figures saved to: {args.output_dir}/")
    print("Use these PNG files directly in your paper (300 DPI).")

if __name__ == "__main__":
    main()
