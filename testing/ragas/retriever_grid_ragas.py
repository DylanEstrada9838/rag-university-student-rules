"""
retriever_grid_ragas.py — RAGAS-based retriever grid search.

Evaluates the top 5 retriever configs (by Hit Rate from grid_search_results.json)
using RAGAS ContextPrecision and ContextRecall metrics only.

This is faster than full evaluation because:
  - No RAG chain (LLM answer generation) is needed for context-only metrics.
  - Only 5 configs are tested instead of the full grid.

Results are ranked by ContextRecall (then ContextPrecision as tiebreaker)
and saved to testing/ragas/ragas_grid_results.json.

Usage:
    cd <project_root>
    python testing/ragas/retriever_grid_ragas.py
"""

import os
import sys
import json

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from vectorstore import load_vectorstore
from chunking import get_chunks
from retriever import get_reranker_retriever

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.run_config import RunConfig

# Local imports
sys.path.append(os.path.dirname(__file__))
from dataset import get_ground_truth_with_references
from metrics_config import get_context_only_metrics

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'ragas_grid_results.json')
TESTING_DIR = os.path.join(os.path.dirname(__file__), '..')

# ── Top 5 configs from grid_search_results.json (by Hit Rate, then MRR) ────────
# Taken directly from grid_search_results.json ranks 1–5.
TOP_CONFIGS = [
    {
        "label": "reranker_hybrid | recursive 512/128 | k=10 | db_v3",
        "hit_rate": 0.90, "mrr": 0.7333, "recall": 0.85,
        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 128},
        "retriever": {
            "type": "reranker_hybrid", "k": 10, "fetch_k": 20,
            "lambda_mult": 0.3, "weights": [0.6, 0.4], "top_n": 5,
        },
        "persist_dir": os.path.join(TESTING_DIR, "chroma_db_v3"),
    },
    {
        "label": "hybrid | recursive 1024/128 | k=5 | db_v4",
        "hit_rate": 0.90, "mrr": 0.5283, "recall": 0.85,
        "chunking": {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 128},
        "retriever": {
            "type": "hybrid", "k": 5, "fetch_k": 15,
            "lambda_mult": 0.5, "weights": [0.5, 0.5],
        },
        "persist_dir": os.path.join(TESTING_DIR, "chroma_db_v4"),
    },
    {
        "label": "hybrid | recursive 512/64 | k=8 | db_v2",
        "hit_rate": 0.90, "mrr": 0.3350, "recall": 0.90,
        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        "retriever": {
            "type": "hybrid", "k": 8, "fetch_k": 20,
            "lambda_mult": 0.3, "weights": [0.6, 0.4],
        },
        "persist_dir": os.path.join(TESTING_DIR, "chroma_db_v2"),
    },
    {
        "label": "hybrid | semantic percentile-80 | k=10 | db_v8",
        "hit_rate": 0.90, "mrr": 0.2819, "recall": 0.90,
        "chunking": {
            "method": "semantic",
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 80,
            "min_chunk_size": 100,
        },
        "retriever": {
            "type": "hybrid", "k": 10, "fetch_k": 20,
            "lambda_mult": 0.5, "weights": [0.7, 0.3],
        },
        "persist_dir": os.path.join(TESTING_DIR, "chroma_db_v8"),
    },
    {
        "label": "hybrid | recursive 256/32 | k=5 | db_v1 (weights 0.5/0.5)",
        "hit_rate": 0.70, "mrr": 0.2850, "recall": 0.60,
        "chunking": {"method": "recursive", "chunk_size": 256, "chunk_overlap": 32},
        "retriever": {
            "type": "hybrid", "k": 5, "fetch_k": 15,
            "lambda_mult": 0.5, "weights": [0.5, 0.5],
        },
        "persist_dir": os.path.join(TESTING_DIR, "chroma_db_v1"),
    },
]


def build_retriever(config):
    """Builds the appropriate retriever from a config dict."""
    chunks = get_chunks(config["chunking"])
    persist_dir = os.path.normpath(config["persist_dir"])
    vectorstore = load_vectorstore(persist_dir=persist_dir)

    ret_cfg = config["retriever"]
    ret_type = ret_cfg["type"]

    if ret_type == "reranker_hybrid":
        vec_ret = vectorstore.as_retriever(
            search_kwargs={"k": ret_cfg["k"]},
            search_type="mmr",
            fetch_k=ret_cfg["fetch_k"],
            lambda_mult=ret_cfg["lambda_mult"],
        )
        bm25_ret = BM25Retriever.from_documents(documents=chunks, k=ret_cfg["k"])
        hybrid = EnsembleRetriever(
            retrievers=[vec_ret, bm25_ret],
            weights=ret_cfg["weights"],
        )
        return get_reranker_retriever(hybrid, top_n=ret_cfg["top_n"])

    elif ret_type == "hybrid":
        vec_ret = vectorstore.as_retriever(
            search_kwargs={"k": ret_cfg["k"]},
            search_type="mmr",
            fetch_k=ret_cfg["fetch_k"],
            lambda_mult=ret_cfg["lambda_mult"],
        )
        bm25_ret = BM25Retriever.from_documents(documents=chunks, k=ret_cfg["k"])
        return EnsembleRetriever(
            retrievers=[vec_ret, bm25_ret],
            weights=ret_cfg["weights"],
        )

    elif ret_type == "base":
        return vectorstore.as_retriever(
            search_kwargs={"k": ret_cfg["k"]},
            search_type=ret_cfg.get("search_type", "similarity"),
        )

    else:
        raise ValueError(f"Unknown retriever type: {ret_type}")


def build_context_dataset(retriever, samples_meta):
    """
    Retrieves contexts for all questions and builds a RAGAS EvaluationDataset
    with only user_input, retrieved_contexts and reference (no response needed
    for context-only metrics — RAGAS accepts a placeholder).
    """
    ragas_samples = []
    for item in samples_meta:
        docs = retriever.invoke(item["question"])
        contexts = [doc.page_content for doc in docs]
        ragas_samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=contexts,
                response="",          # not used by context-only metrics
                reference=item["reference"],
            )
        )
    return EvaluationDataset(samples=ragas_samples)


# ── Main loop ──────────────────────────────────────────────────────────────────
samples_meta = get_ground_truth_with_references()
metrics = get_context_only_metrics()
ranked_results = []

print("=" * 70)
print("RAGAS RETRIEVER GRID SEARCH  (ContextPrecision + ContextRecall)")
print("=" * 70)
print(f"  Configs to evaluate : {len(TOP_CONFIGS)}")
print(f"  Questions per config: {len(samples_meta)}")
print("=" * 70)

for i, cfg in enumerate(TOP_CONFIGS):
    print(f"\n[{i + 1}/{len(TOP_CONFIGS)}] {cfg['label']}")
    print(f"         Prior metrics — Hit Rate: {cfg['hit_rate']:.0%}  "
          f"MRR: {cfg['mrr']:.4f}  Recall: {cfg['recall']:.0%}")

    try:
        retriever = build_retriever(cfg)
        dataset = build_context_dataset(retriever, samples_meta)

        result = evaluate(
            dataset=dataset, 
            metrics=metrics,
            run_config=RunConfig(max_workers=1, timeout=1200)
        )
        df = result.to_pandas()

        cp = float(df["context_precision"].mean()) if "context_precision" in df.columns else None
        cr = float(df["context_recall"].mean()) if "context_recall" in df.columns else None

        print(f"         RAGAS — ContextPrecision: {cp:.4f if cp else 'N/A'}  "
              f"ContextRecall: {cr:.4f if cr else 'N/A'}")

        ranked_results.append({
            "label": cfg["label"],
            "prior_hit_rate": cfg["hit_rate"],
            "prior_mrr": cfg["mrr"],
            "prior_recall": cfg["recall"],
            "ragas_context_precision": cp,
            "ragas_context_recall": cr,
            "config": {k: v for k, v in cfg.items() if k not in ("label", "hit_rate", "mrr", "recall")},
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        ranked_results.append({
            "label": cfg["label"],
            "error": str(e),
            "config": cfg,
        })

# ── Rank by ContextRecall (desc), then ContextPrecision (desc) ────────────────
valid = [r for r in ranked_results if "ragas_context_recall" in r and r["ragas_context_recall"] is not None]
errors = [r for r in ranked_results if "error" in r]

valid.sort(key=lambda r: (r["ragas_context_recall"], r.get("ragas_context_precision", 0)), reverse=True)

print("\n" + "=" * 70)
print("RAGAS GRID — RANKED RESULTS")
print("=" * 70)
print(f"{'Rank':<5}{'CtxRecall':>12}{'CtxPrecision':>14}  Label")
print("-" * 70)
for rank, r in enumerate(valid, 1):
    cr = r["ragas_context_recall"]
    cp = r.get("ragas_context_precision")
    print(f"  #{rank:<3} {cr:>10.4f}  {(cp if cp else 0):>12.4f}  {r['label']}")

if errors:
    print(f"\n  {len(errors)} config(s) failed with errors.")

print("=" * 70)

# ── Save results ───────────────────────────────────────────────────────────────
output = {
    "ranked": valid,
    "errors": errors,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nResults saved → {OUTPUT_FILE}")
