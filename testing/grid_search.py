"""
grid_search.py  –  Hyperparameter search for chunking + retriever

Evaluates different combinations of:
    • Chunking strategy & params  (recursive vs semantic)
    • Retriever type              (base / hybrid / reranker)
    • Retriever params            (k, search_type, lambda_mult, weights, top_n …)

Uses random search when the full grid is too large (> MAX_COMBOS).
Results are sorted by Hit Rate → MRR → Recall and printed as a table.

Supports two modes:
    --reuse    Reuse existing chroma_db_vN directories (skip vectorstore creation)
    (default)  Create new versioned vectorstores from scratch

Usage:
    python testing/grid_search.py              (create new vectorstores)
    python testing/grid_search.py --reuse      (reuse existing ones)
"""

import os
import sys
import json
import random
import shutil
import itertools
import time
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from vectorstore import create_vector_db, load_vectorstore
from chunking import get_chunks
from retriever import get_base_retriever, get_hybrid_retriever, get_reranker_retriever
from ground_truth import ground_truth
from retrieval_metrics import hit_rate, mrr, recall

# ── constants ─────────────────────────────────────────────────────────────
MAX_COMBOS = 20          # If total combos exceed this, switch to random search
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

REUSE_MODE = "--reuse" in sys.argv

# ══════════════════════════════════════════════════════════════════════════
# 1.  PARAMETER GRID
# ══════════════════════════════════════════════════════════════════════════

# --- Chunking configs ---------------------------------------------------
chunking_grid = [
    # Recursive variants
    {"method": "recursive", "chunk_size": 256,  "chunk_overlap": 32},
    {"method": "recursive", "chunk_size": 512,  "chunk_overlap": 64},
    {"method": "recursive", "chunk_size": 512,  "chunk_overlap": 128},
    {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 128},
    {"method": "recursive", "chunk_size": 1024, "chunk_overlap": 256},
    # Semantic variants
    {"method": "semantic", "breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 90, "min_chunk_size": 150},
    {"method": "semantic", "breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 95, "min_chunk_size": 200},
    {"method": "semantic", "breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 80, "min_chunk_size": 100},
]

# --- Retriever configs --------------------------------------------------
retriever_grid = [
    # ---- Base (similarity) ----
    {"type": "base", "search_type": "similarity", "k": 4},
    {"type": "base", "search_type": "similarity", "k": 6},
    {"type": "base", "search_type": "similarity", "k": 10},
    # ---- Base (MMR) ----
    {"type": "base", "search_type": "mmr", "k": 4,  "fetch_k": 12, "lambda_mult": 0.3},
    {"type": "base", "search_type": "mmr", "k": 6,  "fetch_k": 18, "lambda_mult": 0.3},
    {"type": "base", "search_type": "mmr", "k": 6,  "fetch_k": 12, "lambda_mult": 0.5},
    {"type": "base", "search_type": "mmr", "k": 10, "fetch_k": 20, "lambda_mult": 0.5},
    # ---- Hybrid ----
    {"type": "hybrid", "k": 5,  "fetch_k": 15, "lambda_mult": 0.5, "weights": [0.7, 0.3]},
    {"type": "hybrid", "k": 5,  "fetch_k": 15, "lambda_mult": 0.5, "weights": [0.5, 0.5]},
    {"type": "hybrid", "k": 8,  "fetch_k": 20, "lambda_mult": 0.3, "weights": [0.6, 0.4]},
    {"type": "hybrid", "k": 10, "fetch_k": 20, "lambda_mult": 0.5, "weights": [0.7, 0.3]},
    # ---- Reranker on top of base ----
    {"type": "reranker_base", "search_type": "mmr", "k": 10, "fetch_k": 20, "lambda_mult": 0.5, "top_n": 3},
    {"type": "reranker_base", "search_type": "mmr", "k": 10, "fetch_k": 20, "lambda_mult": 0.5, "top_n": 5},
    {"type": "reranker_base", "search_type": "similarity", "k": 10, "top_n": 5},
    # ---- Reranker on top of hybrid ----
    {"type": "reranker_hybrid", "k": 8,  "fetch_k": 20, "lambda_mult": 0.5, "weights": [0.7, 0.3], "top_n": 3},
    {"type": "reranker_hybrid", "k": 10, "fetch_k": 20, "lambda_mult": 0.5, "weights": [0.5, 0.5], "top_n": 5},
    {"type": "reranker_hybrid", "k": 10, "fetch_k": 20, "lambda_mult": 0.3, "weights": [0.6, 0.4], "top_n": 5},
]


# ══════════════════════════════════════════════════════════════════════════
# 2.  BUILDER HELPERS
# ══════════════════════════════════════════════════════════════════════════

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def _build_retriever(vectorstore, chunks, retriever_config):
    """
    Build a retriever from a config dict.
    Returns the retriever object.
    """
    rtype = retriever_config["type"]

    if rtype == "base":
        search_kwargs = {"k": retriever_config.get("k", 6)}
        search_type = retriever_config.get("search_type", "similarity")
        extra = {}
        if search_type == "mmr":
            extra["fetch_k"] = retriever_config.get("fetch_k", 12)
            extra["lambda_mult"] = retriever_config.get("lambda_mult", 0.5)
        return vectorstore.as_retriever(
            search_kwargs=search_kwargs,
            search_type=search_type,
            **extra,
        )

    elif rtype == "hybrid":
        k = retriever_config.get("k", 5)
        search_type = retriever_config.get("search_type", "mmr")
        fetch_k = retriever_config.get("fetch_k", 15)
        lambda_mult = retriever_config.get("lambda_mult", 0.5)
        weights = retriever_config.get("weights", [0.7, 0.3])

        vec_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k},
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=k)
        return EnsembleRetriever(
            retrievers=[vec_retriever, bm25_retriever],
            weights=weights,
        )

    elif rtype in ("reranker_base", "reranker_hybrid"):
        top_n = retriever_config.get("top_n", 3)
        if rtype == "reranker_base":
            base_config = {**retriever_config, "type": "base"}
        else:
            base_config = {**retriever_config, "type": "hybrid"}
        base_retriever = _build_retriever(vectorstore, chunks, base_config)
        return get_reranker_retriever(base_retriever, top_n=top_n)

    else:
        raise ValueError(f"Unknown retriever type: {rtype}")


def _chunking_key(cfg):
    """Stable string key for a chunking config (for caching)."""
    return json.dumps(cfg, sort_keys=True)


# ══════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_retriever(retriever):
    """Run the retriever against ground_truth and return metrics dict."""
    results = []
    for item in ground_truth:
        query = item["question"]
        expected_pages = item["expected_pages"]
        retrieved_docs = retriever.invoke(query)
        retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs]
        results.append({
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "retrieved_pages_ordered": retrieved_pages,
        })

    return {
        "hit_rate": hit_rate(results),
        "mrr": mrr(results),
        "recall": recall(results),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4.  MAIN SEARCH LOOP
# ══════════════════════════════════════════════════════════════════════════

def _short_label(chunking_cfg, retriever_cfg):
    """Human-readable summary of one combo."""
    method = chunking_cfg.get("method", "semantic")
    if method == "recursive":
        c_label = f"recursive(cs={chunking_cfg.get('chunk_size')}, co={chunking_cfg.get('chunk_overlap')})"
    else:
        c_label = f"semantic(type={chunking_cfg.get('breakpoint_threshold_type')}, amt={chunking_cfg.get('breakpoint_threshold_amount')}, min={chunking_cfg.get('min_chunk_size')})"

    rtype = retriever_cfg["type"]
    r_label = f"{rtype}(k={retriever_cfg.get('k')}"
    if "search_type" in retriever_cfg:
        r_label += f", st={retriever_cfg['search_type']}"
    if "lambda_mult" in retriever_cfg:
        r_label += f", λ={retriever_cfg['lambda_mult']}"
    if "weights" in retriever_cfg:
        r_label += f", w={retriever_cfg['weights']}"
    if "top_n" in retriever_cfg:
        r_label += f", top_n={retriever_cfg['top_n']}"
    r_label += ")"
    return c_label, r_label


def main():
    all_combos = list(itertools.product(chunking_grid, retriever_grid))
    total = len(all_combos)

    use_random = total > MAX_COMBOS
    if use_random:
        combos = random.sample(all_combos, MAX_COMBOS)
        print(f"\n⚡ {total} total combos → using RANDOM SEARCH with {MAX_COMBOS} samples (seed={RANDOM_SEED})")
    else:
        combos = all_combos
        print(f"\n🔍 Running FULL GRID SEARCH over {total} combinations")

    if REUSE_MODE:
        print("♻️  REUSE MODE: Loading existing chroma_db_vN directories (no new vectorstores created)\n")
    else:
        print("🆕  CREATE MODE: Creating new versioned vectorstores\n")

    # ── In reuse mode, pre-build chunking → (vectorstore, chunks, dir) cache ──
    # We assign each unique chunking config a stable version number (1-indexed
    # in the order they appear in chunking_grid) so the mapping is deterministic.
    chunking_cache = {}  # key: _chunking_key(cfg) → (vectorstore, chunks, persist_dir)

    if REUSE_MODE:
        # Build the mapping: chunking_grid index → chroma_db_v{index+1}
        for i, cfg in enumerate(chunking_grid):
            version = i + 1
            persist_dir = os.path.join(SCRIPT_DIR, f"chroma_db_v{version}")
            if not os.path.exists(persist_dir):
                print(f"  ⚠️  {persist_dir} not found – will create it")
                vectorstore, persist_dir, chunks = create_vector_db(
                    chunking_config=cfg, persist_dir=persist_dir
                )
            else:
                print(f"  ✓ Loading {persist_dir}")
                vectorstore = load_vectorstore(persist_dir=persist_dir)
                chunks = get_chunks(cfg)
            chunking_cache[_chunking_key(cfg)] = (vectorstore, chunks, persist_dir)
        print()

    all_results = []
    chroma_dirs_used = []

    for idx, (chunking_cfg, retriever_cfg) in enumerate(combos, start=1):
        c_label, r_label = _short_label(chunking_cfg, retriever_cfg)
        print(f"[{idx}/{len(combos)}]  Chunking: {c_label}")
        print(f"{'':>{len(str(len(combos)))+4}}Retriever: {r_label}")

        try:
            t0 = time.time()

            key = _chunking_key(chunking_cfg)

            if REUSE_MODE:
                # Use cached vectorstore
                vectorstore, chunks, persist_dir = chunking_cache[key]
            else:
                # Check if we already created this chunking config in this run
                if key in chunking_cache:
                    vectorstore, chunks, persist_dir = chunking_cache[key]
                else:
                    vectorstore, persist_dir, chunks = create_vector_db(
                        chunking_config=chunking_cfg
                    )
                    chunking_cache[key] = (vectorstore, chunks, persist_dir)

            if persist_dir not in chroma_dirs_used:
                chroma_dirs_used.append(persist_dir)
            print(f"  → vectorstore: {persist_dir}  ({len(chunks)} chunks)")

            # 2. Build retriever
            retriever = _build_retriever(vectorstore, chunks, retriever_cfg)

            # 3. Evaluate
            metrics = evaluate_retriever(retriever)
            elapsed = time.time() - t0

            metrics["chunking"] = chunking_cfg
            metrics["retriever"] = retriever_cfg
            metrics["persist_dir"] = persist_dir
            metrics["time_s"] = round(elapsed, 1)
            all_results.append(metrics)

            print(f"  → Hit Rate: {metrics['hit_rate']:.2%}  |  MRR: {metrics['mrr']:.4f}  |  Recall: {metrics['recall']:.2%}  ({elapsed:.1f}s)")

        except Exception as e:
            import traceback
            print(f"  ✗ ERROR: {e}")
            traceback.print_exc()

        print()

    # ── Sort by hit_rate desc, then mrr desc, then recall desc ────────────
    all_results.sort(key=lambda r: (r["hit_rate"], r["mrr"], r["recall"]), reverse=True)

    # ── Pretty-print top results ──────────────────────────────────────────
    print("\n" + "=" * 100)
    print("                         GRID SEARCH RESULTS  (sorted by Hit Rate → MRR → Recall)")
    print("=" * 100)
    print(f"{'#':>3}  {'Hit Rate':>9}  {'MRR':>8}  {'Recall':>8}  {'Time':>6}  {'Chunking':<55}  {'Retriever'}")
    print("-" * 100)

    for i, r in enumerate(all_results, start=1):
        c_label, r_label = _short_label(r["chunking"], r["retriever"])
        print(f"{i:>3}  {r['hit_rate']:>8.2%}  {r['mrr']:>8.4f}  {r['recall']:>8.2%}  {r['time_s']:>5.1f}s  {c_label:<55}  {r_label}")

    # ── Save full results to JSON ─────────────────────────────────────────
    results_path = os.path.join(SCRIPT_DIR, "grid_search_results.json")
    # Convert any non-serializable values
    for r in all_results:
        if "weights" in r.get("retriever", {}):
            r["retriever"]["weights"] = list(r["retriever"]["weights"])
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Full results saved to {results_path}")

    # ── Best config summary ───────────────────────────────────────────────
    if all_results:
        best = all_results[0]
        print("\n" + "=" * 100)
        print("🏆  BEST CONFIGURATION")
        print("=" * 100)
        print(f"  Hit Rate : {best['hit_rate']:.2%}")
        print(f"  MRR      : {best['mrr']:.4f}")
        print(f"  Recall   : {best['recall']:.2%}")
        print(f"  Chunking : {json.dumps(best['chunking'], ensure_ascii=False)}")
        print(f"  Retriever: {json.dumps(best['retriever'], ensure_ascii=False)}")
        print(f"  DB path  : {best['persist_dir']}")
        print("=" * 100)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n📁 Used {len(chroma_dirs_used)} vectorstore directories: {chroma_dirs_used}")


if __name__ == "__main__":
    main()
