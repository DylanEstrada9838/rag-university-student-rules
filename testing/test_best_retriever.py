"""
test_best_retriever.py – Evaluate the best retriever configuration found by grid search.

Best config:
    Chunking : semantic (percentile=80, min_chunk_size=100)
    Retriever: hybrid (k=10, fetch_k=20, lambda_mult=0.5, weights=[0.7, 0.3])
    DB path  : ./chroma_db_v15
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from vectorstore import load_vectorstore
from chunking import get_chunks
from ground_truth import ground_truth
from retrieval_metrics import hit_rate, mrr, recall

# ── Best configuration ────────────────────────────────────────────────────
CHUNKING_CONFIG = {
    "method": "semantic",
    "breakpoint_threshold_type": "percentile",
    "breakpoint_threshold_amount": 80,
    "min_chunk_size": 100,
}

RETRIEVER_CONFIG = {
    "type": "hybrid",
    "k": 10,
    "fetch_k": 20,
    "lambda_mult": 0.5,
    "weights": [0.7, 0.3],
}

PERSIST_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db_v15')

# ── Setup ──────────────────────────────────────────────────────────────────
chunks = get_chunks(CHUNKING_CONFIG)
vectorstore = load_vectorstore(persist_dir=PERSIST_DIR)

vec_retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVER_CONFIG["k"]},
    search_type="mmr",
    fetch_k=RETRIEVER_CONFIG["fetch_k"],
    lambda_mult=RETRIEVER_CONFIG["lambda_mult"],
)

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=RETRIEVER_CONFIG["k"],
)

retriever = EnsembleRetriever(
    retrievers=[vec_retriever, bm25_retriever],
    weights=RETRIEVER_CONFIG["weights"],
)

# ── Evaluate ───────────────────────────────────────────────────────────────
results = []

print("=" * 70)
print("BEST RETRIEVER EVALUATION")
print("=" * 70)
print(f"  Chunking  : {CHUNKING_CONFIG}")
print(f"  Retriever : {RETRIEVER_CONFIG}")
print(f"  DB path   : {PERSIST_DIR}")
print("=" * 70)

for i, item in enumerate(ground_truth):
    query = item["question"]
    expected_pages = item["expected_pages"]

    retrieved_docs = retriever.invoke(query)
    retrieved_pages_ordered = [doc.metadata.get("page") for doc in retrieved_docs]

    hit = bool(set(expected_pages) & set(retrieved_pages_ordered))

    results.append({
        "expected_pages": expected_pages,
        "retrieved_pages": retrieved_pages_ordered,
        "retrieved_pages_ordered": retrieved_pages_ordered,
    })

    status = "✓ HIT" if hit else "✗ MISS"
    print(f"\nQ{i+1}: {query}")
    print(f"  Expected pages : {expected_pages}")
    print(f"  Retrieved pages: {retrieved_pages_ordered}")
    print(f"  Result         : {status}")

# ── Final metrics ──────────────────────────────────────────────────────────
final_hit_rate = hit_rate(results)
final_mrr = mrr(results)
final_recall = recall(results)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Hit Rate : {final_hit_rate:.2%}")
print(f"  MRR      : {final_mrr:.4f}")
print(f"  Recall   : {final_recall:.2%}")
print("=" * 70)
