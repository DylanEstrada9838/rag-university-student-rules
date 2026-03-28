"""
test_best_retriever.py – Evaluate the best retriever configuration found by grid search.

Best config:
  Hit Rate : 90.00%
  MRR      : 0.7333
  Recall   : 85.00%
  Chunking : recursive (chunk_size=512, chunk_overlap=128)
  Retriever: reranker_hybrid (k=10, fetch_k=20, lambda_mult=0.3, weights=[0.6, 0.4], top_n=5)
  DB path  : testing/chroma_db_v3
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
from retriever import get_reranker_retriever
from ground_truth import ground_truth
from retrieval_metrics import hit_rate, mrr, recall

# ── Best configuration ────────────────────────────────────────────────────
CHUNKING_CONFIG = {
    "method": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 128,
}

RETRIEVER_CONFIG = {
    "type": "reranker_hybrid",
    "k": 10,
    "fetch_k": 20,
    "lambda_mult": 0.3,
    "weights": [0.6, 0.4],
    "top_n": 5,
}

PERSIST_DIR = os.path.join(os.path.dirname(__file__), 'chroma_db_v3')

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

hybrid_retriever = EnsembleRetriever(
    retrievers=[vec_retriever, bm25_retriever],
    weights=RETRIEVER_CONFIG["weights"],
)

retriever = get_reranker_retriever(hybrid_retriever, top_n=RETRIEVER_CONFIG["top_n"])

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
