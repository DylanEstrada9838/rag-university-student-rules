import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking import get_chunks
from vectorstore import load_vectorstore
from retriever import get_hybrid_retriever, get_reranker_retriever
from ground_truth import ground_truth
from retrieval_metrics import hit_rate, mrr, recall

# Setup
chunks = get_chunks()
vectorstore = load_vectorstore()
retriever = get_reranker_retriever(get_hybrid_retriever(vectorstore, chunks), top_n=5)

# Evaluate
results = []

print("=" * 70)
print("RETRIEVER EVALUATION")
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

    status = "HIT" if hit else "MISS"
    print(f"\nQ{i+1}: {query}")
    print(f"  Expected pages : {expected_pages}")
    print(f"  Retrieved pages: {retrieved_pages_ordered}")
    print(f"  Result         : {status}")

# Final metrics
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
