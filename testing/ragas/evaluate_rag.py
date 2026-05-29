"""
evaluate_rag.py — End-to-end RAGAS evaluation of the RAG pipeline using the best config.

Best config (from grid_search_results.json — rank #1):
  Hit Rate : 90%    MRR : 0.7333    Recall : 85%
  Chunking : recursive (chunk_size=512, chunk_overlap=128)
  Retriever: reranker_hybrid (k=10, fetch_k=20, lambda_mult=0.3, weights=[0.6,0.4], top_n=5)
  DB path  : testing/chroma_db_v3

Usage:
    cd <project_root>
    python testing/ragas/evaluate_rag.py
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
from llm import get_rag_chain

from ragas import evaluate
from ragas.run_config import RunConfig

# Local imports (within testing/ragas/)
sys.path.append(os.path.dirname(__file__))
from dataset import build_evaluation_dataset
from metrics_config import get_all_metrics

# ── Best configuration ─────────────────────────────────────────────────────────
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

PERSIST_DIR = os.path.join(os.path.dirname(__file__), '..', 'chroma_db_v3')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'ragas_results.json')

# ── Setup retriever ────────────────────────────────────────────────────────────
print("=" * 70)
print("RAGAS END-TO-END RAG EVALUATION")
print("=" * 70)
print(f"  Chunking  : {CHUNKING_CONFIG}")
print(f"  Retriever : {RETRIEVER_CONFIG}")
print(f"  DB path   : {PERSIST_DIR}")
print("=" * 70)
print()

chunks = get_chunks(CHUNKING_CONFIG)
vectorstore = load_vectorstore(persist_dir=os.path.normpath(PERSIST_DIR))

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

# ── Build RAG chain ────────────────────────────────────────────────────────────
rag_chain = get_rag_chain(retriever)

# ── Build RAGAS dataset (runs retriever + LLM for each question) ───────────────
dataset = build_evaluation_dataset(retriever, rag_chain)

# ── Run RAGAS evaluation ───────────────────────────────────────────────────────
print("Running RAGAS evaluation… (this may take several minutes)")
print("Metrics: AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall, AnswerCorrectness\n")

import time
import pandas as pd
from ragas import EvaluationDataset

metrics = get_all_metrics()
results_list = []
total_start = time.time()

# Extract samples from EvaluationDataset for individual evaluation
samples = dataset.samples if hasattr(dataset, 'samples') else list(dataset)
num_samples = len(samples)

for i, sample in enumerate(samples):
    print(f"\nEvaluating Q{i+1}/{num_samples}...")
    q_start = time.time()
    
    single_ds = EvaluationDataset(samples=[sample])
    res = evaluate(
        dataset=single_ds, 
        metrics=metrics,
        run_config=RunConfig(max_workers=1, timeout=1200)
    )
    
    q_time = time.time() - q_start
    print(f"✓ Q{i+1} completed in {q_time:.2f} seconds.")
    
    df_row = res.to_pandas()
    df_row["eval_time_s"] = q_time
    results_list.append(df_row)

total_time = time.time() - total_start
print(f"\nTotal evaluation time: {total_time:.2f} seconds.")

# ── Display results ────────────────────────────────────────────────────────────
df = pd.concat(results_list, ignore_index=True)

print("\n" + "=" * 70)
print("RAGAS RESULTS — PER QUESTION")
print("=" * 70)
pd_cols = ["user_input", "answer_relevancy", "faithfulness",
           "context_precision", "context_recall", "answer_correctness", "eval_time_s"]

available_cols = [c for c in pd_cols if c in df.columns]
print(df[available_cols].to_string(index=True))

print("\n" + "=" * 70)
print("RAGAS RESULTS — AGGREGATE")
print("=" * 70)
metric_names = {
    "answer_relevancy": "Answer Relevancy  ",
    "faithfulness": "Faithfulness      ",
    "context_precision": "Context Precision ",
    "context_recall": "Context Recall    ",
    "answer_correctness": "Answer Correctness",
    "eval_time_s": "Avg Time per Q (s)",
}
for col, label in metric_names.items():
    if col in df.columns:
        print(f"  {label}: {df[col].mean():.4f}")

print("=" * 70)

# ── Save results to JSON ───────────────────────────────────────────────────────
output = {
    "config": {
        "chunking": CHUNKING_CONFIG,
        "retriever": RETRIEVER_CONFIG,
        "persist_dir": str(PERSIST_DIR),
    },
    "aggregate": {
        col: float(df[col].mean())
        for col in metric_names
        if col in df.columns
    },
    "per_question": df[available_cols].to_dict(orient="records"),
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nResults saved → {OUTPUT_FILE}")
