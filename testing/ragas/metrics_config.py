"""
metrics_config.py — RAGAS metrics configured for the Reglamento Tec RAG pipeline.

All metrics use:
  - LLM judge : local Ollama llama3  (same model as the RAG chain)
  - Embeddings: HuggingFace all-MiniLM-L6-v2  (same embedder as the vectorstore)

NOTE: RAGAS wraps LangChain LLMs via LangchainLLMWrapper / LangchainEmbeddingsWrapper.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)


def get_ragas_llm():
    """Returns the LangChain LLM wrapped for RAGAS (local Ollama llama3)."""
    llm = ChatOllama(model="llama3", temperature=0.0)
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    """Returns the HuggingFace embedder wrapped for RAGAS."""
    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def get_all_metrics():
    """
    Returns the full list of RAGAS metrics for end-to-end RAG evaluation.

    Metrics:
        AnswerRelevancy   – Is the answer relevant to the question?
        Faithfulness      – Is the answer grounded in the retrieved context?
        ContextPrecision  – Are retrieved chunks actually relevant? (needs reference)
        ContextRecall     – Does the context cover the reference answer? (needs reference)
        AnswerCorrectness – Does the answer match the reference answer? (needs reference)
    """
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    metrics = [
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        Faithfulness(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    ]
    return metrics


def get_context_only_metrics():
    """
    Returns only the context-level RAGAS metrics (no LLM answer generation needed).
    Used by the retriever grid search to avoid running the full RAG chain.

    Metrics:
        ContextPrecision – Are retrieved chunks relevant?
        ContextRecall    – Does context cover the reference answer?
    """
    ragas_llm = get_ragas_llm()

    metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    return metrics
