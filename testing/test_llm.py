import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chunking import get_chunks
from vectorstore import load_vectorstore
from retriever import get_hybrid_retriever, get_reranker_retriever
from llm import get_llm, get_rag_chain


# --- Test full RAG chain ---
print("\n" + "=" * 60)
print("TEST: Full RAG chain")
print("=" * 60)

chunks = get_chunks()
vectorstore = load_vectorstore()
retriever = get_reranker_retriever(get_hybrid_retriever(vectorstore, chunks), top_n=3)

chain = get_rag_chain(retriever)

query = "¿Qué implica la baja definitiva de un estudiante?"
print(f"\nQuery: {query}\n")

response = chain.invoke(query)
print("Response:")
print(response)
