import os
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


# Add the parent directory to sys.path to easily import the loader module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectorstore import load_vectorstore
from retriever import get_hybrid_retriever, get_reranker_retriever
from chunking import get_chunks

# Obtain chunks directly
chunks = get_chunks()

#Vector Store
vectorstore = load_vectorstore()

#Retriever
#retriever = get_base_retriever(vectorstore)
#hybrid_retrieved = get_hybrid_retriever(vectorstore, chunks)
reranked_retriever = get_reranker_retriever(get_hybrid_retriever(vectorstore, chunks), top_n=3)

# Test retrieval
query = "Qué implica la baja definitiva de un estudiante?"
#retrieved_docs = retriever.invoke(query)
#retrieved_docs = hybrid_retrieved.invoke(query)
retrieved_docs = reranked_retriever.invoke(query)

print(f"Retrieved {len(retrieved_docs)} chunks")
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Chunk {i+1} (page {doc.metadata.get('page', '?')}) ---")
    print(doc.page_content)


