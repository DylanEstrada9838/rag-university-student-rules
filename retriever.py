from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever

def get_base_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={"k": 6}, 
        search_type="mmr",
        fetch_k=12,
        lambda_mult=0.3,
        )

def get_hybrid_retriever(vectorstore,chunks):
    vec_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}, 
        search_type="mmr",
        fetch_k=20,
        lambda_mult=0.3,
        )
    bm25_retriever = BM25Retriever.from_documents(
        documents=chunks,
        k=10,
        k1=1.2,
        b=0.75,
    )
    return EnsembleRetriever(
        retrievers=[vec_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

def get_reranker_retriever(base_retriever, top_n=5):
    """
    Wraps an existing retriever with a Cross-Encoder reranker.
    Uses a multilingual model since documents and queries are in Spanish.
    """
    model = HuggingFaceCrossEncoder(
        # BAAI/bge-reranker-v2-m3 is excellent for multilingual queries
        model_name="BAAI/bge-reranker-v2-m3",
        model_kwargs={"device": "cpu"}
    )
    compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    return compression_retriever