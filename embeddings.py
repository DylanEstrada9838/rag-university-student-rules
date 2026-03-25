
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight
        # model_name="BAAI/bge-large-en-v1.5",               # Higher quality
        model_kwargs={"device": "cpu"},   # Use "cuda" for GPU
        encode_kwargs={"normalize_embeddings": True},
    )