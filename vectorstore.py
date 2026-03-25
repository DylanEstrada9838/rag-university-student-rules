import os
import shutil
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chunking import get_chunks
from embeddings import get_embeddings

def create_vector_db():
    load_dotenv()
    chunks = get_chunks()
    embeddings = get_embeddings()
    persist_dir = "./chroma_db"
    
    # Check if the directory exists and delete it if it does
    if os.path.exists(persist_dir):
        print(f"Removing existing vectorstore at {persist_dir}...")
        shutil.rmtree(persist_dir)
        
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

def load_vectorstore():
    return Chroma(persist_directory="./chroma_db", embedding_function=get_embeddings())