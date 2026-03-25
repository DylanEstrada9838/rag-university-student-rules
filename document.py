import os
from langchain_community.document_loaders import PyMuPDFLoader
from process_doc import process_documents

pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documento", "reglamento-general-estudiantes-esp.pdf")

def get_doc():
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    docs = process_documents(docs)
    return docs
