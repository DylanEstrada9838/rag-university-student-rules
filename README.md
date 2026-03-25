# RAG – Reglamento General de Estudiantes (Tec de Monterrey)

A Retrieval-Augmented Generation (RAG) system built to query and retrieve information from the **Reglamento General de Estudiantes** PDF (Tecnológico de Monterrey).

## Features

- **PDF loading & preprocessing** – Loads the regulation PDF, removes headers, page numbers, blank pages, and the index page.
- **Semantic chunking** – Uses `SemanticChunker` for intelligent document splitting based on embedding similarity.
- **Hybrid retrieval** – Combines dense vector search (MMR) with sparse keyword search (BM25) via `EnsembleRetriever`.
- **Cross-encoder reranking** – Applies a multilingual reranker (`BAAI/bge-reranker-v2-m3`) for higher-precision top-k results.
- **Chroma vector store** – Persists embeddings locally for fast repeated queries.

## Project Structure

```
├── document.py         # PDF loading + document processing
├── process_doc.py      # Text cleaning functions (headers, page numbers, blanks)
├── chunking.py         # Semantic chunking + get_chunks() helper
├── embeddings.py       # HuggingFace embedding model configuration
├── vectorstore.py      # Create and load Chroma vector store
├── retriever.py        # Base, hybrid, and reranker retriever builders
├── testing/
│   ├── test_loader.py      # Test PDF loading pipeline
│   └── test_retriever.py   # Test full retrieval pipeline
├── documento/
│   └── reglamento-general-estudiantes-esp.pdf
├── requirements.txt
├── .env.example        # Template for environment variables
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-reglamento-tec.git
cd rag-reglamento-tec
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and add your HuggingFace token:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual token:

```
HF_TOKEN=hf_your_actual_token_here
```

You can get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Build the vector database

Open a Python shell or create a script:

```python
from vectorstore import create_vector_db

create_vector_db()
```

This will process the PDF, chunk it, generate embeddings, and persist the Chroma database in the `chroma_db/` folder.

### 6. Test the retrieval pipeline

```bash
python testing/test_retriever.py
```

## Usage Example

```python
from embeddings import get_embeddings
from vectorstore import load_vectorstore
from retriever import get_hybrid_retriever, get_reranker_retriever
from chunking import get_chunks

chunks = get_chunks()
embeddings = get_embeddings()
vectorstore = load_vectorstore(embeddings)

# Build the retriever
retriever = get_reranker_retriever(
    get_hybrid_retriever(vectorstore, chunks),
    top_n=3
)

# Query
docs = retriever.invoke("¿Qué implica la baja definitiva de un estudiante?")
for doc in docs:
    print(doc.page_content)
```

## Tech Stack

| Component | Library |
|---|---|
| PDF Loader | `PyMuPDF` via LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunking | `SemanticChunker` (LangChain Experimental) |
| Vector Store | `ChromaDB` |
| Sparse Retrieval | `BM25Retriever` (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Orchestration | `LangChain` |
