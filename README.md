# RAG – Reglamento General de Estudiantes (Tec de Monterrey)

A Retrieval-Augmented Generation (RAG) system built to query and retrieve information from the **Reglamento General de Estudiantes** PDF (Tecnológico de Monterrey).

## Features

- **PDF loading & preprocessing** – Loads the regulation PDF, removes headers, page numbers, blank pages, and the index page.
- **Semantic chunking** – Uses `SemanticChunker` for intelligent document splitting based on embedding similarity.
- **Hybrid retrieval** – Combines dense vector search (MMR) with sparse keyword search (BM25) via `EnsembleRetriever`.
- **Cross-encoder reranking** – Applies a multilingual reranker (`BAAI/bge-reranker-v2-m3`) for higher-precision top-k results.
- **Chroma vector store** – Persists embeddings locally for fast repeated queries.
- **Local LLM Output** – Leverages `Ollama` (Llama 3) locally to synthesize concise, context-aware answers natively through LangChain runnables.

## Project Structure

```
├── document.py         # PDF loading + document processing
├── process_doc.py      # Text cleaning functions (headers, page numbers, blanks)
├── chunking.py         # Semantic chunking + get_chunks() helper
├── embeddings.py       # HuggingFace embedding model configuration
├── vectorstore.py      # Create and load Chroma vector store
├── retriever.py        # Base, hybrid, and reranker retriever builders
├── llm.py              # LLM integration and full RAG generation chain
├── testing/
│   ├── test_loader.py      # Test PDF loading pipeline
│   ├── test_retriever.py   # Test full retrieval pipeline
│   └── test_llm.py         # Test standalone LLM and full RAG flow
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

### 5. Install Ollama for Local Generation

This project uses [Ollama](https://ollama.com/) for entirely local LLM generation. 
1. Install Ollama according to your operating system.
2. Download and run the `llama3` model:
```bash
ollama run llama3
```

### 6. Build the vector database

Open a Python shell or create a script:

```python
from vectorstore import create_vector_db

create_vector_db()
```

This will process the PDF, chunk it, generate embeddings, and persist the Chroma database in the `chroma_db/` folder.

### 7. Run Tests

Test the retriever logic to ensure chunks load correctly:
```bash
python testing/test_retriever.py
```

Test the full RAG generation pipeline using the LLM:
```bash
python testing/test_llm.py
```

## Usage Example

```python
from vectorstore import load_vectorstore
from retriever import get_hybrid_retriever, get_reranker_retriever
from chunking import get_chunks
from llm import get_rag_chain

chunks = get_chunks()
vectorstore = load_vectorstore()

# Build the retriever
retriever = get_reranker_retriever(
    get_hybrid_retriever(vectorstore, chunks),
    top_n=3
)

# Build the final LLM-powered chain
chain = get_rag_chain(retriever)

# Query
query = "¿Qué implica la baja definitiva de un estudiante?"
response = chain.invoke(query)

print(response)
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
| Local LLM | `Ollama` (`llama3`) |
| Orchestration | `LangChain` |
