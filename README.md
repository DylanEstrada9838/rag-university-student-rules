# RAG – Reglamento General de Estudiantes (Tec de Monterrey)

A Retrieval-Augmented Generation (RAG) system built to query and retrieve information from the **Reglamento General de Estudiantes** PDF (Tecnológico de Monterrey).

## Features

- **PDF loading & preprocessing** – Loads the regulation PDF, removes headers, page numbers, blank pages, and the index page.
- **Configurable chunking** – Supports both `RecursiveCharacterTextSplitter` and `SemanticChunker`, selectable via config dict.
- **Hybrid retrieval** – Combines dense vector search (MMR) with sparse keyword search (BM25) via `EnsembleRetriever`.
- **Cross-encoder reranking** – Applies a multilingual reranker (`BAAI/bge-reranker-v2-m3`) for higher-precision top-k results.
- **Versioned Chroma vector store** – Each build creates a new `chroma_db_vN/` directory, making it easy to compare configurations.
- **Hyperparameter grid search** – Automated evaluation of chunking × retriever combinations with random search support.
- **Retrieval metrics** – Hit Rate, MRR, and Recall evaluated against a ground truth dataset.
- **Local LLM output** – Leverages `Ollama` (Llama 3) locally to synthesize concise, context-aware answers via LangChain runnables.

## Project Structure

```
├── document.py             # PDF loading + document processing
├── process_doc.py          # Text cleaning (headers, page numbers, blanks)
├── chunking.py             # Recursive & semantic chunkers + get_chunks()
├── embeddings.py           # HuggingFace embedding model configuration
├── vectorstore.py          # Create versioned & load Chroma vector stores
├── retriever.py            # Base, hybrid, and reranker retriever builders
├── llm.py                  # LLM integration and full RAG generation chain
├── testing/
│   ├── ground_truth.py         # Ground truth Q&A pairs (metadata pages)
│   ├── retrieval_metrics.py    # Hit Rate, MRR, Recall metrics
│   ├── grid_search.py          # Hyperparameter grid/random search
│   ├── test_best_retriever.py  # Evaluate the best configuration
│   ├── evaluation_retriever.py # Single-config retriever evaluation
│   ├── test_loader.py          # Test PDF loading pipeline
│   ├── test_retriever.py       # Test full retrieval pipeline
│   └── test_llm.py             # Test standalone LLM and full RAG flow
├── documento/
│   └── reglamento-general-estudiantes-esp.pdf
├── requirements.txt
├── .env.example            # Template for environment variables
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

```python
from vectorstore import create_vector_db

# Default semantic chunking → creates chroma_db_v1/
vectorstore, persist_dir, chunks = create_vector_db()

# Or with custom chunking config:
vectorstore, persist_dir, chunks = create_vector_db(
    chunking_config={
        "method": "semantic",
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 80,
        "min_chunk_size": 100,
    }
)
```

Each call auto-creates a new versioned directory (`chroma_db_v1/`, `chroma_db_v2/`, …).

### 7. Run Tests

Test the retriever logic:
```bash
python testing/test_retriever.py
```

Test the full RAG generation pipeline:
```bash
python testing/test_llm.py
```

Evaluate retrieval with the best config:
```bash
python testing/test_best_retriever.py
```

## Hyperparameter Search

Find the best chunking + retriever configuration automatically:

```bash
python testing/grid_search.py
```

This evaluates combinations of:
- **Chunking**: `recursive` (chunk_size, overlap) and `semantic` (threshold type/amount, min size)
- **Retriever**: `base` (similarity/MMR), `hybrid` (BM25 + dense), `reranker_base`, `reranker_hybrid`

Uses **random search** (20 samples) when total combos exceed the threshold. Results are saved to `testing/grid_search_results.json`.

### Best Configuration Found

| Metric | Value |
|--------|-------|
| Hit Rate | 70.00% |
| MRR | 0.1460 |
| Recall | 65.00% |

| Parameter | Value |
|-----------|-------|
| Chunking | Semantic (percentile=80, min_chunk_size=100) |
| Retriever | Hybrid (k=10, fetch_k=20, λ=0.5, weights=[0.7, 0.3]) |
| Vector Store | `chroma_db_v15` |

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
| Chunking | `RecursiveCharacterTextSplitter` / `SemanticChunker` |
| Vector Store | `ChromaDB` (versioned) |
| Sparse Retrieval | `BM25Retriever` (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Local LLM | `Ollama` (`llama3`) |
| Orchestration | `LangChain` |
