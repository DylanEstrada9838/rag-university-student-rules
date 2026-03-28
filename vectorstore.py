import os
import shutil
import glob
from dotenv import load_dotenv
from langchain_chroma import Chroma
from chunking import get_chunks
from embeddings import get_embeddings


def _next_version_dir(base_dir="."):
    """Return the next available chroma_db_vN directory name."""
    existing = glob.glob(os.path.join(base_dir, "chroma_db_v*"))
    max_version = 0
    for d in existing:
        basename = os.path.basename(d)
        try:
            version = int(basename.split("_v")[-1])
            max_version = max(max_version, version)
        except ValueError:
            continue
    return os.path.join(base_dir, f"chroma_db_v{max_version + 1}")


def create_vector_db(chunking_config=None, persist_dir=None):
    """
    Create a new versioned Chroma vector store.

    Args:
        chunking_config: dict passed to get_chunks() to control chunking.
        persist_dir:     explicit directory path. If None, auto-increments
                         to the next chroma_db_vN version.

    Returns:
        (vectorstore, persist_dir, chunks)
    """
    load_dotenv()
    chunks = get_chunks(chunking_config)
    embeddings = get_embeddings()

    if persist_dir is None:
        persist_dir = _next_version_dir()

    # Always start fresh for this version
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore, persist_dir, chunks


def load_vectorstore(persist_dir="./chroma_db_v15"):
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())