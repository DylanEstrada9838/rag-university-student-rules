from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from embeddings import get_embeddings
from document import get_doc

embeddings = get_embeddings()


def recursive_chunker(doc, chunk_size=512, chunk_overlap=128, separators=None):
    """
    Split documents using RecursiveCharacterTextSplitter.

    Args:
        doc:           list of Document objects
        chunk_size:    maximum size of each chunk (default 512)
        chunk_overlap: overlap between consecutive chunks (default 128)
        separators:    list of separator strings (default ["\n\n", "\n", " ", ""])
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )
    return splitter.split_documents(doc)


def semantic_chunker(doc, breakpoint_threshold_type="percentile",
                     breakpoint_threshold_amount=80, min_chunk_size=100):
    """
    Split documents using SemanticChunker.

    Args:
        doc:                          list of Document objects
        breakpoint_threshold_type:    "percentile", "standard_deviation", or "interquartile"
        breakpoint_threshold_amount:  numeric threshold value (default 80)
        min_chunk_size:               minimum characters per chunk (default 100)
    """
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        min_chunk_size=min_chunk_size,
    )
    return splitter.split_documents(doc)


def get_chunks(chunking_config=None):
    """
    Load the document and chunk it according to chunking_config.

    chunking_config dict keys:
        method: "recursive" | "semantic"  (default: "semantic")
        + any kwargs accepted by the corresponding chunker function.
    """
    if chunking_config is None:
        chunking_config = {}

    doc = get_doc()
    method = chunking_config.get("method", "recursive")

    if method == "recursive":
        return recursive_chunker(
            doc,
            chunk_size=chunking_config.get("chunk_size", 512),
            chunk_overlap=chunking_config.get("chunk_overlap", 128),
            separators=chunking_config.get("separators"),
        )
    elif method == "semantic":
        return semantic_chunker(
            doc,
            breakpoint_threshold_type=chunking_config.get("breakpoint_threshold_type", "percentile"),
            breakpoint_threshold_amount=chunking_config.get("breakpoint_threshold_amount", 80),
            min_chunk_size=chunking_config.get("min_chunk_size", 100),
        )
    else:
        raise ValueError(f"Unknown chunking method: {method}")