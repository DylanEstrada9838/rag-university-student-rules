from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from embeddings import get_embeddings
from document import get_doc

embeddings = get_embeddings()


def chunker(doc):

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=512,
    #     chunk_overlap = 64,
    #     length_function = len,
    #     separators=["\n\n", "\n", " ", ""],
    # )

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=200
    )
    chunks = splitter.split_documents(doc)
    return chunks


def get_chunks():
    doc = get_doc()
    chunks = chunker(doc)
    return chunks