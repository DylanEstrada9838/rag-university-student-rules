import re

def remove_index_page(docs):
    """
    Removes the index page (where metadata 'page' is 8).
    """
    return [doc for doc in docs if doc.metadata.get('page') != 8]

# def remove_header(text):
#     """
#     Removes the string 'REGLAMENTO GENERAL DE ESTUDIANTES' at the beginning of the page.
#     """
#     return re.sub(r'^\s*REGLAMENTO GENERAL DE ESTUDIANTES\s*', '', text, flags=re.IGNORECASE)

def remove_header(text):
    pattern = r'^\s*REGLAMENTO\s+GENERAL\s+DE\s+ESTUDIANTES\s*'
    
    return re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE).strip()

def remove_page_numbers(text):
    """
    Removes lines that contain only digits (which are typically page numbers).
    """
    return re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

def clean_newlines(text):
    """
    Cleans up any multiple consecutive newlines left over from removals.
    """
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def is_blank_page(text):
    """
    Returns True if the page is completely empty or just contains whitespace,
    without using .strip().
    """
    return text == "" or text.isspace()

def process_documents(docs):
    """
    Processes a list of Langchain Document objects by removing standalone page numbers 
    from their page_content, removes any pages that end up being entirely blank,
    removes the 'REGLAMENTO GENERAL DE ESTUDIANTES' header, and drops the index page.
    """
    # 1. Filter out the index page
    docs = remove_index_page(docs)
    
    processed_docs = []
    
    for doc in docs:
        # 2. Text preprocessing steps
        doc.page_content = remove_header(doc.page_content)
        doc.page_content = remove_page_numbers(doc.page_content)
        doc.page_content = clean_newlines(doc.page_content)
        
        # 3. Filter blank pages without using strip
        if not is_blank_page(doc.page_content):
            processed_docs.append(doc)
            
    return processed_docs
