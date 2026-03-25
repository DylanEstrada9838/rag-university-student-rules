import sys
import os

# Add the parent directory to sys.path to easily import the loader module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from document import get_doc

docs = get_doc()

print(len(docs))
# The slice docs[0:10] returns a list, so we iterate to print their content
for doc in docs[34:37]:
    print(f"Page {doc.metadata.get('page', '?')}:\n{doc.page_content}\n")