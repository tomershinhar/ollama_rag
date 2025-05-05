import os
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PythonLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
# List of URLs to load documents from
urls_default_list = [
    "<https://lilianweng.github.io/posts/2023-06-23-agent/>",
    "<https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>",
    "<https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/>",
]
def load_from_urls(urls):
    # Load documents from the URLs
    if isinstance(urls, list):
        urls = urls + urls_default_list
    elif isinstance(urls, str):
        urls = urls_default_list.append(urls)
    else:
        urls = urls_default_list
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    return docs_list

def load_from_directories(paths):
    # Load the document
    pdf_files = []
    text_files = []
    python_files = []
    for path in paths:
        directory = os.fsencode(path)
        for file in os.listdir(directory):
            filename = os.fsdecode(b'/'.join([directory, file]))
            if filename.endswith(".pdf"): 
                pdf_files.append(PyPDFLoader(filename).load())
            elif filename.endswith(".py"):
                python_files.append(PythonLoader(filename).load())
            elif filename.endswith(".txt"):
                text_files.append(TextLoader(filename).load())
    documents = (
        [item for sublist in pdf_files for item in sublist] + 
        [item for sublist in text_files for item in sublist] + 
        [item for sublist in python_files for item in sublist]
    )


    return documents
    

def load_docs(urls=None, directories=None):
    docs_list = []
    if urls is not None:
        web_docs_list = load_from_urls(urls)
        docs_list += web_docs_list
    if directories is not None:
        local_docs_list = load_from_directories(directories)
        docs_list += local_docs_list
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits
