"""Load html from files, clean up, split, ingest into vector db."""
import logging
import os
import re
from parser import langchain_docs_extractor

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    # Initialize the DefaultEmbedding class with the desired parameters
    # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html
    # https://qdrant.github.io/fastembed/examples/Supported_Models/
    return FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5", max_length=512)


def load_docs():
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
    loader = PyPDFDirectoryLoader("./pdf/")
    docs = loader.load()
    return docs


def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    
    embeddings = get_embeddings_model()

    docs = load_docs()
    logger.info(f"Loaded {len(docs)} docs")
    with open('./docs.txt', 'w') as f:
        f.write('\n'.join(map(str, docs)))

    docs_splitted = text_splitter.split_documents(docs)
    logger.info(f"docs_splitted")
    with open('./docs_splitted.txt', 'w') as f:
        f.write('\n'.join(map(str, docs_splitted)))

    docs_transformed = [doc for doc in docs_splitted if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # it will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    logger.info(f"docs_transformed")
    with open('./docs_transformed.txt', 'w') as f:
        f.write('\n'.join(map(str, docs_transformed)))

    db = FAISS.from_documents(docs_transformed, embeddings)
    db.save_local("faiss_index")


if __name__ == "__main__":
    ingest_docs()
