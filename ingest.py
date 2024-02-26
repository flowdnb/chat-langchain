"""Load html from files, clean up, split, ingest into vector db."""
import logging
import os
import re
from parser import langchain_docs_extractor

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    # Initialize the DefaultEmbedding class with the desired parameters
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en", max_length=512)


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_docs():
    docs_intranet=SitemapLoader(
        "https://intranet.dkfz.de/en/sitemap.xml?sitemap=pages&cHash=7e244f5c28a54d2a65d08c0842148c7d",
        filter_urls=["https://intranet.dkfz.de/"],
        continue_on_failure=False,
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()
    # docs_homepage=SitemapLoader(
    #     "https://www.dkfz.de/de/sitemap.xml",
    #     filter_urls=["https://www.dkfz.de/en/"],
    #     continue_on_failure=True,
    #     parsing_function=langchain_docs_extractor,
    #     default_parser="lxml",
    #     bs_kwargs={
    #         "parse_only": SoupStrainer(
    #             name=("article", "title", "html", "lang", "content")
    #         ),
    #     },
    #     meta_function=metadata_extractor,
    # ).load()
    docs_wiki=RecursiveUrlLoader(
        url="https://webcms47.inet.dkfz-heidelberg.de/",
        max_depth=7,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=False,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"]|\/export\/|\/revisions|\/revisions\/|\/attachments\/|\/user\/|\/search\?.*|\/uploads\/|\/dist\/|\/references|\+496221422376|\+496221422323)"
        ),
        check_response_status=True,
    ).load()
    docs = docs_intranet + docs_homepage + docs_wiki
    return docs


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


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
