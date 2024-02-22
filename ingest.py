"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor

from langchain_community.vectorstores import Qdrant
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import FAISS
# from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def get_embeddings_model() -> Embeddings:
    # return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)

def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text") # ValueError: shapes (3357,768) and (4096,) not aligned: 768 (dim 1) != 4096 (dim 0)
    # return OllamaEmbeddings(model="llama2")


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


def load_sitemap_docs():
    return SitemapLoader(
        # "https://intranet.dkfz.de/en/sitemap.xml?sitemap=pages&cHash=7e244f5c28a54d2a65d08c0842148c7d",
        "https://www.dkfz.de/de/sitemap.xml",
        # filter_urls=["https://intranet.dkfz.de/"],
        filter_urls=["https://www.dkfz.de/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_recursiveurl_docs():
    return RecursiveUrlLoader(
        url="https://webcms47.inet.dkfz-heidelberg.de/",
        max_depth=7,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            # r"(?:[\#'\"]|\/[\#'\"])"
            r"(?:[\#'\"]|\/[\#'\"]|\/export\/|\/revisions\/|\/attachments\/|\/users\/|\/uploads\/|\/dist\/|\+496221422376|\+496221422323)"
        ),
        check_response_status=True,
    ).load()


def ingest_docs():
    # WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    # WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    # RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # client = weaviate.Client(
    #     url=WEAVIATE_URL,
    #     auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    # )
    embeddings = get_embeddings_model()
    # vectorstore = Weaviate(
    #     client=client,
    #     index_name=WEAVIATE_DOCS_INDEX_NAME,
    #     text_key="text",
    #     embedding=embedding,
    #     by_text=False,
    #     attributes=["source", "title"],
    # )

    # record_manager = SQLRecordManager(
    #     f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    # )
    # record_manager.create_schema()

    docs_from_sitemap = load_sitemap_docs()
    logger.info(f"Loaded {len(docs_from_sitemap)} docs from sitemap")
    # docs_from_recursiveurl = load_recursiveurl_docs()
    # logger.info(f"Loaded {len(docs_from_recursiveurl)} docs from recursiveurl")
    # docs_from_langsmith = load_langsmith_docs()
    # logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    docs_transformed = text_splitter.split_documents(
        docs_from_sitemap
        # +
        # docs_from_recursiveurl
        # +
        # docs_from_langsmith
    )
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    # vector = FAISS.from_documents(docs_transformed, embedding)
    qdrant = Qdrant.from_documents(
        docs_transformed,
        embeddings,
        force_recreate=True,
        path="./local_qdrant",
        collection_name="my_documents",
    )

    # indexing_stats = index(
    #     docs_transformed,
    #     record_manager,
    #     vectorstore,
    #     cleanup="full",
    #     source_id_key="source",
    #     force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    # )

    # logger.info(f"Indexing stats: {indexing_stats}")
    # num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    # logger.info(
    #     f"LangChain now has this many vectors: {num_vecs}",
    # )


if __name__ == "__main__":
    ingest_docs()
