from datetime import datetime
from pathlib import Path
from typing import List
from langchain_chroma import Chroma

ROOT = Path(__file__).parents[1]

import logging

logging.basicConfig(
    format="[%(asctime)s]-%(levelname)s %(message)s",
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

import os, json
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class OpenAICompatibleEmbeddings(Embeddings):
    def __init__(
        self, api_base: str, api_key: str, model: str = "openai/Qwen3-Embedding-8B"
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def generate_embedding(self, texts: List[str]):

        from litellm import embedding

        start = datetime.now()
        logger.info(f"Embedding sentences: {texts}")
        response = embedding(
            model=self.model, api_base=self.api_base, api_key=self.api_key, input=texts
        )

        embeddings = [r["embedding"] for r in response.get("data", [])]
        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Sentences: {texts} embedded in {duration:.2f} s")
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.generate_embedding([query])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.generate_embedding(texts)


embeddings = OpenAICompatibleEmbeddings(
    model="openai/Qwen3-Embedding-8B",
    api_base=os.getenv("GITEE_API_BASE"),
    api_key=os.getenv("GITEE_API_KEY"),
)


def get_or_create_vectorstore(
    collection_name: str,
):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        collection_configuration={"hnsw": {"space": "cosine"}},
        persist_directory="./chroma_db",
    )

    return vector_store


def create_sample_docs():
    docs = [
        Document(page_content="你好", metadata={"source": "test"}),
        Document(page_content="世界很大，人很多", metadata={"source": "test"}),
        Document(page_content="你好世界", metadata={"source": "test"}),
        Document(
            page_content="如果有一天，我变得很有钱，我会做什么",
            metadata={"source": "test"},
        ),
        Document(
            page_content="爱是一道光，照进了我漆黑的灵魂", metadata={"source": "test"}
        ),
        Document(page_content="凡人修仙传", metadata={"source": "test"}),
        Document(page_content="this is a test", metadata={"source": "test"}),
        Document(page_content="hello world", metadata={"source": "test"}),
    ]
    return docs


def filter_k2_doc_items(label: str = "K2工作流平台"):
    file_paht = ROOT / "datasets" / "k2docs.json"
    with open(file_paht, "r", encoding="utf-8") as f:
        all_item = json.load(f)
    logger.info(f"Loaded {len(all_item)} items from {file_paht}")
    k2_items = [item for item in all_item if item["label"] == label]
    # Write to oakb_k2.json
    logger.info(f"Writing {len(k2_items)} items to k2docs.json")
    with open(ROOT / "datasets" / "oakb_k2.json", "w", encoding="utf-8") as f:
        json.dump(k2_items, f, ensure_ascii=False, indent=2)
    logger.info(f"Finished writing {len(k2_items)} items to oakb_k2.json")


def init_k2_doc_vectorstore(collection_name: str = "k2docs", drop_first: bool = False):
    logger.info(f"Getting vectorstore with collection: {collection_name}")
    vector_store = get_or_create_vectorstore(collection_name)
    if drop_first:
        logger.info(f"Deleting collection {collection_name} and creating new one")
        vector_store.reset_collection()

    # Read items from oakb_k2.json
    file_path = ROOT / "datasets" / "k2docs.json"
    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    logger.info(f"Loaded {len(items)} items from {Path(file_path).name}")

    # Building documents from items
    logger.info(f"Building documents for {len(items)} items")
    docs = [
        Document(
            page_content=item["content"],
            metadata={
                "label": item["label"],
                "title": item["title"],
                "id": item["id"],
                "content": item["content"]
            },
        )
        for item in items
    ]
    logger.info(f"Adding {len(docs)} documents to vectorstore")
    # Add by batch with 5 documents each time
    for i in range(0, len(docs), 5):
        logger.info(f"Processing batch {i // 5} with {len(docs[i : i + 5])} documents")
        vector_store.add_documents(docs[i : i + 5])
    logger.info(f"Finished adding {len(docs)} documents to vectorstore")
    return vector_store


def k2_doc_vectorstore_similarity_search(
    user_query: str,
    k: int = 5,
    score_threshold: float | None = None,
    collection_name: str = "k2docs",
):
    logger.info(f"Getting vectorstore with collection: {collection_name}")
    vector_store = get_or_create_vectorstore(collection_name)
    logger.info(f"Searching for `{user_query}` in vectorstore")
    results = vector_store.similarity_search_with_relevance_scores(user_query, k=k)
    logger.info(f"Fetched {len(results)} results for `{user_query}`")
    if score_threshold:
        results = [res for res, score in results if score >= score_threshold]
    logger.info(f"Filtered {len(results)} results for `{user_query}`")
    return results


if __name__ == "__main__":
    # ---------------------------------------
    # VectorStore
    # ---------------------------------------

    # vector_store = get_vectorstore("test")
    # # vector_store.add_documents(create_sample_docs())
    # results = vector_store.similarity_search_with_relevance_scores("hello world", k=3)
    # for res, score in results:
    #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    # ----------------------------------------
    # K2DOCS VectorStore
    # ----------------------------------------
    # vector_store = init_k2_doc_vectorstore(drop_first=True)
    results = k2_doc_vectorstore_similarity_search("审批按钮解释？", k=5)
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} ")
