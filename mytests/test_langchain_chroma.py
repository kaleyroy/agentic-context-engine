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
                "content": item["content"],
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


def build_k2qa_doc_samples():
    file_path = ROOT / "datasets" / "k2docs.json"
    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    logger.info(f"Loaded {len(items)} items from {Path(file_path).name}")
    samples = []
    logger.info(f"Building samples for {len(items)} items")
    for item in items:
        content = item["content"]
        title = item["title"]
        pos = content.rfind(f"{title}？") or content.rfind(f"{title}?")
        final_asnwer = content[pos + len(title) + 1 :]
        question_str = content[: pos + len(title) + 1]
        questions = (
            question_str.split("？")
            if "？" in question_str
            else question_str.split("?")
        )
        for question in questions:
            if len(question.strip()) == 0 or len(final_asnwer.strip()) == 0:
                continue

            samples.append(
                {
                    "question": question.strip() + "？",
                    "answer": final_asnwer.strip(),
                    "docid": item["id"],
                }
            )

    # Write to k2qa_samples.json
    logger.info(f"Writing {len(samples)} samples to k2qa_samples.json")
    with open(ROOT / "datasets" / "k2qa_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    return samples


def random_k2qa_samples(count: int = 10):
    """Get random k2qa samples from k2qa_samples.json with unique docid"""
    import random

    with open(ROOT / "datasets" / "k2qa_samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    sample_results = {}
    logger.info(f"Loaded {len(samples)} samples from k2qa_samples.json")
    if count > len(samples):
        count = len(samples)
    while True:
        if len(sample_results) >= count:
            break
        random_samples = random.sample(samples, count)
        # deduplicates by docid
        for sample in random_samples:
            if len(sample_results) >= count:
                break
            if sample["docid"] not in sample_results:
                sample_results[sample["docid"]] = sample

    # Write to k2qa_rand_samples.json
    logger.info(f"Writing {len(sample_results)} samples to k2qa_rand_samples.json")
    with open(ROOT / "datasets" / "k2qa_rand_samples.json", "w", encoding="utf-8") as f:
        json.dump(list(sample_results.values()), f, ensure_ascii=False, indent=2)


def build_k2qa_ctxt_samples(k: int = 3):
    """Build k2qa context samples from k2qa_rand_samples.json"""
    with open(ROOT / "datasets" / "k2qa_rand_samples.json", "r", encoding="utf-8") as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples from k2qa_rand_samples.json")
    # Build context samples
    context_samples = []
    for sample in samples:
        question = sample["question"]
        logger.info(f"Getting context for `{question}` in vectorstore")
        results = k2_doc_vectorstore_similarity_search(question, k=k)
        logger.info(f"Fetched {len(results)} results for `{question}`")
        context_samples.append(
            {
                "docid": sample["docid"],
                "question": question,
                "answer": sample["answer"],
                "context": [res.page_content for res, _ in results],
                "context_ids": [res.metadata["id"] for res, _ in results],
            }
        )
    # Write to k2qa_ctxt_samples.json
    logger.info(f"Writing {len(context_samples)} samples to k2qa_ctxt_samples.json")
    with open(ROOT / "datasets" / "k2qa_ctxt_samples.json", "w", encoding="utf-8") as f:
        json.dump(context_samples, f, ensure_ascii=False, indent=2)


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
    # results = k2_doc_vectorstore_similarity_search("审批按钮解释？", k=5)
    # for res, score in results:
    #     print(f"* [SIM={score:3f}] {res.page_content} ")

    # ----------------------------------------
    # K2QA Samples
    # ----------------------------------------
    # build_k2qa_doc_samples()
    # random_k2qa_samples(count=25)
    build_k2qa_ctxt_samples(k=3)
