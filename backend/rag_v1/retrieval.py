from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
    )

    count = vectorstore._collection.count()
    print(f"Vectorstore loaded — {count} vectors")
    return vectorstore

def build_retriever(vectorstore: Chroma):
    base_retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs ={
            "k": config.TOP_K,
            "fetch_k": config.TOP_K * 2,
            "lambda_mult" : 0.7,
        },
    )

    reranker = FlashrankRerank(top_n=config.RERANK_TOP_N, score_threshold=0.1)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    print(f"Retriever ready — MMR top-{config.TOP_K} → Rerank top-{config.RERANK_TOP_N}")
    return retriever


def retrieve(query: str, retriever) -> list:
    print(f"\nQuery: {query}")
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} chunks after reranking")
    for i, doc in enumerate(docs, 1):
        print(f"  [{i}] {doc.metadata.get('source')} | {doc.page_content[:80]}...")
    return docs


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    retrieve("What is this paper about?", retriever)