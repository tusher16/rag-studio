from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from rag_v1 import config


def format_context(docs: list) -> str:
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        sections.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(sections)


SYSTEM_PROMPT = """You are a precise, grounded assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say: "I don't have enough information."
Never use your own knowledge. Always cite the source.

Context:
{context}"""


def build_rag_chain(retriever):
    llm = ChatOllama(
        base_url="http://ollama:11434",
        model="qwen2.5:3b",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context" : retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG chain ready — LLM: qwen2.5:3b via Ollama")
    return chain


def ask(question: str, chain) -> str:
    print(f"\nQuestion: {question}")
    print("Generating answer...\n")
    answer = chain.invoke(question)
    print(f"Answer:\n{answer}")
    return answer


if __name__ == "__main__":
    from rag_v1.retrieval import load_vectorstore, build_retriever

    vectorstore = load_vectorstore()
    retriever   = build_retriever(vectorstore)
    chain       = build_rag_chain(retriever)

    ask("What is this paper about?", chain)