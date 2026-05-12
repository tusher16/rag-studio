from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


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
    llm = ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="deepseek/deepseek-r1-0528-qwen3-8b",
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

    print(f"RAG chain ready — LLM: deepseek/deepseek-r1-0528-qwen3-8b")
    return chain

def ask(question: str, chain) -> str:
    print(f"\nQuestion: {question}")
    print("Generating answer...\n")

    answer = chain.invoke(question)

    print(f"Answer:\n{answer}")
    return answer

if __name__ == "__main__":
    from src.retrieval import load_vectorstore, build_retriever

    vectorstore = load_vectorstore()
    retriever   = build_retriever(vectorstore)
    chain       = build_rag_chain(retriever)

    ask("What is this paper about?", chain)