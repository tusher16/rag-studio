import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from ragas.run_config import RunConfig

import config


def get_evaluator_llm():
    return LangchainLLMWrapper(ChatOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        model="qwen2.5-3b-instruct-mlx",
        temperature=0,
        n=1,
    ))


def get_evaluator_embeddings():
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))


from eval.test_dataset import TEST_CASES

def build_test_dataset():
    return TEST_CASES


def run_evaluation():
    from src.retrieval import load_vectorstore, build_retriever
    from src.generation import build_rag_chain, format_context

    print("RAG Evaluation\n")

    # load pipeline
    vectorstore = load_vectorstore()
    retriever   = build_retriever(vectorstore)
    chain       = build_rag_chain(retriever)

    test_cases = build_test_dataset()

    questions    = []
    answers      = []
    contexts     = []
    ground_truths = []

    for test in test_cases:
        question = test["question"]
        print(f"Running: {question}")

        # get answer from RAG
        answer = chain.invoke(question)

        # get retrieved chunks
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(test["ground_truth"])

    return questions, answers, contexts, ground_truths

def score(questions, answers, contexts, ground_truths):
    dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    })

    llm        = get_evaluator_llm()
    embeddings = get_evaluator_embeddings()

    result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False,
    run_config=RunConfig(timeout=120, max_workers=1),
    )

    print("\n=== Evaluation Results ===")
    print(result)
    return result


if __name__ == "__main__":
    questions, answers, contexts, ground_truths = run_evaluation()
    score(questions, answers, contexts, ground_truths)