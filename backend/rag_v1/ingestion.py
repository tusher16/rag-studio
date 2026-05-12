import hashlib
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_pdfs(data_dir: Path) -> list:
    pdf_files = sorted(data_dir.glob("**/*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    documents = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata.update({
                "source"      : pdf_path.name,
                "file_hash"   : hashlib.md5(pdf_path.read_bytes()).hexdigest(),
                "ingested_at" : datetime.now(timezone.utc).isoformat(),
                "category"    : "documentation",
            })

        documents.extend(pages)
        print(f"Loaded: {pdf_path.name} ({len(pages)} pages)")

    return documents

def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        length_function = len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Chuncked: {len(documents)} pages -> {len(chunks)} chunks")

    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name = config.EMBED_MODEL,
        model_kwargs = {"device":"mps"},
        encode_kwargs = {"normalize_embeddings": True}
    )


def build_vectorstore(chunks: list) -> Chroma:
    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )
    print(f"Stored {len(chunks)} vectors to {config.VECTORSTORE_DIR}")
    return vectorstore


def run_ingestion():
    print("Phase1: Ingestion")
    documents = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    print("Ingestion complete!")
    return vectorstore

if __name__=="__main__":
    run_ingestion()
