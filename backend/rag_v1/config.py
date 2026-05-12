from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

TOP_K = 20
RERANK_TOP_N = 10

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL   = "qwen2.5:3b"

COLLECTION_NAME = "production_rag"