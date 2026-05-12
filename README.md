# RAG Studio

A web-based RAG (Retrieval Augmented Generation) playground built from scratch.
Live at: [rag.tusher16.com](https://rag.tusher16.com)

---

## Features

| Tab | Description |
|---|---|
| **Pipeline** | Visual ingestion flow — upload PDF/CSV → chunk → embed → store |
| **Chat** | Ask questions against ingested documents |
| **Retrieval View** | See chunks, MMR scores, and rerank scores live |
| **Evaluation** | Run RAGAS evaluation and view scores dashboard |
| **Docs** | Architecture decisions and version history |

---

## Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | HTML + Tailwind CSS + Vanilla JS |
| Embeddings | BAAI/bge-m3 |
| Vector DB | ChromaDB (persistent) |
| Reranker | FlashrankRerank |
| LLM | Qwen2.5:3b via Ollama |
| Evaluation | RAGAS |
| Infra | Docker + nginx-proxy + Let's Encrypt |
| CI/CD | GitHub Actions → home server |

---

## RAG Versions

```
rag-studio/
├── backend/
│   ├── rag_v1/     ← BGE-M3 + ChromaDB + Qwen2.5:3b (current)
│   └── rag_v2/     ← coming soon
└── frontend/
```

---

## Local Development

```bash
git clone https://github.com/tusher16/rag-studio.git
cd rag-studio

cp .env.example .env
# fill in your values

docker compose up --build
```

App runs at `http://localhost:8000`

---

## Deployment

Auto-deploys to `rag.tusher16.com` on every push to `main` via GitHub Actions.