from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from fastapi import UploadFile, File
import shutil
import sys
sys.path.insert(0, ".")
from fastapi import BackgroundTasks


load_dotenv()

app = FastAPI(title="RAG Studio")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

DATA_DIR = Path("data/docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def index():
    return FileResponse("frontend/static/index.html")


INGEST_STATUS = {"state": "idle", "filename": None, "message": ""}


# Add near the top, after INGEST_STATUS
RAG_CHAIN = None

@app.on_event("startup")
async def preload_models():
    global RAG_CHAIN
    try:
        from rag_v1.retrieval import load_vectorstore, build_retriever
        from rag_v1.generation import build_rag_chain
        vs = load_vectorstore()
        retriever = build_retriever(vs)
        RAG_CHAIN = build_rag_chain(retriever)
        RAG_CHAIN.invoke("warmup")
        print("✓ Models preloaded on startup")
    except Exception as e:
        print(f"Preload failed: {e}")

@app.post("/api/ingest")
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    INGEST_STATUS["state"] = "processing"
    INGEST_STATUS["filename"] = file.filename
    INGEST_STATUS["message"] = "Loading PDF..."
    
    def _run():
        try:
            from rag_v1.ingestion import run_ingestion
            run_ingestion()
            INGEST_STATUS["state"] = "complete"
            INGEST_STATUS["message"] = "Done"
        except Exception as e:
            INGEST_STATUS["state"] = "failed"
            INGEST_STATUS["message"] = str(e)
    
    background_tasks.add_task(_run)
    return {"status": "processing", "filename": file.filename}

@app.get("/api/ingest/status")
async def ingest_status():
    return INGEST_STATUS


from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

@app.post("/api/chat")
async def chat_api(req: ChatRequest):
    global RAG_CHAIN
    if RAG_CHAIN is None:
        from rag_v1.retrieval import load_vectorstore, build_retriever
        from rag_v1.generation import build_rag_chain
        vs = load_vectorstore()
        retriever = build_retriever(vs)
        RAG_CHAIN = build_rag_chain(retriever)
    
    answer = RAG_CHAIN.invoke(req.question)
    return {"answer": answer}

@app.get("/api/stats")
async def stats():
    try:
        from rag_v1.retrieval import load_vectorstore
        vectorstore = load_vectorstore()
        vector_count = vectorstore._collection.count()
        
        # count files in data/docs
        pdf_files = list(DATA_DIR.glob("**/*.pdf"))
        
        return {
            "documents": len(pdf_files),
            "chunks": vector_count,
            "vectors": vector_count,
            "last_run": INGEST_STATUS.get("filename", "—"),
        }
    except Exception as e:
        return {"documents": 0, "chunks": 0, "vectors": 0, "last_run": "—"}
    


class RetrieveRequest(BaseModel):
    query: str

@app.post("/api/retrieve")
async def retrieve_api(req: RetrieveRequest):
    from rag_v1.retrieval import load_vectorstore, build_retriever
    
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    docs = retriever.invoke(req.query)
    
    results = []
    for i, doc in enumerate(docs, 1):
        results.append({
            "rank": i,
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": int(doc.metadata.get("page", 0)),
            "score": float(doc.metadata.get("relevance_score", 0.0)),
        })
    return {"results": results, "count": len(results)}


import json
EVAL_RESULTS_FILE = Path("data/eval_results.json")

EVAL_STATUS = {
    "state": "idle",
    "progress": 0,
    "total": 0,
    "current_question": "",
    "results": None,
}

# Load previous results on startup
if EVAL_RESULTS_FILE.exists():
    try:
        EVAL_STATUS["results"] = json.loads(EVAL_RESULTS_FILE.read_text())
        EVAL_STATUS["state"] = "complete"
    except Exception as e:
        print(f"Could not load eval results: {e}")


@app.post("/api/evaluate")
async def evaluate_api(background_tasks: BackgroundTasks):
    if EVAL_STATUS["state"] == "running":
        return {"status": "already_running"}
    
    EVAL_STATUS["state"] = "running"
    EVAL_STATUS["progress"] = 0
    
    def _run():
        try:
            from rag_v1.test_dataset import TEST_CASES
            from rag_v1.retrieval import load_vectorstore, build_retriever
            from datetime import datetime
            
            vs = load_vectorstore()
            retriever = build_retriever(vs)
            
            EVAL_STATUS["total"] = len(TEST_CASES)
            per_question = []
            
            for i, test in enumerate(TEST_CASES):
                EVAL_STATUS["progress"] = i
                EVAL_STATUS["current_question"] = test["question"][:60]
                
                answer = RAG_CHAIN.invoke(test["question"])
                
                gt = test["ground_truth"].lower()
                ans = answer.lower()
                gt_words = set(gt.split())
                ans_words = set(ans.split())
                overlap = len(gt_words & ans_words) / max(len(gt_words), 1)
                
                per_question.append({
                    "id": i + 1,
                    "q": test["question"],
                    "a": answer[:200],
                    "ground_truth": test["ground_truth"][:200],
                    "score": round(overlap, 3),
                    "status": "pass" if overlap > 0.5 else "warn" if overlap > 0.3 else "fail",
                })
            
            EVAL_STATUS["progress"] = len(TEST_CASES)
            results = {
                "per_question": per_question,
                "avg_score": round(sum(q["score"] for q in per_question) / len(per_question), 3),
                "pass_count": sum(1 for q in per_question if q["status"] == "pass"),
                "warn_count": sum(1 for q in per_question if q["status"] == "warn"),
                "fail_count": sum(1 for q in per_question if q["status"] == "fail"),
                "ran_at": datetime.utcnow().isoformat(),
                "model": "qwen2.5:3b",
                "total_questions": len(TEST_CASES),
            }
            EVAL_STATUS["results"] = results
            EVAL_RESULTS_FILE.write_text(json.dumps(results, indent=2))
            EVAL_STATUS["state"] = "complete"
        except Exception as e:
            EVAL_STATUS["state"] = "failed"
            EVAL_STATUS["current_question"] = str(e)
    
    background_tasks.add_task(_run)
    return {"status": "started"}


@app.get("/api/evaluate/status")
async def evaluate_status():
    return EVAL_STATUS