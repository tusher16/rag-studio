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
    from rag_v1.retrieval import load_vectorstore, build_retriever
    from rag_v1.generation import build_rag_chain
    
    vectorstore = load_vectorstore()
    retriever = build_retriever(vectorstore)
    chain = build_rag_chain(retriever)
    
    answer = chain.invoke(req.question)
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
            "page": doc.metadata.get("page", "?"),
            "score": doc.metadata.get("relevance_score", 0.0),
        })
    return {"results": results, "count": len(results)}