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