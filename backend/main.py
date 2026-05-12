from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
from fastapi import UploadFile, File
import shutil
import sys
sys.path.insert(0, ".")


load_dotenv()

app = FastAPI(title="RAG Studio")

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

DATA_DIR = Path("data/docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def index():
    return FileResponse("frontend/static/index.html")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    from rag_v1.ingestion import run_ingestion
    vectorstore = run_ingestion()
    return {"status": "success", "filename": file.filename}