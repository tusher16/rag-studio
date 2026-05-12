from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import shutil
import os
from pathlib import Path

load_dotenv()

app = FastAPI(title="RAG Studio")

templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

DATA_DIR = Path("data/docs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def pipeline(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/retrieval", response_class=HTMLResponse)
async def retrieval(request: Request):
    return templates.TemplateResponse("retrieval.html", {"request": request})

@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation(request: Request):
    return templates.TemplateResponse("evaluation.html", {"request": request})

@app.get("/docs-rag", response_class=HTMLResponse)
async def docs(request: Request):
    return templates.TemplateResponse("docs.html", {"request": request})