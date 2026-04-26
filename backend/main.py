import secrets
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.rag.service import answer_question, index_uploaded_pdf


app = FastAPI(title="Trabalho1-ODS2 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8)
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=15)


class ChatResponse(BaseModel):
    answer: str
    context: str


_SESSIONS: Dict[str, Dict[str, str]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nome do arquivo invalido")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF sao aceitos")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Arquivo PDF vazio")

    try:
        metadata = index_uploaded_pdf(file.filename, file_bytes)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Falha ao indexar documento: {error}")

    session_id = secrets.token_urlsafe(16)
    _SESSIONS[session_id] = {
        "persist_directory": metadata["persist_directory"],
        "collection_name": metadata["collection_name"],
        "doc_name": file.filename,
        "chunks_count": str(metadata["chunks_count"]),
    }

    return {
        "session_id": session_id,
        "doc_name": file.filename,
        "chunks_count": metadata["chunks_count"],
        "collection_name": metadata["collection_name"],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session = _SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sessao nao encontrada. Reenvie o PDF.")

    try:
        result = answer_question(
            question=request.question,
            persist_directory=session["persist_directory"],
            collection_name=session["collection_name"],
            top_k=request.top_k,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Falha ao gerar resposta: {error}")

    return {
        "answer": result["answer"],
        "context": result["context"],
    }
