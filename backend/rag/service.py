import hashlib
import os
import tempfile
from threading import Lock

from backend.llm.chat import generate_rag_answer
from backend.llm.model import load_llm
from backend.rag.database import get_vector_db, retrieve_context
from backend.rag.ingestion import processar_documento


_MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "device": None,
}
_MODEL_LOCK = Lock()


def _slug_from_filename(filename):
    base_name = os.path.splitext(filename)[0].strip().lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in base_name)
    return "-".join(part for part in safe.split("-") if part) or "documento"


def _document_fingerprint(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()[:12]


def _get_cached_llm():
    with _MODEL_LOCK:
        if _MODEL_CACHE["model"] is None or _MODEL_CACHE["tokenizer"] is None:
            model, tokenizer, device = load_llm()
            if not model or not tokenizer:
                raise RuntimeError("Falha ao carregar o modelo LLM.")

            _MODEL_CACHE["model"] = model
            _MODEL_CACHE["tokenizer"] = tokenizer
            _MODEL_CACHE["device"] = device

        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"], _MODEL_CACHE["device"]


def index_uploaded_pdf(file_name, file_bytes, db_root="./db/chroma_db"):
    """Indexa um PDF enviado por upload e retorna metadados da base criada."""
    slug = _slug_from_filename(file_name)
    fingerprint = _document_fingerprint(file_bytes)
    collection_name = f"{slug}-{fingerprint}"
    persist_directory = os.path.join(db_root, collection_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_bytes)
        temp_path = temp_pdf.name

    try:
        result = processar_documento(
            temp_path,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        return {
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "chunks_count": len(result["chunks"]),
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def answer_question(question, persist_directory, collection_name, top_k=5):
    """Executa recuperacao + geracao para responder uma pergunta."""
    vector_db = get_vector_db(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    if not vector_db:
        raise RuntimeError("Banco vetorial nao encontrado para este documento.")

    docs, context = retrieve_context(vector_db, query=question, k=top_k)
    model, tokenizer, device = _get_cached_llm()

    answer = generate_rag_answer(
        model=model,
        tokenizer=tokenizer,
        question=question,
        context=context,
        device=device,
    )
    return {
        "answer": answer,
        "docs": docs,
        "context": context,
    }
