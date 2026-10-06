import time
import os
import tempfile
import logging

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import TOP_K, API_HOST, API_PORT, CONVERSATION_MEMORY_TURNS, SOURCE_DIR
from app.pipeline import build_index
from app.retrieval.hybrid_search import hybrid_search
from app.generation.llm import generate_answer, stream_answer, LLMError
from app.generation.query_rewriter import rewrite_query
from app.generation.relevance_filter import filter_relevant_chunks
from app.ingestion.loaders import DocumentLoadError
from app.ingestion.source_registry import add_source, get_all_sources
from app.embeddings.embedding_model import EmbeddingModelError
from app.vectorstore.faiss_store import VectorStoreError
from app.retrieval.bm25_search import BM25Error
from app.api.formatting import build_sources
from app.history.db import init_db
from app.history.session_store import (
    create_session,
    save_message,
    get_recent_context,
    get_session_messages,
    list_sessions,
    rename_session,
)
from app.voice.speech_to_text import transcribe_audio, SpeechToTextError
from app.voice.text_to_speech import synthesize_speech, TextToSpeechError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="RAG Voice Assistant API")

# Allows the static HTML/JS frontend (served from a different origin, e.g.
# a local file server on another port) to call this API from the browser.
# Wide open here for local development; a real deployment would restrict
# allow_origins to the frontend's actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Holds the built index so we don't rebuild it on every request.
# Populated once, in startup(), when the server starts.
_state: dict = {}


@app.on_event("startup")
def startup():
    init_db()
    logger.info("Building index...")
    vectorstore, bm25, bm25_documents = build_index()
    _state["vectorstore"] = vectorstore
    _state["bm25"] = bm25
    _state["bm25_documents"] = bm25_documents
    logger.info("Index ready.")


class AskRequest(BaseModel):
    question: str
    k: int = TOP_K
    session_id: str | None = None  # omit to start a new chat session


def _retrieve(question: str, k: int):
    search_query = rewrite_query(question)
    return hybrid_search(_state["vectorstore"], _state["bm25"], _state["bm25_documents"], search_query, k)


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": "vectorstore" in _state}


# --- Session endpoints ---

@app.post("/sessions")
def new_session():
    return {"session_id": create_session()}


@app.get("/sessions")
def get_sessions():
    return {"sessions": list_sessions()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    return {"session_id": session_id, "messages": get_session_messages(session_id)}


class RenameSessionRequest(BaseModel):
    title: str


@app.patch("/sessions/{session_id}")
def patch_session(session_id: str, payload: RenameSessionRequest):
    rename_session(session_id, payload.title)
    return {"session_id": session_id, "title": payload.title.strip()[:60]}


# --- Sources: what documents/URLs the assistant is currently drawing from ---

@app.get("/sources")
def get_sources():
    return {"sources": get_all_sources()}


# --- Ask endpoints ---

@app.post("/ask")
def ask(payload: AskRequest):
    start_time = time.time()
    session_id = payload.session_id or create_session()

    try:
        retrieved = _retrieve(payload.question, payload.k)
    except (VectorStoreError, BM25Error) as exc:
        return {"error": str(exc)}

    save_message(session_id, "user", payload.question)

    if not retrieved:
        answer = "I don't know based on the provided sources."
        save_message(session_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": [],
            "session_id": session_id,
            "time_taken_seconds": round(time.time() - start_time, 2),
        }

    history = get_recent_context(session_id, max_turns=CONVERSATION_MEMORY_TURNS)

    try:
        answer = generate_answer(payload.question, retrieved, history=history)
    except LLMError as exc:
        return {"error": str(exc)}

    relevant_chunks = filter_relevant_chunks(retrieved, answer)
    sources = build_sources(relevant_chunks)
    save_message(session_id, "assistant", answer, sources=sources)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id,
        "time_taken_seconds": round(time.time() - start_time, 2),
    }


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    session_id = payload.session_id or create_session()

    try:
        retrieved = _retrieve(payload.question, payload.k)
    except (VectorStoreError, BM25Error) as exc:
        return {"error": str(exc)}

    save_message(session_id, "user", payload.question)

    if not retrieved:
        async def empty_generator():
            answer = "I don't know based on the provided sources."
            save_message(session_id, "assistant", answer)
            yield answer
        return StreamingResponse(
            empty_generator(), media_type="text/plain", headers={"X-Session-Id": session_id}
        )

    history = get_recent_context(session_id, max_turns=CONVERSATION_MEMORY_TURNS)

    async def token_generator():
        start_time = time.time()
        collected: list[str] = []
        try:
            for token in stream_answer(payload.question, retrieved, history=history):
                # If the client closed the connection (e.g. the user hit
                # "Stop"), stop generating instead of wasting LLM compute
                # on tokens nobody will see.
                if await request.is_disconnected():
                    logger.info("Client disconnected; stopping generation early.")
                    break
                collected.append(token)
                yield token
        except LLMError as exc:
            yield f"\n[Error: {exc}]"
        finally:
            full_answer = "".join(collected)
            if full_answer:
                relevant_chunks = filter_relevant_chunks(retrieved, full_answer)
                sources = build_sources(relevant_chunks)
                save_message(session_id, "assistant", full_answer, sources=sources)
            logger.info("Stream finished in %.2fs", time.time() - start_time)

    return StreamingResponse(
        token_generator(), media_type="text/plain", headers={"X-Session-Id": session_id}
    )


# --- Voice endpoints ---

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        text = transcribe_audio(tmp_path)
    except SpeechToTextError as exc:
        return {"error": str(exc)}
    finally:
        os.remove(tmp_path)

    return {"text": text}


class SpeakRequest(BaseModel):
    text: str


@app.post("/speak")
def speak(payload: SpeakRequest):
    try:
        audio_path = synthesize_speech(payload.text)
    except TextToSpeechError as exc:
        return {"error": str(exc)}

    return FileResponse(audio_path, media_type="audio/wav", filename="answer.wav")


# --- Ingestion endpoints (add documents/URLs at runtime, e.g. from the frontend) ---

def _rebuild_and_replace_state():
    vectorstore, bm25, bm25_documents = build_index(force_rebuild=True)
    _state["vectorstore"] = vectorstore
    _state["bm25"] = bm25
    _state["bm25_documents"] = bm25_documents


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    ext = os.path.splitext(filename)[1].lower()
    type_map = {".pdf": "pdf", ".txt": "txt", ".csv": "csv"}
    source_type = type_map.get(ext)

    if not source_type:
        return {"error": f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .csv"}

    os.makedirs(SOURCE_DIR, exist_ok=True)
    dest_path = os.path.join(SOURCE_DIR, filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    add_source(source_type, dest_path)

    try:
        _rebuild_and_replace_state()
    except (DocumentLoadError, EmbeddingModelError, VectorStoreError, BM25Error) as exc:
        return {"error": f"File saved but failed to rebuild index: {exc}"}

    return {"message": f"'{filename}' ingested and index rebuilt."}


class IngestWebRequest(BaseModel):
    url: str


@app.post("/ingest/web")
def ingest_web(payload: IngestWebRequest):
    add_source("web", payload.url)

    try:
        _rebuild_and_replace_state()
    except (DocumentLoadError, EmbeddingModelError, VectorStoreError, BM25Error) as exc:
        return {"error": f"Failed to ingest URL: {exc}"}

    return {"message": f"'{payload.url}' ingested and index rebuilt."}


# --- Serve the frontend from this same server/port ---
# Mounted LAST and at "/" so every specific route above (e.g. /ask,
# /health) is matched first; anything else falls through to these
# static files, so `python api.py` alone serves the whole app.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
