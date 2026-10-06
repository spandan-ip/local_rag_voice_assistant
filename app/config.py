import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

TOP_K = int(os.getenv("TOP_K", 6))

# K values evaluate.py reports metrics for.
EVAL_K_VALUES = [int(x) for x in os.getenv("EVAL_K_VALUES", "1,3,5,10").split(",")]

# --- Hybrid retrieval (BM25 + RRF) ---
# RRF_K: the constant in the RRF formula (1 / (RRF_K + rank)). Higher values
#   flatten the difference between top-ranked and lower-ranked results.
RRF_K = int(os.getenv("RRF_K", 60))
# HYBRID_CANDIDATE_K: how many candidates EACH method (semantic, BM25)
#   retrieves before they get fused and cut down to the final TOP_K.
HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", 20))

# --- API server ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.getenv("SOURCE_DIR", os.path.join(BASE_DIR, "source"))
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(BASE_DIR, "vector_store"))

# --- Chat history (SQLite) ---
CHAT_HISTORY_DB_PATH = os.getenv("CHAT_HISTORY_DB_PATH", os.path.join(BASE_DIR, "chat_history.db"))
# How many past Q&A turns to feed back to the LLM so it can understand follow-up questions.
CONVERSATION_MEMORY_TURNS = int(os.getenv("CONVERSATION_MEMORY_TURNS", 3))

# --- Voice: Speech-to-Text (faster-whisper) ---
# "tiny" and "base" are the lightest models -- good choice for CPU-only,
# 8GB-RAM machines. Larger models ("small", "medium") are more accurate
# but slower and heavier.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Voice: Text-to-Speech (pyttsx3) ---
TTS_RATE = int(os.getenv("TTS_RATE", 175))
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", os.path.join(BASE_DIR, "audio_output"))

# Edit this list to control what gets ingested. Each entry is loaded by
# app/ingestion/ingestion_pipeline.py via load_document(source_type, source).
SOURCES = [
    {"source_type": "pdf", "source": os.path.join(SOURCE_DIR, "sample.pdf")},
    # {"source_type": "txt", "source": os.path.join(SOURCE_DIR, "notes.txt")},
    # {"source_type": "csv", "source": os.path.join(SOURCE_DIR, "data.csv")},
    # {"source_type": "web", "source": "https://example.com"},
]
