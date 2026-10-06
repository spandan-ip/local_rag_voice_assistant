# local-rag-voice-assistant

A local, private RAG (Retrieval-Augmented Generation) assistant: ask
questions about your own documents (PDF/TXT/CSV/websites), by text or
voice, and get grounded, sourced answers -- spoken back to you when you
asked by voice. Everything runs locally: Ollama for the LLM, a local
embedding model, FAISS + BM25 for retrieval, faster-whisper for speech
recognition, and your OS's built-in voice for speech output.

```
PDF / TXT / CSV / Web  →  Chunking  →  Embeddings  →  FAISS + BM25 (RRF)
                                                            │
                                                        Top-K chunks
                                                            │
                                                     Llama 3.2 (Ollama)
                                                        /        \
                                                  Text answer   Spoken answer
```

## Project structure

```
local-rag-voice-assistant/
├── app/
│   ├── ingestion/       PDF/TXT/CSV/Web loaders, document IDs, dynamic sources
│   ├── chunking/        Text splitting
│   ├── embeddings/      Local embedding model
│   ├── vectorstore/     FAISS
│   ├── retrieval/       Semantic search, BM25, RRF, hybrid search
│   ├── generation/      Prompt template, Ollama LLM (incl. streaming)
│   ├── evaluation/      Precision@K / Recall@K / Hit Rate@K
│   ├── history/         SQLite chat sessions
│   ├── voice/           Speech-to-text (faster-whisper), text-to-speech (pyttsx3)
│   ├── api/              Small shared helpers used by api.py
│   ├── pipeline.py       Builds the index (used by main.py, api.py, evaluate.py)
│   └── config.py         Every tunable setting, in one place
├── frontend/              Plain HTML/CSS/JS chat UI
├── main.py                CLI chat (no server needed)
├── api.py                 FastAPI backend (used by the frontend)
├── evaluate.py             Retrieval evaluation report
├── tests/                  pytest suite (one file per phase of development)
├── source/                 Put your documents here
├── requirements.txt
└── .env.example
```

## Setup

1. `python -m venv venv` then activate it (`venv\Scripts\activate` on Windows).
2. `pip install -r requirements.txt`
3. Install [Ollama](https://ollama.com), then `ollama pull llama3.2`.
4. Copy `.env.example` to `.env` if you want to change any defaults (optional).
5. Put a PDF (or TXT/CSV) in `source/`, and list it in `SOURCES` in `app/config.py`.

## Running it

**Option A -- just the CLI, no server:**
```
python main.py --rebuild
```

**Option B -- full app with the web frontend:**
```
python api.py
```
Wait for "Index ready.", then open **http://127.0.0.1:8000** in your
browser. That's it -- one command, one terminal. `api.py` serves both
the API and the frontend (`frontend/`) from the same address, so there's
no separate frontend server to start.

**Evaluate retrieval quality:**
```
python evaluate.py
```
(Edit `app/evaluation/eval_dataset.py` first with real questions about
your actual documents.)

**Run the tests:**
```
pytest -v -m "not slow"
```

## What the frontend supports

- Text or voice questions (voice → transcribed → shown on screen → answered → spoken back, only for voice questions)
- Streaming answers with a working Stop button
- A live "Answering... Xs" timer
- Chat history that survives a page reload (backed by SQLite, not browser memory)
- A sidebar to switch between past conversations, or start a new chat
- Uploading a new PDF/TXT/CSV, or adding a website URL, without restarting anything
- Source attribution on every answer
- A breathing orb that changes pace/glow depending on state (idle, listening, thinking)

## Known limitations

- Document/URL ingestion from the frontend rebuilds the *entire* index
  synchronously -- the app is unresponsive during that rebuild. A
  production version would do this as a background job.
- Small local models (like Llama 3.2 3B) don't always perfectly follow
  "don't use outside knowledge" instructions -- worth testing on your
  own documents (see the prompt in `app/generation/prompts.py`).
- `evaluate.py`'s dataset (`app/evaluation/eval_dataset.py`) has to be
  written by hand to match your real documents -- it's not generated
  automatically, since the "correct answer key" has to come from a
  human, not the system being evaluated.