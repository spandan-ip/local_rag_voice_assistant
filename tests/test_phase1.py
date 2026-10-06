"""
test_phase1.py
---------------
Purpose:
    Automated tests for the Phase 1 pipeline. These are the tests
    you'd show in a GitHub Actions badge / README screenshot.

    We deliberately avoid depending on a running Ollama server or
    internet access for most tests, so `pytest` works out of the box
    in CI. The one test that needs the real embedding model
    (test_chunking_and_embedding_pipeline) will download bge-small on
    first run -- mark/skip it if you want a fully offline CI run.

Run:
    pytest -v
"""

import os
import tempfile

import pytest
from langchain_core.documents import Document

from app.ingestion.loaders import load_document, DocumentLoadError
from app.chunking.splitter import chunk_documents
from app.vectorstore.faiss_store import similarity_search, VectorStoreError


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

def test_load_missing_pdf_raises_clear_error():
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document("pdf", "source/does_not_exist.pdf")


def test_load_empty_pdf_raises_clear_error():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name  # zero-byte file
    try:
        with pytest.raises(DocumentLoadError, match="empty"):
            load_document("pdf", path)
    finally:
        os.remove(path)


def test_unsupported_source_type_raises_clear_error():
    with pytest.raises(DocumentLoadError, match="Unsupported source_type"):
        load_document("docx", "source/whatever.docx")


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

def test_chunking_respects_chunk_size_and_overlap():
    long_text = "sentence number %d. " % 0
    long_text = " ".join(f"This is sentence number {i}." for i in range(200))
    doc = Document(page_content=long_text, metadata={"source": "test.txt", "source_type": "txt"})

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1, "Long text should be split into multiple chunks"
    for chunk in chunks:
        # RecursiveCharacterTextSplitter can slightly exceed chunk_size
        # when it can't find a good split point; allow small slack.
        assert len(chunk.page_content) <= 150


def test_chunking_preserves_metadata():
    doc = Document(
        page_content="A" * 500,
        metadata={"source": "policy.pdf", "source_type": "pdf", "page": 3},
    )

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "policy.pdf"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["page"] == 3


def test_chunking_empty_list_returns_empty_list():
    assert chunk_documents([]) == []


# ---------------------------------------------------------------------------
# Vector store tests
# ---------------------------------------------------------------------------

def test_similarity_search_rejects_empty_query():
    class DummyStore:
        def similarity_search(self, query, k):
            return []

    with pytest.raises(VectorStoreError, match="empty query"):
        similarity_search(DummyStore(), "   ", k=3)


def test_similarity_search_returns_results_from_store():
    expected_docs = [Document(page_content="result", metadata={"source": "a.pdf"})]

    class DummyStore:
        def similarity_search(self, query, k):
            assert query == "hello"
            assert k == 3
            return expected_docs

    results = similarity_search(DummyStore(), "hello", k=3)
    assert results == expected_docs


# ---------------------------------------------------------------------------
# End-to-end (slow) test: real embedding model + real FAISS index.
# Skipped automatically if the model can't be downloaded (e.g. offline CI).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_chunking_and_embedding_pipeline_end_to_end(tmp_path):
    from app.embeddings.embedding_model import get_embedding_model
    from app.vectorstore.faiss_store import build_or_load_vectorstore

    docs = [
        Document(
            page_content="The company offers 20 days of annual leave per year.",
            metadata={"source": "policy.pdf", "source_type": "pdf", "page": 1},
        ),
        Document(
            page_content="Employees must submit expense reports within 30 days.",
            metadata={"source": "policy.pdf", "source_type": "pdf", "page": 2},
        ),
    ]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

    embedding_model = get_embedding_model()
    index_path = str(tmp_path / "vector_store")
    vectorstore = build_or_load_vectorstore(chunks, embedding_model, index_path)

    results = similarity_search(vectorstore, "How many vacation days do I get?", k=1)

    assert len(results) == 1
    assert "leave" in results[0].page_content.lower()
