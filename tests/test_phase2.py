import os
import tempfile

import pytest

from app.ingestion.loaders import load_document, DocumentLoadError
from app.ingestion.metadata import generate_document_id, add_document_id
from app.ingestion.ingestion_pipeline import load_all_sources


# ---------------------------------------------------------------------------
# TXT loader
# ---------------------------------------------------------------------------

def test_load_txt_missing_file():
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document("txt", "source/does_not_exist.txt")


def test_load_txt_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(DocumentLoadError, match="empty"):
            load_document("txt", path)
    finally:
        os.remove(path)


def test_load_txt_success():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world, this is a test file.")
        path = f.name
    try:
        docs = load_document("txt", path)
        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["source_type"] == "txt"
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def test_load_csv_missing_file():
    with pytest.raises(DocumentLoadError, match="not found"):
        load_document("csv", "source/does_not_exist.csv")


def test_load_csv_success():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("name,leave_days\nAlice,20\nBob,18\n")
        path = f.name
    try:
        docs = load_document("csv", path)
        assert len(docs) == 2
        assert docs[0].metadata["source_type"] == "csv"
        assert docs[0].metadata["row"] == 1
        assert docs[1].metadata["row"] == 2
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Web loader
# ---------------------------------------------------------------------------

def test_load_web_invalid_url():
    with pytest.raises(DocumentLoadError, match="Invalid URL"):
        load_document("web", "not-a-url")


# ---------------------------------------------------------------------------
# document_id
# ---------------------------------------------------------------------------

def test_document_id_is_stable_for_same_source():
    id1 = generate_document_id("pdf", "source/report.pdf")
    id2 = generate_document_id("pdf", "source/report.pdf")
    assert id1 == id2


def test_document_id_differs_for_different_sources():
    id1 = generate_document_id("pdf", "source/report.pdf")
    id2 = generate_document_id("pdf", "source/other.pdf")
    assert id1 != id2


def test_add_document_id_sets_metadata_on_all_docs():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Some content.")
        path = f.name
    try:
        docs = load_document("txt", path)
        docs = add_document_id(docs, "txt", path)
        expected_id = generate_document_id("txt", path)
        for doc in docs:
            assert doc.metadata["document_id"] == expected_id
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Multi-source pipeline
# ---------------------------------------------------------------------------

def test_load_all_sources_skips_failed_source_and_keeps_others():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Valid content.")
        good_path = f.name

    sources = [
        {"source_type": "txt", "source": good_path},
        {"source_type": "txt", "source": "source/missing.txt"},
    ]
    try:
        docs = load_all_sources(sources)
        assert len(docs) == 1
        assert docs[0].metadata["source"] == os.path.basename(good_path)
    finally:
        os.remove(good_path)


def test_load_all_sources_raises_if_everything_fails():
    sources = [{"source_type": "txt", "source": "source/missing.txt"}]
    with pytest.raises(DocumentLoadError, match="No documents were loaded"):
        load_all_sources(sources)
