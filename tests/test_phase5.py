from langchain_core.documents import Document

from app.api.formatting import build_sources


def test_build_sources_includes_page_when_present():
    docs = [Document(page_content="x", metadata={"source": "a.pdf", "page": 3})]
    result = build_sources(docs)
    assert result == [{"source": "a.pdf", "page": 3}]


def test_build_sources_deduplicates_same_source_and_page():
    docs = [
        Document(page_content="x", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="y", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="z", metadata={"source": "a.pdf", "page": 2}),
    ]
    result = build_sources(docs)
    assert result == [{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 2}]


def test_build_sources_handles_missing_page_for_txt_source():
    docs = [Document(page_content="x", metadata={"source": "notes.txt"})]
    result = build_sources(docs)
    assert result == [{"source": "notes.txt", "page": None}]


def test_build_sources_empty_list():
    assert build_sources([]) == []
