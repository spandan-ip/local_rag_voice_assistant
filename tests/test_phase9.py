from unittest.mock import patch

from langchain_core.documents import Document

from app.generation.relevance_filter import filter_relevant_chunks
from app.generation.query_rewriter import rewrite_query
from app.history.session_store import create_session, save_message, get_session_messages, rename_session, list_sessions


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

def test_filter_keeps_chunk_that_overlaps_with_answer():
    chunks = [
        Document(page_content="The company offers 20 days of annual leave per year.", metadata={"source": "policy.pdf"}),
    ]
    answer = "The company provides 20 days of annual leave per year."
    result = filter_relevant_chunks(chunks, answer)
    assert len(result) == 1


def test_filter_falls_back_when_the_only_chunk_is_unrelated():
    # A lone irrelevant chunk still gets kept -- dropping it would leave
    # zero sources, which the fallback rule specifically avoids.
    chunks = [
        Document(page_content="The office kitchen is on the third floor.", metadata={"source": "facilities.pdf"}),
    ]
    answer = "The company provides 20 days of annual leave per year."
    result = filter_relevant_chunks(chunks, answer)
    assert result == chunks


def test_filter_keeps_relevant_and_drops_irrelevant_from_mixed_set():
    relevant = Document(page_content="Annual leave is 20 days per year for all employees.", metadata={"source": "policy.pdf"})
    irrelevant = Document(page_content="The parking garage closes at midnight on weekends.", metadata={"source": "facilities.pdf"})
    answer = "Employees get 20 days of annual leave per year."

    result = filter_relevant_chunks([relevant, irrelevant], answer)

    sources = {d.metadata["source"] for d in result}
    assert "policy.pdf" in sources
    assert "facilities.pdf" not in sources


def test_filter_falls_back_to_original_when_nothing_matches():
    chunks = [Document(page_content="Completely unrelated sentence about weather.", metadata={"source": "x.pdf"})]
    answer = "This talks about something totally different entirely."
    result = filter_relevant_chunks(chunks, answer)
    # nothing overlaps -> fall back to original list rather than showing zero sources
    assert result == chunks


def test_filter_handles_empty_answer():
    chunks = [Document(page_content="Some text.", metadata={"source": "x.pdf"})]
    assert filter_relevant_chunks(chunks, "") == chunks


# ---------------------------------------------------------------------------
# Query rewriter (fallback behavior, no real Ollama call)
# ---------------------------------------------------------------------------

def test_rewrite_query_falls_back_to_original_on_llm_failure():
    with patch("app.generation.query_rewriter.get_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.side_effect = Exception("Ollama not running")
        result = rewrite_query("what is the leave policy?")
    assert result == "what is the leave policy?"


def test_rewrite_query_uses_llm_output_on_success():
    with patch("app.generation.query_rewriter.get_llm") as mock_get_llm:
        mock_get_llm.return_value.invoke.return_value = "annual leave policy days"
        result = rewrite_query("how many days off do I get?")
    assert result == "annual leave policy days"


def test_rewrite_query_empty_question_returns_unchanged():
    assert rewrite_query("") == ""
    assert rewrite_query("   ") == "   "


# ---------------------------------------------------------------------------
# Session titles
# ---------------------------------------------------------------------------

def test_first_user_message_becomes_session_title(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    save_message(session_id, "user", "What is the leave policy for new hires?", db_path=db_path)

    sessions = list_sessions(db_path=db_path)
    match = next(s for s in sessions if s["session_id"] == session_id)
    assert match["title"] == "What is the leave policy for new hires?"


def test_second_user_message_does_not_overwrite_title(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    save_message(session_id, "user", "First question", db_path=db_path)
    save_message(session_id, "assistant", "First answer", db_path=db_path)
    save_message(session_id, "user", "Second question", db_path=db_path)

    sessions = list_sessions(db_path=db_path)
    match = next(s for s in sessions if s["session_id"] == session_id)
    assert match["title"] == "First question"


def test_rename_session_overrides_auto_title(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)
    save_message(session_id, "user", "Auto title from question", db_path=db_path)

    rename_session(session_id, "My custom name", db_path=db_path)

    sessions = list_sessions(db_path=db_path)
    match = next(s for s in sessions if s["session_id"] == session_id)
    assert match["title"] == "My custom name"


def test_long_title_gets_truncated(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)
    long_question = "A" * 200

    save_message(session_id, "user", long_question, db_path=db_path)

    sessions = list_sessions(db_path=db_path)
    match = next(s for s in sessions if s["session_id"] == session_id)
    assert len(match["title"]) == 60
