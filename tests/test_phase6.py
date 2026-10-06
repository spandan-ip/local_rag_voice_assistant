from app.history.session_store import (
    create_session,
    save_message,
    get_session_messages,
    list_sessions,
    get_recent_context,
)


def test_create_session_returns_unique_ids(tmp_path):
    db_path = str(tmp_path / "test.db")
    id1 = create_session(db_path=db_path)
    id2 = create_session(db_path=db_path)
    assert id1 != id2


def test_save_and_get_messages_preserves_order(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    save_message(session_id, "user", "What is the leave policy?", db_path=db_path)
    save_message(
        session_id,
        "assistant",
        "20 days per year.",
        sources=[{"source": "policy.pdf", "page": 4}],
        db_path=db_path,
    )

    messages = get_session_messages(session_id, db_path=db_path)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the leave policy?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["sources"] == [{"source": "policy.pdf", "page": 4}]


def test_messages_from_different_sessions_do_not_mix(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_a = create_session(db_path=db_path)
    session_b = create_session(db_path=db_path)

    save_message(session_a, "user", "Question in session A", db_path=db_path)
    save_message(session_b, "user", "Question in session B", db_path=db_path)

    messages_a = get_session_messages(session_a, db_path=db_path)
    messages_b = get_session_messages(session_b, db_path=db_path)

    assert len(messages_a) == 1
    assert len(messages_b) == 1
    assert messages_a[0]["content"] == "Question in session A"
    assert messages_b[0]["content"] == "Question in session B"


def test_list_sessions_returns_all_created_sessions(tmp_path):
    db_path = str(tmp_path / "test.db")
    id1 = create_session(db_path=db_path)
    id2 = create_session(db_path=db_path)

    sessions = list_sessions(db_path=db_path)
    session_ids = {s["session_id"] for s in sessions}

    assert id1 in session_ids
    assert id2 in session_ids


def test_get_recent_context_formats_as_qa_pairs(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    save_message(session_id, "user", "How many leave days?", db_path=db_path)
    save_message(session_id, "assistant", "20 days.", db_path=db_path)

    context = get_recent_context(session_id, max_turns=3, db_path=db_path)

    assert "Q: How many leave days?" in context
    assert "A: 20 days." in context


def test_get_recent_context_limits_to_max_turns(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    for i in range(5):
        save_message(session_id, "user", f"Question {i}", db_path=db_path)
        save_message(session_id, "assistant", f"Answer {i}", db_path=db_path)

    context = get_recent_context(session_id, max_turns=2, db_path=db_path)

    # only the last 2 turns (4 messages) should appear
    assert "Question 0" not in context
    assert "Question 3" in context
    assert "Question 4" in context


def test_get_recent_context_empty_session_returns_empty_string(tmp_path):
    db_path = str(tmp_path / "test.db")
    session_id = create_session(db_path=db_path)

    assert get_recent_context(session_id, db_path=db_path) == ""
