import json
import uuid
from datetime import datetime, timezone

from app.config import CHAT_HISTORY_DB_PATH
from app.history.db import get_connection, init_db


def create_session(db_path: str = CHAT_HISTORY_DB_PATH) -> str:
    init_db(db_path)
    session_id = str(uuid.uuid4())[:8]

    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
        (session_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return session_id


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list | None = None,
    db_path: str = CHAT_HISTORY_DB_PATH,
) -> None:
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            session_id,
            role,
            content,
            json.dumps(sources) if sources else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    # First user message in a session becomes its title automatically,
    # unless the user has already renamed it. Skipped for the assistant
    # role and for any later user messages in the same session.
    if role == "user":
        row = conn.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is not None and not row["title"]:
            auto_title = content.strip()[:60]
            conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (auto_title, session_id))

    conn.commit()
    conn.close()


def rename_session(session_id: str, title: str, db_path: str = CHAT_HISTORY_DB_PATH) -> None:
    conn = get_connection(db_path)
    conn.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (title.strip()[:60], session_id))
    conn.commit()
    conn.close()


def get_session_messages(session_id: str, db_path: str = CHAT_HISTORY_DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def list_sessions(db_path: str = CHAT_HISTORY_DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT session_id, created_at, title FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"session_id": row["session_id"], "created_at": row["created_at"], "title": row["title"]}
        for row in rows
    ]


def get_recent_context(session_id: str, max_turns: int = 3, db_path: str = CHAT_HISTORY_DB_PATH) -> str:
    """Return the last `max_turns` Q&A pairs as a short text block, for
    feeding back into the LLM prompt so it can understand follow-up
    questions ("what about the second one?").
    """
    messages = get_session_messages(session_id, db_path)
    recent = messages[-(max_turns * 2):]  # each turn = 1 user + 1 assistant message

    if not recent:
        return ""

    lines = []
    for msg in recent:
        prefix = "Q" if msg["role"] == "user" else "A"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)
