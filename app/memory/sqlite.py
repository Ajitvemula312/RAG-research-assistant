import sqlite3
from pathlib import Path


class ConversationStore:
    def __init__(self, path: str = "data/processed/conversations.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("CREATE TABLE IF NOT EXISTS messages (session_id TEXT, role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        self.connection.commit()

    def append(self, session_id: str, role: str, content: str) -> None:
        self.connection.execute("INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        self.connection.commit()

    def history(self, session_id: str) -> list[dict[str, str]]:
        rows = self.connection.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at", (session_id,)).fetchall()
        return [{"role": role, "content": content} for role, content in rows]