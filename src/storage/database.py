import sqlite3
import os
import contextlib
import logging
from src.config.settings import DB_PATH

log = logging.getLogger("storage.database")

def init_db():
    """Initializes the database schema if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS call_sessions (
                session_id TEXT PRIMARY KEY,
                caller_number TEXT,
                call_start_time REAL,
                call_end_time REAL,
                call_duration_s REAL,
                call_status TEXT,
                summary TEXT,
                sentiment TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                speaker TEXT,
                message TEXT,
                turn_index INTEGER,
                timestamp REAL,
                FOREIGN KEY(session_id) REFERENCES call_sessions(session_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session_turn ON conversations(session_id, turn_index)")
        log.info(f"Database initialized at {DB_PATH}")

@contextlib.contextmanager
def get_connection():
    """Provides a SQLite connection with automatic commit/rollback."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        log.error(f"Database error: {e}")
        raise
    finally:
        conn.close()
