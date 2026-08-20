"""Local user authentication and persistent storage for the knowledge assistant."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import bcrypt

DATA_DIR = Path("data")
DATABASE_PATH = DATA_DIR / "assistant.db"


class AuthStore:
    def __init__(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_tables(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash BLOB NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    content BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def register(self, username: str, password: str) -> tuple[bool, str]:
        username = username.strip()
        if len(username) < 3:
            return False, "Username must have at least 3 characters."
        if len(password) < 8:
            return False, "Password must have at least 8 characters."
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, self._now()),
                )
            return True, "Account created. You can now log in."
        except sqlite3.IntegrityError:
            return False, "That username is already in use."

    def login(self, username: str, password: str) -> sqlite3.Row | None:
        with self._connect() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
        if user and bcrypt.checkpw(password.encode(), user["password_hash"]):
            return user
        return None

    def user(self, user_id: int) -> sqlite3.Row | None:
        with self._connect() as db:
            return db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def update_name(self, user_id: int, name: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE users SET display_name = ? WHERE id = ?", (name.strip(), user_id))

    def connection(self) -> sqlite3.Connection:
        return self._connect()
