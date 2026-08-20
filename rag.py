"""Per-user document retrieval and conversational response generation."""

from __future__ import annotations

import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from docx import Document
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from auth import AuthStore

load_dotenv()


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    if suffix == ".docx":
        return "\n".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs)
    raise ValueError("Supported formats are TXT, MD, PDF, and DOCX.")


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


class RAGService:
    def __init__(self, store: AuthStore) -> None:
        self.store = store
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.encoder = SentenceTransformer(self.embedding_model)
        self.chat_model = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")

    def _embed(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    @staticmethod
    def _to_blob(vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def _from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def ingest(self, user_id: int, filename: str, content: bytes) -> int:
        parts = split_text(extract_text(filename, content))
        if not parts:
            raise ValueError(f"No readable text found in {filename}.")
        vectors = self._embed(parts)
        with self.store.connection() as db:
            document = db.execute(
                "INSERT INTO documents (user_id, filename, content, created_at) VALUES (?, ?, ?, datetime('now'))",
                (user_id, filename, content),
            )
            db.executemany(
                "INSERT INTO chunks (id, document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [(str(uuid.uuid4()), document.lastrowid, i + 1, part, self._to_blob(vectors[i])) for i, part in enumerate(parts)],
            )
        return len(parts)

    def retrieve(self, user_id: int, question: str, limit: int = 4) -> list[tuple[str, int, str, float]]:
        with self.store.connection() as db:
            rows = db.execute(
                """SELECT d.filename, c.chunk_index, c.text, c.embedding
                   FROM chunks c JOIN documents d ON d.id = c.document_id
                   WHERE d.user_id = ?""",
                (user_id,),
            ).fetchall()
        if not rows:
            return []
        vectors = np.vstack([self._from_blob(row["embedding"]) for row in rows])
        query = self._embed([question])[0]
        scores = vectors @ query / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query) + 1e-10)
        selected = np.argsort(scores)[::-1][:limit]
        return [(rows[i]["filename"], rows[i]["chunk_index"], rows[i]["text"], float(scores[i])) for i in selected]

    def history(self, user_id: int) -> list[dict[str, str]]:
        with self.store.connection() as db:
            rows = db.execute(
                "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT 40", (user_id,)
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def answer(self, user_id: int, display_name: str, question: str) -> tuple[str, list[tuple[str, int, str, float]]]:
        results = self.retrieve(user_id, question)
        relevant = [result for result in results if result[3] >= 0.30]
        context = "\n\n".join(
            f"[Source: {filename}, chunk {index}]\n{text}" for filename, index, text, _ in relevant
        ) or "No relevant uploaded-document context was found."
        instructions = (
            "You are a helpful, friendly personal knowledge assistant. Answer general questions using your own "
            "knowledge. When document context is relevant, use it as the source of truth and cite its filename in "
            "square brackets. Do not claim a document says something it does not say. "
            f"The user's saved name is {display_name or 'not known'}; use it naturally only when useful."
        )
        messages = [{"role": "system", "content": instructions}] + self.history(user_id)[-8:]
        messages.append({"role": "user", "content": f"Document context:\n{context}\n\nQuestion: {question}"})
        response = self.client.chat.completions.create(model=self.chat_model, messages=messages, temperature=0.2)
        answer = response.choices[0].message.content or "I could not generate an answer."
        with self.store.connection() as db:
            db.executemany(
                "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, datetime('now'))",
                [(user_id, "user", question), (user_id, "assistant", answer)],
            )
        return answer, relevant

    def sources(self, user_id: int) -> list[str]:
        with self.store.connection() as db:
            rows = db.execute("SELECT DISTINCT filename FROM documents WHERE user_id = ? ORDER BY filename", (user_id,)).fetchall()
        return [row["filename"] for row in rows]

    def clear_knowledge_base(self, user_id: int) -> None:
        with self.store.connection() as db:
            db.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))

    def clear_history(self, user_id: int) -> None:
        with self.store.connection() as db:
            db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
