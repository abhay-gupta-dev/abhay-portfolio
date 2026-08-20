# Retrieval-Augmented Knowledge Assistant

A conversational RAG app that answers general questions and questions about documents you upload. It supports **TXT, Markdown, PDF, and DOCX** files, remembers your saved name and recent conversation locally, and shows retrieved source passages when documents are used. Groq generates answers; embeddings run locally on your computer.

## Accounts and database

The first screen lets people create a username and password, then log in. Passwords are stored as secure bcrypt hashes, never as plain text. Each account has a separate document collection and chat history. All user records, documents, chunk embeddings, and messages are persisted in the local SQLite database at `data/assistant.db`.

This local setup is appropriate for learning and development. Before making it public, move the database to a managed service, serve it over HTTPS, add email verification/password reset, protect against brute-force attempts, and manage secrets outside local files.

## Run it

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`, then set `GROQ_API_KEY` to your Groq API key. The first app start downloads the free local embedding model.

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

5. In the browser, upload documents, click **Index uploaded documents**, and ask a question.

## Project layout

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface: upload, index, chat, and source display. |
| `rag.py` | Text extraction, chunking, embeddings, vector search, and grounded response generation. |
| `data/` | Created at runtime; stores local vectors and chunk metadata. Ignored by Git. |
| `tests/test_rag.py` | Basic test for the chunking behavior. |

## How this RAG pipeline works

```
Documents → extract text → split into overlapping chunks → embeddings → local vector store
Question  → embedding → similarity search ───────────────────────────────┘
                                      ↓
                         relevant chunks + question → LLM answer + citations
```

- **Ingestion:** Each document becomes small, overlapping sections. Overlap preserves a sentence or idea that crosses a chunk boundary.
- **Embeddings:** Each chunk and question becomes a list of numbers that represents semantic meaning. This project uses the free local `all-MiniLM-L6-v2` model by default.
- **Retrieval:** The app compares the question embedding against all document embeddings using cosine similarity and selects the four closest sections.
- **Generation:** The selected sections are sent with the question to the model, with an instruction to answer only from that evidence. This reduces hallucination and makes answers traceable.

## Conversation memory

The app remembers the name you save and the last 20 question-and-answer pairs in `data/conversation.json`, on your own computer. Use **Clear conversation** in the sidebar whenever you want to remove it. This is simple short-term memory, not a private cloud account or long-term personal profile.

RAG does **not** retrain the model. It gives a general-purpose model fresh, relevant context at question time. Quality depends mostly on clean source documents, sensible chunking, strong retrieval, and strict answer instructions.

## Configuration

Set these optional variables in `.env`:

```env
CHAT_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Changing the embedding model requires clearing the local knowledge base because different models can produce vectors with different dimensions.

## Notes

- Documents, embeddings, and vector search are stored locally in `data/`. Retrieved chunks are sent to Groq only to generate an answer.
- Scanned PDFs without embedded text need OCR before this app can use them.
- This is an educational starter. For production, add authentication, per-user collections, access controls, a managed vector database, ingestion queues, evaluations, and monitoring.
