import os

import streamlit as st

from auth import AuthStore
from rag import RAGService

st.set_page_config(page_title="Knowledge Assistant", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_store() -> AuthStore:
    return AuthStore()


store = get_store()
st.title("🔎 Retrieval-Augmented Knowledge Assistant")
st.caption("Chat freely, or ask questions grounded in your private uploaded documents.")

if not os.getenv("GROQ_API_KEY"):
    st.error("Add GROQ_API_KEY to a .env file before using the assistant.")
    st.stop()

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    login_tab, signup_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            user = store.login(username, password)
            if user:
                st.session_state.user_id = user["id"]
                st.rerun()
            st.error("Invalid username or password.")
    with signup_tab:
        with st.form("signup"):
            username = st.text_input("Choose a username")
            password = st.text_input("Choose a password", type="password")
            confirmation = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirmation:
                st.error("Passwords do not match.")
            else:
                success, message = store.register(username, password)
                (st.success if success else st.error)(message)
    st.stop()

user = store.user(st.session_state.user_id)
if not user:
    st.session_state.user_id = None
    st.rerun()


@st.cache_resource(show_spinner=False)
def get_service() -> RAGService:
    return RAGService(store)


with st.spinner("Loading local search model…"):
    service = get_service()

with st.sidebar:
    st.header("Your account")
    st.write(f"Signed in as **{user['username']}**")
    with st.form("profile"):
        name = st.text_input("Your name", value=user["display_name"], placeholder="e.g. Disha")
        if st.form_submit_button("Save name"):
            store.update_name(user["id"], name)
            st.success("Saved.")
    if st.button("Log out"):
        st.session_state.user_id = None
        st.rerun()
    if st.button("Clear my conversation"):
        service.clear_history(user["id"])
        st.rerun()
    st.caption("Your account, chats, and documents are saved locally in data/assistant.db.")
    st.divider()
    st.header("Knowledge base")
    uploads = st.file_uploader("Documents", type=["txt", "md", "pdf", "docx"], accept_multiple_files=True)
    if st.button("Index uploaded documents", type="primary", disabled=not uploads):
        with st.spinner("Creating local embeddings…"):
            try:
                total = sum(service.ingest(user["id"], file.name, file.getvalue()) for file in uploads)
                st.success(f"Indexed {total} chunks from {len(uploads)} document(s).")
            except Exception as error:
                st.error(str(error))
    sources = service.sources(user["id"])
    st.write(f"**Indexed sources:** {len(sources)}")
    for source in sources:
        st.caption(f"• {source}")
    if st.button("Clear my knowledge base", disabled=not sources):
        service.clear_knowledge_base(user["id"])
        st.rerun()

for message in service.history(user["id"]):
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask anything, or ask about your documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, results = service.answer(user["id"], user["display_name"], question)
                st.write(answer)
                if results:
                    with st.expander("Retrieved document context"):
                        for filename, index, text, score in results:
                            st.markdown(f"**{filename} — chunk {index}** (similarity: {score:.2f})")
                            st.caption(text[:700] + ("…" if len(text) > 700 else ""))
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
