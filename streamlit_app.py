import os
import json
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

# Import RAG pipeline components for direct in-process execution
from app.document import extract_text
from app.chunking import split_text
from app.vector_store import save_vector_store
from app.retrieval import retrieve_context
from app.qa import stream_answer, answer_question, DEFAULT_MODEL


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .source-card {
            background-color: #F8FAFC;
            border-left: 3px solid #6366F1;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 0.9rem;
        }
        .source-title {
            font-weight: 600;
            color: #4338CA;
            margin-bottom: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session State Initialization
# -----------------------------
if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Helper Functions
# -----------------------------
def get_configured_api_key() -> str:
    """Check secrets, env vars, or user input for the Groq API key."""
    # 1. Streamlit Cloud Secrets
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # 2. Local Environment Variable (.env)
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        return env_key

    return ""


def process_pdf_direct(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """Process uploaded PDF into vector database directly."""
    document_id = str(uuid4())
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = upload_dir / f"{document_id}.pdf"
    pdf_path.write_bytes(file_bytes)

    text = extract_text(file_bytes)
    chunks = split_text(text)

    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = filename

    save_vector_store(document_id, chunks)
    return document_id, len(chunks)


def render_sources(sources: list):
    """Render retrieved sources in an interactive expander."""
    if not sources:
        return
    with st.expander("📚 Retrieved Sources & Citations", expanded=False):
        for idx, src in enumerate(sources, 1):
            source_file = src.get("source", "Document")
            snippet = src.get("snippet", "")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">Source {idx} — {source_file}</div>
                    {f'<div>{snippet}</div>' if snippet else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    # API Key Handling
    detected_key = get_configured_api_key()
    if detected_key:
        groq_api_key = detected_key
        st.success("✅ Groq API Key connected")
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free API key at https://console.groq.com",
        )
        if not groq_api_key:
            st.warning("⚠️ Enter your Groq API Key to chat.")

    # Model Selection
    selected_model = st.selectbox(
        "Groq Model",
        options=[
            "openai/gpt-oss-120b"
        ],
        index=0,
        help="llama-3.3-70b provides high reasoning quality; llama-3.1-8b is ultra-fast.",
    )

    st.divider()

    # Document Upload Section
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF document to query with RAG",
    )

    if uploaded_file is not None:
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Analyzing document and creating vector embeddings..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    doc_id, chunk_count = process_pdf_direct(
                        file_bytes, uploaded_file.name
                    )

                    st.session_state.document_id = doc_id
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success(f"PDF processed! ({chunk_count} chunks)")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")

    # Document Status and Actions
    if st.session_state.document_id:
        st.info(f"Active: **{st.session_state.document_name}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🔄 Reset Doc", use_container_width=True):
                st.session_state.document_id = None
                st.session_state.document_name = None
                st.session_state.messages = []
                st.rerun()
    else:
        st.caption("Upload and process a PDF above to start questioning.")


# -----------------------------
# Main Chat Header
# -----------------------------
st.markdown('<div class="main-header">🤖 RAG Document Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Chat with your PDF documents with real-time streaming responses powered by Groq & FAISS</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Welcome / Empty State
# -----------------------------
if not st.session_state.document_id:
    st.info(
        "👋 **Welcome!** To get started:\n\n"
        "1. Make sure your **Groq API Key** is configured in the sidebar or via secrets.\n"
        "2. Upload a **PDF document** in the sidebar and click **🚀 Process PDF**.\n"
        "3. Ask any questions below and get streaming answers grounded in your document!"
    )


# -----------------------------
# Chat History Replay
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


# -----------------------------
# Chat Input & Streaming Execution
# -----------------------------
user_question = st.chat_input("Ask a question about your PDF...")

if user_question:
    # Validation checks
    if not groq_api_key:
        st.error("Please provide a Groq API key in the sidebar to chat.")
        st.stop()

    if not st.session_state.document_id:
        st.warning("Please upload and process a PDF document first.")
        st.stop()

    # Append & display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_question,
    })
    with st.chat_message("user"):
        st.markdown(user_question)

    # Stream Assistant Response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching document context..."):
                context, sources = retrieve_context(
                    st.session_state.document_id,
                    user_question,
                    k=4,
                )

            if not context.strip():
                no_context_msg = "I could not find any relevant information in this document."
                st.markdown(no_context_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": no_context_msg,
                })
            else:
                # Stream the answer token by token using st.write_stream
                token_stream = stream_answer(
                    question=user_question,
                    context=context,
                    api_key=groq_api_key,
                    model=selected_model,
                )

                full_response = st.write_stream(token_stream)

                # Render sources
                if sources:
                    render_sources(sources)

                # Save assistant response to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

        except Exception as exc:
            st.error(f"Error generating answer: {exc}")
