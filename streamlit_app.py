import requests
import streamlit as st


# -----------------------------
# Configuration
# -----------------------------

API_URL = "http://127.0.0.1:8000"


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="wide",
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .answer-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f5f7fa;
            border: 1px solid #ddd;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Session State
# -----------------------------

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="maigitn-title">🤖 RAG Document Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions about its content'
    '</div>',
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:

        if st.button(
            "🚀 Process PDF",
            use_container_width=True,
        ):

            with st.spinner("Processing document..."):

                try:

                    response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=120,
                    )

                    if response.status_code == 200:

                        data = response.json()

                        st.session_state.document_id = (
                            data["document_id"]
                        )

                        st.session_state.messages = []

                        st.success(
                            "PDF processed successfully!"
                        )

                        st.info(
                            f"Chunks created: "
                            f"{data['chunks_created']}"
                        )

                    else:

                        st.error(
                            response.text
                        )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "FastAPI backend is not running."
                    )

    st.divider()

    if st.session_state.document_id:

        st.success("✅ Document ready")

    else:

        st.warning(
            "Upload and process a PDF first."
        )
        

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])