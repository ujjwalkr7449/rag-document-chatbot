# 🤖 RAG Document Chatbot

An intelligent RAG (Retrieval-Augmented Generation) document chatbot built with **Streamlit**, **LangChain**, **Groq (Llama-3.3-70B / Llama-3.1-8B)**, **FAISS**, and **Sentence-Transformers**. Features real-time token-by-token streaming responses and source citations.

---

## ✨ Features

- ⚡ **Real-time Token Streaming**: Word-by-word streaming responses in Streamlit chat using `st.write_stream`.
- 📄 **Interactive PDF Ingestion**: PyMuPDF extraction, chunking with overlap, and FAISS vector embeddings.
- 📚 **Source Grounding & Citations**: Direct citations and snippets from the uploaded document.
- 🚀 **1-Click Streamlit Cloud Deployment**: Self-contained architecture that runs directly on Streamlit Community Cloud without requiring a separate FastAPI server.
- 🔑 **Flexible API Key Management**: Supports Streamlit Secrets, `.env` file, or sidebar key entry.

---

## 🚀 Deployment Instructions

### Method 1: Deploy to Streamlit Community Cloud (Recommended & Free)

Streamlit Community Cloud is the easiest, official, and completely free way to host this chatbot.

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add streaming and Streamlit Cloud deployment support"
   git push origin main
   ```

2. **Open Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3. **Deploy the App**:
   - Click **"New app"**.
   - Select your repository: `rag-document-chatbot`
   - Select branch: `main`
   - Set Main file path: `streamlit_app.py`

4. **Configure Secrets (API Key)**:
   - Click **"Advanced settings..."** before deploying (or go to App Settings > Secrets afterwards).
   - In the **Secrets** text box, add your Groq API key:
     ```toml
     GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
     ```
   *(Get your free API key at [console.groq.com](https://console.groq.com/keys))*.

5. **Deploy**:
   - Click **"Deploy!"**. Streamlit Cloud will install dependencies from `requirements.txt` and launch your live app!

---

### Method 2: Run Locally

1. **Clone and setup virtual environment**:
   ```bash
   git clone <your-repo-url>
   cd rag-document-chatbot
   python -m venv venv
   
   # Windows:
   .\venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API Key**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

4. **Launch Streamlit**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Open your browser at `http://localhost:8501`.

---

### Method 3: Run with FastAPI Backend (Optional)

If you wish to run the backend as a decoupled API service:

1. **Start FastAPI**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
2. **Access Swagger Docs**:
   Visit `http://localhost:8000/docs` to test `/upload`, `/ask`, and `/ask/stream`.

---

## 📁 Project Structure

```
├── .streamlit/
│   ├── config.toml            # Streamlit theme and server configuration
│   └── secrets.toml.example   # Example secrets file for API keys
├── app/
│   ├── chunking.py            # Recursive character text splitting
│   ├── document.py            # PDF text extraction via PyMuPDF
│   ├── embeddings.py          # Sentence-transformers embedding model with LRU cache
│   ├── main.py                # FastAPI endpoints (including SSE streaming)
│   ├── qa.py                  # LangChain Groq QA chain and token streaming generator
│   ├── retrieval.py           # FAISS similarity search & source context retrieval
│   └── vector_store.py        # Vector store persistence and loading
├── requirements.txt           # Production dependencies
├── streamlit_app.py           # Streamlit frontend with direct streaming pipeline
└── README.md                  # Documentation and deployment guide
```
