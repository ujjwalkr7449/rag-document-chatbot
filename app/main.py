
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from app.document import extract_text
from app.chunking import split_text
from app.vector_store import save_vector_store
from app.retrieval import retrieve_context
from app.qa import answer_question


app = FastAPI(title="RAG Document Chatbot")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QuestionRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {"message": "RAG Document Chatbot API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF")

    content = await file.read()

    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "Invalid PDF file")

    document_id = str(uuid4())
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(content)

    try:
        text = extract_text(str(pdf_path))
        chunks = split_text(text)

        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["source"] = file.filename

        save_vector_store(document_id, chunks)

    except ValueError as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc

    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            500, "Document processing failed"
        ) from exc

    return {
        "message": "PDF processed successfully",
        "document_id": document_id,
        "chunks_created": len(chunks),
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        context, sources = retrieve_context(
            request.document_id,
            request.question,
        )

        if not context.strip():
            raise HTTPException(
                404, "No relevant document content found"
            )

        answer = answer_question(
            request.question,
            context,
        )

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            404, "Document or vector store not found"
        ) from exc
