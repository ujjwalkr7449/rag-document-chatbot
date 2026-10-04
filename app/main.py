from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="RAG Document Chatbot")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def home():
    return {"message": "Welcome to RAG Document Chatbot"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file",
        )

    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{document_id}.pdf"

    content = await file.read()

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF",
        )

    file_path.write_bytes(content)

    return {
        "message": "PDF uploaded successfully",
        "document_id": document_id,
    }