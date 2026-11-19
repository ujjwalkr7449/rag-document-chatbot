from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # Fallback for PyMuPDF


def extract_text(pdf_source) -> str:
    text_parts = []

    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    elif isinstance(pdf_source, bytes):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        data = pdf_source.read() if hasattr(pdf_source, "read") else pdf_source.getvalue()
        doc = fitz.open(stream=data, filetype="pdf")

    with doc as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                text_parts.append(
                    f"[Page {page_number}]\n{page_text}"
                )

    text = "\n\n".join(text_parts)

    if not text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned or empty."
        )

    return text
