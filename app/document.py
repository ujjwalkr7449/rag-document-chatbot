import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    text_parts = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                text_parts.append(
                    f"[Page {page_number}]\n{page_text}"
                )

    text = "\n\n".join(text_parts)

    if not text.strip():
        raise ValueError(
            "No readable text found. The PDF may be scanned."
        )

    return text