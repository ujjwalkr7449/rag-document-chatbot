
from langchain_community.vectorstores import FAISS
from app.embeddings import get_embeddings


VECTOR_DIR = Path("vector_store")


def save_vector_store(document_id: str, chunks):
    embeddings = get_embeddings()

    store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    save_path = VECTOR_DIR / document_id
    save_path.mkdir(parents=True, exist_ok=True)

    store.save_local(str(save_path))

    return str(save_path)


def load_vector_store(document_id: str):
    embeddings = get_embeddings()

    save_path = VECTOR_DIR / document_id

    if not save_path.exists():
        raise FileNotFoundError("Vector store not found")

    return FAISS.load_local(
        str(save_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )