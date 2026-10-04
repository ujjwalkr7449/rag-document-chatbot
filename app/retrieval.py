from app.vector_store import load_vector_store


def retrieve_context(
    document_id: str,
    question: str,
    k: int = 4,
):
    store = load_vector_store(document_id)

    results = store.similarity_search(
        question,
        k=k,
    )

    context = "\n\n".join(
        result.page_content for result in results
    )

    sources = [
        result.metadata for result in results
    ]

    return context, sources