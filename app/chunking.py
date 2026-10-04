from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.create_documents([text])