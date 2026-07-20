from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from pathlib import Path


def chunk_file(
        text: str,
        file_path: str,
        max_chunk_size: int = 2000) -> list[dict]:
    if Path(file_path).suffix == ".py":
        return chunk_code(text, file_path, max_chunk_size)
    return chunk_text(text, file_path, max_chunk_size)


def chunk_text(
        text: str,
        file_path: str,
        max_chunk_size: int = 2000) -> list[dict]:
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append({
            "index": index,
            "file_path": file_path,
            "first_character_index": start,
            "last_character_index": end,
            "text": text[start:end],
        })
        index += 1
        start += max_chunk_size
    return chunks


def chunk_code(
        text: str,
        file_path: str,
        max_chunk_size: int = 2000) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        add_start_index=True,
    )
    index = 0
    chunks = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        chunks.append({
            "index": index,
            "file_path": file_path,
            "first_character_index": start,
            "last_character_index": start + len(doc.page_content),
            "text": doc.page_content
        })
        index += 1
    return chunks
