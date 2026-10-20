from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from typing import Dict, List
from pathlib import Path


def save_metadata(map_chunks: List, chunks, path: str):
    for chunk in chunks:
        metadata = {}
        metadata.update(
            {
                "path": path.name, "first_character_index": chunk.metadata["start_index"],
                "last_character_index": chunk.metadata["start_index"] + len(chunk.page_content)
            }
            )
        map_chunks.append(metadata)

def chunker():
    p = Path("../vllm-0.10.1")
    map_chunks: list[dict] = []
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, add_start_index=True, chunk_size=200)
    makrdown_splitter = RecursiveCharacterTextSplitter(chunk_size=200, add_start_index=True)

    for pth in p.rglob("*"):
        if pth.is_file() and pth.suffix.lower() == ".py":
            docs = pth.read_text()
            chunks = python_splitter.create_documents([docs])
            save_metadata(map_chunks, chunks, pth)

        elif pth.is_file():
            try:
                docs = pth.read_text()
                chunks = makrdown_splitter.create_documents([docs])
                save_metadata(map_chunks, chunks, pth)
            except UnicodeDecodeError as e:
                print(f"{e}: skiping no utf-8 files")

        else:
            continue
    print(map_chunks[:100])
chunker()
