from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
from typing import Dict, List
from pathlib import Path
import os
from uuid import uuid1

def save_metadata(map_chunks: List, chunks, path: str):
    for chunk in chunks:
        metadata = {}
        metadata.update(
            {
                "id": str(uuid1()),
                "path": str(path), "first_character_index": chunk.metadata["start_index"],
                "last_character_index": chunk.metadata["start_index"] + len(chunk.page_content),
            }
            )
        map_chunks.append(metadata)

def chunker(chunk_size, overlap):
    p = Path("vllm-0.10.1")
    map_chunks: list[dict] = []
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, add_start_index=True, chunk_size=300, chunk_overlap=overlap)
    makrdown_splitter = RecursiveCharacterTextSplitter(chunk_size=200, add_start_index=True, chunk_overlap=3)
    
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
    index_path = "data/processed/index.json"
     
    if os.path.exists(index_path) is False:
        os.makedirs("data/processed/")

    with open(index_path, "w") as f:
        json.dump(map_chunks, f, indent=4)

    
chunker(1000, 300)
