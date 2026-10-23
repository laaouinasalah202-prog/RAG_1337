from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
from typing import Dict, List
from pathlib import Path
import os
from uuid import uuid1
import tqdm

class Chunker:
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size
        self.overlap = self.max_chunk_size // 15
        self.map_chunks: list[dict] = []

    def chunker(self):
        p = Path("data/raw/vllm-0.10.1")
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            add_start_index=True,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.overlap
            )
        makrdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            add_start_index=True,
            chunk_overlap=self.overlap
            )
        files = list(p.rglob("*"))
        for pth in tqdm.tqdm(files):
            if pth.is_file() and pth.suffix.lower() == ".py":
                docs = pth.read_text()
                chunks = python_splitter.create_documents([docs])
                self.save_metadata(chunks, pth)

            elif pth.is_file():
                try:
                    docs = pth.read_text()
                    chunks = makrdown_splitter.create_documents([docs])
                    self.save_metadata(chunks, pth)
                except UnicodeDecodeError as e:
                    pass

            else:
                continue
        index_path = "data/processed/index.json"
        if os.path.exists("data/processed/") is False:
            os.makedirs("data/processed/")

        with open(index_path, "w") as f:
            json.dump(self.map_chunks, f, indent=4)
        return self.map_chunks
    def save_metadata(self, chunks, path: str):
        for chunk in chunks:
            metadata = {}
            metadata.update(
                {
                    "id": str(uuid1()),
                    "path": str(path), "first_character_index": chunk.metadata["start_index"],
                    "last_character_index": chunk.metadata["start_index"] + len(chunk.page_content),
                }
                )
            self.map_chunks.append(metadata)
