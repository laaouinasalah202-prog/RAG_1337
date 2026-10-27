from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
from pathlib import Path
import os
from uuid import uuid1
import tqdm


class Chunker:
    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = self.max_chunk_size // 20
        self.map_chunks: list[dict] = []
        self.map_file_stat: list = []
        self.root_path = Path("data/raw/vllm-0.10.1")

    def reindex_modified(self, modified_files: list) -> None:
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
        for file in modified_files:
            pth = Path(file)
            time = pth.stat().st_mtime

            self.map_file_stat.append(
                {
                    "file_path": str(pth),
                    "index_time": time,
                }
            )
            pth = Path(file)
            if pth.is_file() and pth.suffix.lower() == ".py":
                docs = pth.read_text()
                chunks = python_splitter.create_documents([docs])
                self.save_metadata(chunks, pth)

            elif pth.is_file() and any(
                    [
                        pth.suffix.lower() == ".md",
                        pth.suffix.lower() == ".txt"
                    ]):
                try:
                    docs = pth.read_text()
                    chunks = makrdown_splitter.create_documents([docs])
                    self.save_metadata(chunks, pth)
                except UnicodeDecodeError:
                    pass

            else:
                continue

        with open("data/processed/file_stat.json", "w") as f:
            json.dump(self.map_file_stat, f, indent=4)
        index_path = "data/processed/index.json"
        with open(index_path, "w") as f:
            json.dump(self.map_chunks, f, indent=4)

    def check_modified_files(self):
        all_files = (self.root_path).rglob("*")
        all_files = [str(f) for f in all_files]
        modified_files = []
        deleted_files = []
    
        for file in self.map_file_stat[1:]:
            pth = Path(file["file_path"])
            if file["file_path"] not in all_files:
                deleted_files.append(file["file_path"])
                continue
            elif pth.stat().st_mtime != file["index_time"]:
                modified_files.append(file["file_path"])

        files_to_remove = modified_files + deleted_files
        if len(files_to_remove) == 0:
            print("Indexes file already exist and no file is modified")
            return

        self.map_chunks = [chunk for chunk in self.map_chunks
                           if chunk["path"] not in files_to_remove]
        self.map_file_stat = [stat for stat in self.map_file_stat
                              if stat["file_path"] not in files_to_remove]

        if len(deleted_files) > 0:
            print(f"Deleted files: {deleted_files}")
        if len(modified_files) > 0:
            print(f"Reindixing modified files:\n{modified_files}")
            self.reindex_modified(modified_files)

    def __index_exist(self, max_chunk_size):
        if all([os.path.exists("data/processed/index.json"),
                os.path.exists("data/processed/file_stat.json")]):
            with open("data/processed/index.json", "r") as f:
                self.map_chunks = json.load(f)
            with open("data/processed/file_stat.json", "r") as f:
                self.map_file_stat = json.load(f)
            if self.map_file_stat[0] != max_chunk_size:
                return False
            self.check_modified_files()
            return True
        else:
            return False

    def chunker(self, max_chunk_size: int):
        if self.__index_exist(max_chunk_size):
            return self.map_chunks
        self.map_file_stat = []
        self.map_file_stat.append(max_chunk_size)
        print("Chunking all the data in vllm-0.10.1")
        p = self.root_path
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
            time = pth.stat().st_mtime
            self.map_file_stat.append(
                {
                    "file_path": str(pth),
                    "index_time": time,
                }
            )
            if pth.is_file() and pth.suffix.lower() == ".py":
                docs = pth.read_text()
                chunks = python_splitter.create_documents([docs])
                self.save_metadata(chunks, pth)

            elif pth.is_file() and any(
                    [pth.suffix.lower() == ".md", pth.suffix.lower() == ".txt"]
                    ):
                try:
                    docs = pth.read_text(errors="replace")
                    chunks = makrdown_splitter.create_documents([docs])
                    self.save_metadata(chunks, pth)
                except UnicodeDecodeError:
                    pass
            else:
                continue

        index_path = "data/processed/index.json"
        map_file_stat = "data/processed/file_stat.json"
        if os.path.exists("data/processed/") is False:
            os.makedirs("data/processed/")

        with open(map_file_stat, "w") as f:
            json.dump(self.map_file_stat, f, indent=4)

        with open(index_path, "w") as f:
            json.dump(self.map_chunks, f, indent=4)
        return self.map_chunks

    def save_metadata(self, chunks, path: Path):
        for chunk in chunks:
            last_endx = chunk.metadata["start_index"] + len(chunk.page_content)
            metadata = {
                    "id": str(uuid1()),
                    "path": str(path),
                    "first_character_index": chunk.metadata["start_index"],
                    "last_character_index": last_endx,
                }
            self.map_chunks.append(metadata)
