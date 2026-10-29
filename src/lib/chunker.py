from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
from pathlib import Path
import os
from uuid import uuid1
import tqdm


class Chunker:
    """Split Python, Markdown and text files into indexed chunks.

    Walk the ``data/raw/vllm-0.10.1`` directory, split each supported
    file into overlapping chunks, and store chunk metadata and file
    modification times in JSON files under ``data/processed``.

    Attributes:
        max_chunk_size (int): Maximum size of a chunk in characters.
        overlap (int): Overlap between chunks, 5% of the chunk size.
        map_chunks (list[dict]): Metadata of every chunk created.
        map_file_stat (list): The chunk size used, followed by one
            ``{"file_path", "index_time"}`` dict per indexed file.
        root_path (Path): Root directory of the files to chunk.
    """
    def __init__(self, max_chunk_size: int,
                 save_path: Path = Path("data/processed")) -> None:
        """Initialize the chunker.

        Args:
            max_chunk_size (int): Maximum size of a chunk in characters.
        """
        save_path.mkdir(exist_ok=True)
        self.file_stat_pat = save_path.joinpath("file_stat.json")
        self.index_path = save_path.joinpath("index.json")
        self.file_stat_pat = save_path.joinpath("file_stat.json")
        self.max_chunk_size = max_chunk_size
        self.overlap = self.max_chunk_size // 20
        self.map_chunks: list[dict] = []
        self.map_file_stat: list = []
        self.root_path = Path("data/raw/vllm-0.10.1")

    def reindex_modified(self, modified_files: list) -> None:
        """Re-chunk the given files and save the updated indexes.

        Split each ``.py``, ``.md`` and ``.txt`` file, record its new
        modification time, append the chunk metadata, and write
        ``data/processed/file_stat.json`` and
        ``data/processed/index.json``. Skip files that are not
        supported or cannot be decoded.

        Args:
            modified_files (list): Paths of the files to re-chunk.
        """
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
                docs = pth.read_text()
                chunks = makrdown_splitter.create_documents([docs])
                self.save_metadata(chunks, pth)

            else:
                continue

        with open(self.file_stat_pat, "w") as f:
            json.dump(self.map_file_stat, f, indent=4)

        with open(self.index_path, "w") as f:
            json.dump(self.map_chunks, f, indent=4)

    def check_modified_files(self) -> None:
        """Detect modified and deleted files and update the indexes.

        Compare the files on disk with the recorded modification times.
        Remove the chunks and stats of modified and deleted files, then
        re-chunk the modified files with ``reindex_modified``. Print a
        message if nothing changed.
        """

        files = (self.root_path).rglob("*")
        all_files = [str(f) for f in files]
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
        return

    def __index_exist(self, max_chunk_size: int) -> bool:
        """Load the saved indexes if they exist and are still valid.

        Read ``index.json`` and ``file_stat.json``. If the saved chunk
        size differs from ``max_chunk_size``, the indexes are unusable.
        Otherwise, refresh them with ``check_modified_files``.

        Args:
            max_chunk_size (int): Chunk size the indexes must match.

        Returns:
            bool: True if valid indexes were loaded, False otherwise.
        """
        if self.index_path.is_file() and self.file_stat_pat.is_file():
            with open(self.index_path, "r") as f:
                self.map_chunks = json.load(f)
            with open(self.file_stat_pat, "r") as f:
                self.map_file_stat = json.load(f)
            if self.map_file_stat[0] != max_chunk_size:
                return False
            self.check_modified_files()
            return True
        else:
            return False

    def chunker(self, max_chunk_size: int) -> list[dict]:
        """Return the chunk metadata, building the indexes if needed.

        Reuse the saved indexes when they exist and match
        ``max_chunk_size``. Otherwise, split every ``.py``, ``.md`` and
        ``.txt`` file under ``self.root_path`` and save the results to
        ``data/processed``.

        Args:
            max_chunk_size (int): Chunk size the indexes must match.

        Returns:
            list[dict]: Chunk metadata, each with the keys ``id``,
                ``path``, ``first_character_index`` and
                ``last_character_index``.
        """

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
                docs = pth.read_text(errors="replace")
                chunks = makrdown_splitter.create_documents([docs])
                self.save_metadata(chunks, pth)
            else:
                continue

        if os.path.exists("data/processed/") is False:
            os.makedirs("data/processed/")

        with open(self.file_stat_pat, "w") as f:
            json.dump(self.map_file_stat, f, indent=4)
        with open(self.index_path, "w") as f:
            json.dump(self.map_chunks, f, indent=4)
        return self.map_chunks

    def save_metadata(self, chunks: list, path: Path) -> None:
        """Build and store the metadata of each chunk of a file.

        Create a dict with a unique ID, the file path, and the start and
        end character indices of the chunk, and append it to
        ``self.map_chunks``.

        Args:
            chunks (list): Documents returned by the text splitter, each
                with a ``start_index`` in its metadata.
            path (Path): Path of the file the chunks come from.
        """

        for chunk in chunks:
            last_endx = chunk.metadata["start_index"] + len(chunk.page_content)
            metadata = {
                    "id": str(uuid1()),
                    "path": str(path),
                    "first_character_index": chunk.metadata["start_index"],
                    "last_character_index": last_endx,
                }
            self.map_chunks.append(metadata)
