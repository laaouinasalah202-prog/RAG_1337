from sentence_transformers import SentenceTransformer
import json
import tqdm
import numpy as np
from typing import Dict, List
import os
import pickle


class Semantic_search:
    """Rank document chunks by embedding similarity to a query.

    Load chunk metadata from ``data/processed/index.json``, embed every
    chunk with a sentence-transformer model, cache the embeddings in a
    pickle file, and score chunks against a query with cosine
    similarity.

    Attributes:
        chunks_index (list[dict]): Chunk metadata loaded from the JSON
            index file. Not set if the file does not exist.
        model_name (str): Name of the requested embedding model.
        model (SentenceTransformer): The loaded embedding model.
        path_embeddings (str): Path where the embeddings are cached.
        chunk_embedding (list): Embedding vector of each chunk.
    """

    def __init__(self, model_name):
        """Initialize the semantic search.

        Load the chunk metadata if ``data/processed/index.json`` exists,
        and load the embedding model.

        Args:
            model_name (str): Name of the embedding model.
        """

        if os.path.exists("data/processed/index.json"):
            with open("data/processed/index.json", "r") as f:
                self.chunks_index = json.load(f)
        self.model_name = model_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.path_embeddings = "data/processed/embeddings.pkl"
        self.chunk_embedding = []

    def __read_chunk(self, chunk):
        """Read the text of a single chunk from its source file.

        Args:
            chunk (dict): Chunk metadata with the keys ``path``,
                ``first_character_index`` and ``last_character_index``.

        Returns:
            str: The part of the file between the chunk's start and end
                indices.
        """
        with open(chunk["path"], "r", encoding="utf-8", errors="replace") as f:
            last = "last_character_index"
            first = "first_character_index"
            chunk_string = f.read()[chunk[first]:chunk[last]]

        return chunk_string

    def __embed_text(self, text):
        """Embed a single text with the model.

        Args:
            text (str): The text to embed.

        Returns:
            numpy.ndarray: Embedding of the text, with shape
                ``(1, embedding_size)``.
        """

        text_emb = self.model.encode([text])
        return text_emb

    def __strAll_docs(self) -> list:
        """Read the text of every chunk.

        Returns:
            list[str]: Text of each chunk, in the order of
                ``self.chunks_index``.
        """

        all_chunks = []
        for chunk in self.chunks_index:
            all_chunks.append(self.__read_chunk(chunk))
        return all_chunks

    def embed_chunks(self):
        """Embed every chunk and save the embeddings to disk.

        Read and encode each chunk, store the vectors in
        ``self.chunk_embedding``, and pickle them to
        ``self.path_embeddings``.
        """

        all_chunkstr = self.__strAll_docs()
        chunks_embeddings = []
        print("chunking with tqdm")
        for chunk in tqdm.tqdm(all_chunkstr):
            chunks_embeddings.append(self.model.encode(chunk))

        self.chunk_embedding = chunks_embeddings
        with open(self.path_embeddings, "wb") as f:
            pickle.dump(self.chunk_embedding, f)

    def cosine_similarity(self, vec1, vec2):
        """Compute the cosine similarity between two vectors.

        Args:
            vec1 (numpy.ndarray): The first vector.
            vec2 (numpy.ndarray): The second vector.

        Returns:
            float: Cosine similarity, between -1 and 1.
        """

        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return np.dot(vec1, vec2) / norm_product

    def load_or_create_embeddings(self):
        """Load the embeddings from the cache, or build them if missing.

        If ``self.path_embeddings`` exists, load the embeddings from it.
        Otherwise, build them with ``embed_chunks``.
        """

        if os.path.exists(self.path_embeddings):
            with open(self.path_embeddings, "rb") as f:
                self.chunk_embedding = pickle.load(f)
        else:
            self.embed_chunks()

    def semantic_search(self, query: str, limit: int) -> Dict:
        """Search the chunks for a query and rank them by similarity.

        Embed the query, load or build the chunk embeddings, compute the
        cosine similarity with every chunk, and keep the best matches.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.

        Returns:
            dict: Maps chunk ID to
                ``{"chunk_indx": int, "cosine_similarity": float}``,
                ordered from highest to lowest similarity, with at most
                ``limit`` entries.
        """

        query_embd = self.__embed_text(query)
        self.load_or_create_embeddings()
        map_chunks_score: List[Dict] = []

        for i, emb in enumerate(self.chunk_embedding):
            map_chunks_score.append(
                {
                    "chunk_indx": i,
                    "cosine_similarity": float(
                        self.cosine_similarity(query_embd, emb))
                }
            )
        map_chunks_score = sorted(map_chunks_score,
                                  key=lambda item: item["cosine_similarity"],
                                  reverse=True)
        with open("data/processed/index.json", "r") as f:
            doc_chunks = json.load(f)
        semantic_map = {
            doc_chunks[r["chunk_indx"]]["id"]: {
                "chunk_indx": r["chunk_indx"],
                "cosine_similarity": r["cosine_similarity"],
            }
            for r in map_chunks_score[:limit]
        }
        return semantic_map
