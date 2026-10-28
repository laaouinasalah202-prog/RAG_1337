from src.lib.tokenizer import tokenizer
import os
import pickle
import json
import math
from typing import Dict, List
from tqdm import tqdm


class KeyWordSearch:
    """Keyword search engine that ranks document chunks with BM25.

    Load chunk metadata from ``data/processed/index.json``, build (or
    load from a pickle file) an inverted index of token frequencies per
    chunk, and score chunks against a query.

    Attributes:
        inverse_index (dict): Maps each token to a dict of
            ``{chunk_id: term_frequency}``.
        doc_chunks (list[dict]): Chunk metadata loaded from the JSON
            index file.
        index_pickle_path (str): Path where the inverted index is
            cached.
        all_doc_len (int): Total length, in characters, of all chunks.
    """

    def __init__(self):

        """Initialize the search engine.

        Load the chunk metadata from ``data/processed/index.json``, set
        the path of the pickle cache, and compute the total chunk length.

        Raises:
            FileNotFoundError: If ``data/processed/index.json`` is
                missing.
            json.JSONDecodeError: If the index file is not valid JSON.
        """

        self.inverse_index = {}
        with open("data/processed/index.json", "r") as f:
            self.doc_chunks = json.load(f)
        self.index_pickle_path = "data/processed/index.pkl"
        self.all_doc_len = self._calculate_doc_lenght()

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
            start = chunk["first_character_index"]
            end = chunk["last_character_index"]
            chunk_string = f.read()[start:end]
        return chunk_string

    def _build_inverse_indx(self):
        """Build the inverted index from all chunks and save it to disk.

        Read and tokenize every chunk, count how many times each token
        appears in each chunk, store the result in ``self.inverse_index``
        and pickle it to ``self.index_pickle_path``.
        """
        for chunk in tqdm(self.doc_chunks):
            chunk_str = self.__read_chunk(chunk)
            chunk_tokenized = tokenizer(chunk_str)
            for token in chunk_tokenized:
                if self.inverse_index.get(token) is None:
                    self.inverse_index.update(
                        {
                            token: {
                                chunk["id"]: 1
                            }
                        }
                        )
                elif self.inverse_index[token].get(chunk["id"]) is None:
                    self.inverse_index[token].update(
                        {
                            chunk["id"]: 1
                        }
                    )
                else:
                    self.inverse_index[token][chunk["id"]] += 1

        with open(self.index_pickle_path, "wb") as f:
            pickle.dump(self.inverse_index, f)

    def load_or_creatIndex(self):
        """Load the inverted index from the cache, or build it if missing.

        If ``self.index_pickle_path`` exists, load the index from it.
        Otherwise, build a new index with ``_build_inverse_indx``.
        """

        if os.path.exists(self.index_pickle_path):
            with open(self.index_pickle_path, "rb") as f:
                self.inverse_index = pickle.load(f)
        else:
            print("Building the inverse index...")
            self._build_inverse_indx()

    def _calculate_doc_lenght(self) -> int:
        """Calculate the total length of all chunks.

        Returns:
            int: Sum of ``last_character_index - first_character_index``
                over all chunks.
        """
        all_docs_lenght = 0
        for chunk in self.doc_chunks:
            end = chunk["last_character_index"]
            start = chunk["first_character_index"]
            all_docs_lenght += end - start
        return all_docs_lenght

    def bm25_tf(self, token, chunk_id, chunk_len):
        """Compute the BM25 term-frequency score of a token in a chunk.

        Use ``k = 1.2`` and ``b = 0.75``, with length normalization
        against the average chunk length.

        Args:
            token (str): The query token.
            chunk_id (str | int): Identifier of the chunk being scored.
            chunk_len (int): Length of the chunk in characters.

        Returns:
            float: The BM25 TF score, or 0 if the token is not in the
                chunk.
        """
        avrg = self.all_doc_len/len(self.doc_chunks)
        b = 0.75
        k = 1.2
        bm25tf = 0
        lenght_normalization = 1 - b + b * (chunk_len/avrg)
        token_index = self.inverse_index.get(token)
        if token_index and token_index.get(chunk_id):
            tf = token_index[chunk_id]
            bm25tf = (tf * (k + 1)) / (tf + k * lenght_normalization)
        return bm25tf

    def bm25_idf(self, token):
        """Compute the BM25 inverse document frequency of a token.

        Use a base-2 logarithm: ``log2(1 + (N - n + 0.5) / (n + 0.5))``,
        where ``N`` is the number of chunks and ``n`` is the number of
        chunks containing the token.

        Args:
            token (str): The query token.

        Returns:
            float: The IDF score. A token missing from the index counts
                as appearing in zero chunks.
        """
        is_index = self.inverse_index.get(token)
        token_appearance = len(self.inverse_index[token]) if is_index else 0
        n_docs = len(self.doc_chunks)
        ratio = (n_docs - token_appearance + 0.5) / (token_appearance + 0.5)
        bm25_idf = math.log(1 + ratio, 2)
        return bm25_idf

    def bm25(self, query, limit) -> Dict:
        """Search the chunks for a query and rank results with BM25.

        Tokenize the query, load or build the inverted index, score every
        chunk by summing ``idf * tf`` over the query tokens, and keep the
        highest-scoring ones.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results to return.

        Returns:
            dict: Maps chunk ID to
                ``{"chunk_indx": int, "bm25_score": float}``, ordered
                from highest to lowest score, with at most ``limit``
                entries.
        """
        bm25_search: List[Dict] = []
        query_tokenized = tokenizer(query)
        self.load_or_creatIndex()

        for i, chunk in enumerate(self.doc_chunks):
            score = 0
            start = chunk["first_character_index"]
            end = chunk["last_character_index"]
            chunk_len = end - start
            chunk_id = chunk["id"]
            for token in query_tokenized:
                bm25_tf = self.bm25_tf(token, chunk_id, chunk_len)
                score += self.bm25_idf(token) * bm25_tf

            chunk_bm25 = {
                "chunk_indx": i,
                "bm25_score": score,
            }
            bm25_search.append(chunk_bm25)

        bm25_limit = sorted(bm25_search,
                            key=lambda item: item["bm25_score"],
                            reverse=True)
        map_bm25 = {
                    self.doc_chunks[r["chunk_indx"]]["id"]: {
                        "chunk_indx": r["chunk_indx"],
                        "bm25_score": r["bm25_score"]}
                    for r in bm25_limit[:limit]
            }

        return map_bm25
