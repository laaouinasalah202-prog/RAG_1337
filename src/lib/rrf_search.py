from src.lib.bm25_search import KeyWordSearch
from typing import List
from src.lib.semantic_search import Semantic_search


class Rrf_Search:
    """Rank document chunks by combining BM25 and semantic search.

    Run a keyword search and a vector search for the same query,
    normalize both scores, blend them with a weight, and rank the
    chunks with a reciprocal rank formula.

    Attributes:
        semantic_search: Embedding model exposing a ``semantic_search``
            method.
        bm25_result (KeyWordSearch): Keyword search engine.
        query (str): The search query.
        limit (int): Maximum number of results for each search.
        vector_search (dict): Semantic results, set by ``__search``.
        bm25 (dict): BM25 results, set by ``__search``.
    """

    def __init__(self, query: str, limit: int,
                 embedding_model: Semantic_search) -> None:
        """Initialize the hybrid search.

        Args:
            query (str): The search query.
            limit (int): Maximum number of results for each search.
            embedding_model: Embedding model exposing a
                ``semantic_search(query, limit)`` method.
        """

        self.semantic_search = embedding_model
        self.bm25_result = KeyWordSearch()
        self.query = query
        self.limit = limit

    def __search(self) -> None:
        """Run the semantic search and the BM25 search.

        Store the results in ``self.vector_search`` and ``self.bm25``.
        """
        self.vector_search = self.semantic_search.semantic_search(
            self.query, self.limit)
        self.bm25 = self.bm25_result.bm25(self.query, self.limit)

    def score_normalize(self, score: float, max: float, min: float) -> float:
        """Scale a score to the range 0 to 1 using min-max scaling.

        Args:
            score (float): The score to normalize.
            max (float): The highest score of the result set.
            min (float): The lowest score of the result set.

        Returns:
            float: The normalized score, or 0 if ``max`` equals ``min``.
        """

        if max - min == 0:
            return 0
        return (score - min) / (max - min)

    def score_normalization(self) -> None:
        """Normalize the BM25 and cosine similarity scores in place.

        Apply min-max scaling separately to ``self.bm25`` and
        ``self.vector_search``, so both scores lie between 0 and 1.
        """

        max_bm25 = max(self.bm25.items(),
                       key=lambda item: item[1]["bm25_score"])[1]["bm25_score"]
        min_bm25 = min(self.bm25.items(),
                       key=lambda item: item[1]["bm25_score"])[1]["bm25_score"]

        for _, b in self.bm25.items():
            new_score = self.score_normalize(b["bm25_score"],
                                             max_bm25, min_bm25)
            b.update({"bm25_score": new_score})
        max_vec_search = max(self.vector_search.items(),
                             key=lambda item: item[1][
                                 "cosine_similarity"])[1]["cosine_similarity"]
        min_vec_search = min(self.vector_search.items(),
                             key=lambda item: item[1][
                                 "cosine_similarity"])[1]["cosine_similarity"]

        for s in self.vector_search.items():
            new_score = self.score_normalize(s[1]["cosine_similarity"],
                                             max_vec_search, min_vec_search)
            s[1].update({"cosine_similarity": new_score})

    def weight_normalization(self, alpha: float) -> list:
        """Combine the BM25 and semantic scores with a weight.

        For each chunk found by BM25, compute
        ``alpha * bm25 + (1 - alpha) * semantic``. A chunk missing from
        the semantic results gets a semantic score of 0.

        Args:
            alpha (float): Weight of the BM25 score, between 0 and 1.

        Returns:
            list[dict]: Dicts with the keys ``chunk_indx``, ``score``,
                ``bm25_score`` and ``semantic_score``, sorted from
                highest to lowest ``score``.
        """

        weight_combo: list[dict] = []
        for chunk_id, chunk_metadat in self.bm25.items():
            if self.vector_search.get(chunk_id):
                semantic_score = self.vector_search[chunk_id][
                    "cosine_similarity"]
            else:
                semantic_score = 0
            bm25_score = chunk_metadat["bm25_score"]

            new_score = alpha * bm25_score + (1 - alpha) * semantic_score
            weight_combo.append(
                {
                    "chunk_indx": chunk_metadat["chunk_indx"],
                    "score": new_score,
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score
                }
                )
        weight_combo = sorted(weight_combo,
                              key=lambda item: item["score"],
                              reverse=True)
        return weight_combo

    def rrf_search(self, k: int = 60) -> List:
        """Search and rank chunks with reciprocal rank fusion.

        Run both searches, normalize the scores, blend them with a BM25
        weight of 0.7, and give each chunk the score
        ``1 / (rank + k)``.

        Args:
            k (int): Smoothing constant of the rank formula. Defaults
                to 60.

        Returns:
            list: Dicts with the keys ``chunk_indx`` and ``rrf``,
                sorted from highest to lowest ``rrf``.
        """

        self.__search()
        self.score_normalization()

        scores = self.weight_normalization(0.7)

        rrf_map: list[dict] = []

        for rank, s in enumerate(scores, 1):
            rrf = 1 / (rank + k)
            rrf_map.append(
                {
                    "chunk_indx": s["chunk_indx"],
                    "rrf": rrf
                }
            )

        rrf_map = sorted(rrf_map, key=lambda item: item["rrf"], reverse=True)
        return rrf_map
