from lib.semantic_search import Semantic_search
from lib.bm25_search import KeyWordSearch
from typing import List


class Rrf_Search:
    def __init__(self, query, limit):
        self.semantic_search = Semantic_search("all-MiniLM-L6-v2")
        self.bm25_result = KeyWordSearch()
        self.query = query
        self.limit = limit
        
    def __search(self):
        self.vector_search = self.semantic_search.semantic_search(self.query, self.limit)
        self.bm25 = self.bm25_result.bm25(self.query, self.limit)
    
    def score_normalize(self, score, max, min) -> List:
        if max - min  == 0:
            return 0
        return (score - min) / (max - min)

    def score_normalization(self):
        max_bm25 = max(self.bm25.items(), key=lambda item: item[1]["bm25_score"])[1]["bm25_score"]
        min_bm25 = max(self.bm25.items(), key=lambda item: item[1]["bm25_score"])[1]["bm25_score"]
        for _,b in self.bm25.items():
            new_score = self.score_normalize(b["bm25_score"], max_bm25, min_bm25)
            b.update({"bm25_score": new_score})

        max_vec_search = max(self.vector_search.items(), key=lambda item: item[1]["cosine_similarity"])[1]["cosine_similarity"]
        min_vec_search = max(self.vector_search.items(), key=lambda item: item[1]["cosine_similarity"])[1]["cosine_similarity"]
        for s in self.vector_search.items():
            new_score = self.score_normalize(s[1]["cosine_similarity"], max_vec_search, min_vec_search)
            s[1].update({"cosine_similarity": new_score})

    def weight_normalization(self, alpha) -> list:
        weight_combo: list[dict] = []
        i = 0
        for chunk_id, chunk_metadat in self.bm25.items():
            if self.vector_search.get(chunk_id):
                semantic_score = self.vector_search[chunk_id]["cosine_similarity"]
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
        weight_combo = sorted(weight_combo, key=lambda item: item["score"], reverse=True)
        return weight_combo

    def rrf_search(self, k: int = 60) -> List:
        self.__search()
        self.score_normalization()
        scores = self.weight_normalization(1.5)

        rrf_map: list[dict] = []

        for rank, s in enumerate(scores):
            rrf = 1 / (rank + k)
            rrf_map.append(
                {
                    "chunk_indx": s["chunk_indx"],
                    "rrf": rrf
                }
            )
        rrf_map = sorted(rrf_map, key=lambda item: item["rrf"], reverse=True)
        return rrf_map